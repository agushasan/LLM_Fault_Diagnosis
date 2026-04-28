"""
Tool library T = {T1, ..., T5} for the decision-level operator F.

Each tool is a typed deterministic function of the shared state
    Xi_k = (Y_k, s_k, H_k),
where Y_k is the raw measurement window, s_k is the indicator vector produced
by the recursive joint state-and-fault estimator M_theta, and H_k is the
event log. The signatures match Table 1 of:

    A. Hasan, "From Hallucination to Verification: A Tool-Grounded Large
    Language Model Architecture for Fault Diagnosis", 2026.

Determinism contract.
    Every tool is a *pure function* of (argument, shared_state). It must not:
      - invoke the LLM,
      - depend on hidden state outside `shared_state`,
      - reach outside the host (no network, no clock, no random number
        generator).
    The grounding and consistency propositions of the paper are conditional
    on this contract holding. New tools added to T must respect it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Shared state Xi_k
# ---------------------------------------------------------------------------

@dataclass
class SharedState:
    """The triple Xi_k = (Y_k, s_k, H_k) plus the latent fields tools read.

    Populated at every estimator step by the feature-level operator M_theta
    (see ``estimator.py``). Tools below treat this object as read-only.
    """

    # Discrete-time index k associated with this snapshot.
    k: int

    # Raw measurement window Y_k of shape (L, m), as a list of lists.
    Y: List[List[float]]

    # --- components of the indicator vector s_k (Eq. (16) of the paper) ----
    theta_hat: List[float]          # theta_hat_k in R^p
    Gamma: List[List[float]]        # Gamma_k in R^{p x p}, parameter covariance
    innovation: List[float]         # tilde y_k in R^m
    nis: float                      # ||tilde y_k||^2_{Sigma_k^{-1}}, normalised innovation squared

    # --- latent state-side quantities used by some tools --------------------
    x_hat: List[float]              # state estimate hat x_{k|k} in R^n
    P: List[List[float]]            # posterior state covariance P_{k|k}
    Sigma: List[List[float]]        # innovation covariance Sigma_k

    # --- event log H_k as an ordered list of (k_i, event_type) pairs --------
    history: List[Tuple[int, str]] = field(default_factory=list)

    # --- sampling period in seconds, used for k <-> t conversions ----------
    sample_period: float = 1e-3


# ---------------------------------------------------------------------------
# Ollama tool schemas (Table 1 of the paper)
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_fault_estimate",
            "description": (
                "Return the current actuator-fault parameter estimate "
                "theta_hat_k together with its component-wise standard "
                "deviations diag(Gamma_k)^(1/2). T1 in the paper."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state_estimate",
            "description": (
                "Return the current state estimate hat x_{k|k} together with "
                "its component-wise standard deviations diag(P_{k|k})^(1/2). "
                "T2 in the paper."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_residual",
            "description": (
                "Return the current innovation tilde y_k and the normalised "
                "innovation squared ||tilde y_k||^2_{Sigma_k^{-1}}. "
                "T3 in the paper."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_threshold",
            "description": (
                "Return the per-component significance test "
                "1{|theta_hat_k^(j)| > sigma * std_j} where std_j is the "
                "j-th component of diag(Gamma_k)^(1/2). T4 in the paper."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sigma": {
                        "type": "number",
                        "description": (
                            "Positive multiple of the per-component standard "
                            "deviation; common choice is 3 for a 3-sigma test."
                        ),
                    },
                },
                "required": ["sigma"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_history",
            "description": (
                "Return the sublist of the event log H_k whose timestamps lie "
                "in [k - Delta_k, k]. T5 in the paper."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "delta_k": {
                        "type": "integer",
                        "description": (
                            "Non-negative number of past samples to include "
                            "in the lookback window."
                        ),
                    },
                },
                "required": ["delta_k"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Dispatch a named tool against the current shared state.

    The executor holds a reference to a single ``SharedState`` instance that
    is updated externally at every estimator step. Tool calls do not mutate
    it.
    """

    def __init__(self, shared_state: SharedState):
        self.shared_state = shared_state
        self._tools: Dict[str, Callable[..., Dict[str, Any]]] = {
            "get_fault_estimate": self._get_fault_estimate,
            "get_state_estimate": self._get_state_estimate,
            "get_residual": self._get_residual,
            "evaluate_threshold": self._evaluate_threshold,
            "query_history": self._query_history,
        }

    # ----- dispatch ---------------------------------------------------------

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call ``tool_name(**arguments)`` against the current shared state."""
        if tool_name not in self._tools:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return self._tools[tool_name](**arguments)
        except TypeError as exc:
            return {"error": f"Bad arguments for {tool_name}: {exc}"}
        except Exception as exc:
            return {"error": f"Tool execution failed: {exc}"}

    # ----- T1: get_fault_estimate ------------------------------------------

    def _get_fault_estimate(self) -> Dict[str, Any]:
        """T1: theta_hat_k and diag(Gamma_k)^(1/2)."""
        s = self.shared_state
        std = [math.sqrt(s.Gamma[j][j]) for j in range(len(s.theta_hat))]
        return {
            "tool": "get_fault_estimate",
            "k": s.k,
            "t": s.k * s.sample_period,
            "theta_hat": list(s.theta_hat),
            "std": std,
        }

    # ----- T2: get_state_estimate ------------------------------------------

    def _get_state_estimate(self) -> Dict[str, Any]:
        """T2: hat x_{k|k} and diag(P_{k|k})^(1/2)."""
        s = self.shared_state
        std = [math.sqrt(s.P[i][i]) for i in range(len(s.x_hat))]
        return {
            "tool": "get_state_estimate",
            "k": s.k,
            "t": s.k * s.sample_period,
            "x_hat": list(s.x_hat),
            "std": std,
        }

    # ----- T3: get_residual ------------------------------------------------

    def _get_residual(self) -> Dict[str, Any]:
        """T3: innovation tilde y_k and normalised innovation squared."""
        s = self.shared_state
        return {
            "tool": "get_residual",
            "k": s.k,
            "t": s.k * s.sample_period,
            "innovation": list(s.innovation),
            "nis": float(s.nis),
        }

    # ----- T4: evaluate_threshold ------------------------------------------

    def _evaluate_threshold(self, sigma: float) -> Dict[str, Any]:
        """T4: 1{|theta_hat_k^(j)| > sigma * std_j} per component."""
        if not (sigma > 0):
            raise ValueError("sigma must be strictly positive")
        s = self.shared_state
        std = [math.sqrt(s.Gamma[j][j]) for j in range(len(s.theta_hat))]
        flags = [bool(abs(s.theta_hat[j]) > sigma * std[j])
                 for j in range(len(s.theta_hat))]
        return {
            "tool": "evaluate_threshold",
            "k": s.k,
            "t": s.k * s.sample_period,
            "sigma": float(sigma),
            "flags": flags,
        }

    # ----- T5: query_history -----------------------------------------------

    def _query_history(self, delta_k: int) -> Dict[str, Any]:
        """T5: events in [k - Delta_k, k]."""
        if delta_k < 0:
            raise ValueError("delta_k must be non-negative")
        s = self.shared_state
        lower = s.k - delta_k
        events = [(ki, ev) for (ki, ev) in s.history if lower <= ki <= s.k]
        return {
            "tool": "query_history",
            "k": s.k,
            "t": s.k * s.sample_period,
            "delta_k": int(delta_k),
            "events": [
                {"k": ki, "t": ki * s.sample_period, "type": ev}
                for (ki, ev) in events
            ],
        }


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _fmt_float(x: float, ndp: int = 4) -> str:
    return f"{x:.{ndp}f}"


def _fmt_vec(v: Sequence[float], ndp: int = 4) -> str:
    return "[" + ", ".join(_fmt_float(float(x), ndp) for x in v) + "]"


def format_tool_result(result: Dict[str, Any]) -> str:
    """Render a tool return value as plain text for inclusion in the LLM
    conversation. The numerical content of the returned string is the
    canonical record that the system prompt requires the LLM to copy from."""
    if "error" in result:
        return f"Error: {result['error']}"

    tool = result.get("tool", "<unknown>")
    k = result.get("k")
    t = result.get("t")
    header_bits: List[str] = [f"tool: {tool}"]
    if k is not None:
        header_bits.append(f"k = {k}")
    if t is not None:
        header_bits.append(f"t = {_fmt_float(float(t), 4)} s")
    header = ", ".join(header_bits)

    lines: List[str] = [header]

    if tool == "get_fault_estimate":
        lines.append(f"theta_hat = {_fmt_vec(result['theta_hat'])}")
        lines.append(f"std       = {_fmt_vec(result['std'])}")

    elif tool == "get_state_estimate":
        lines.append(f"x_hat = {_fmt_vec(result['x_hat'])}")
        lines.append(f"std   = {_fmt_vec(result['std'])}")

    elif tool == "get_residual":
        lines.append(f"innovation = {_fmt_vec(result['innovation'])}")
        lines.append(f"nis        = {_fmt_float(float(result['nis']))}")

    elif tool == "evaluate_threshold":
        lines.append(f"sigma = {_fmt_float(float(result['sigma']), 2)}")
        lines.append("flags = " + str(result["flags"]))

    elif tool == "query_history":
        lines.append(f"delta_k = {result['delta_k']}")
        events = result.get("events", [])
        if not events:
            lines.append("events = (none in window)")
        else:
            lines.append(f"events ({len(events)}):")
            for ev in events:
                lines.append(
                    f"  - k = {ev['k']}, t = {_fmt_float(float(ev['t']), 4)} s, type = {ev['type']}"
                )

    else:
        for key, value in result.items():
            if key in ("tool", "k", "t"):
                continue
            lines.append(f"{key}: {value}")

    return "\n".join(lines)
