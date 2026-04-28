# LLMs in Fault Diagnosis

The system performs actuator-fault diagnosis on the **Otter**, a small
under-actuated autonomous surface catamaran. A recursive joint state-and-fault
estimator runs at the feature level and tracks the actuator-fault parameter
$\theta = (\theta_u, \theta_r)^\intercal$ together with its covariance
$\Gamma_k$. A locally-served LLM runs at the decision level and answers an
operator's natural-language queries by orchestrating a small typed library of
deterministic tools that read the estimator state. Every numerical token in
the LLM's reply is, by construction, a verbatim copy of a value returned by
one of those tools.

## Architecture in one diagram

```
┌────────────────────────── Decision level ──────────────────────────┐
│  Operator (Gradio UI, app.py)                                      │
│        │  query q_k                              response r_k      │
│        ▼                                              ▲            │
│  LLM policy π_LLM   (Qwen3 8B served locally via Ollama)           │
│        │  call (i, α)                            return v          │
│        ▼                                              ▲            │
│  Tool library T  (tools.py)                                        │
│   five typed deterministic functions of the shared state Ξ_k       │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ reads only
                               ▼
┌────────────────── Shared state Ξ_k = (Y_k, s_k, H_k) ──────────────┐
│  raw window Y_k │ indicator vector s_k │ event log H_k             │
└──────────────────────────────▲─────────────────────────────────────┘
                               │
┌────────────────────────── Feature level ───────────────────────────┐
│  Behaviour model M_θ (estimator.py)                                │
│   Recursive joint state-and-fault estimator                        │
│     – Kalman filter for the state x_k                              │
│     – RLS-with-forgetting for the fault parameter θ                │
│   Otter manoeuvring model with GNSS/IMU pose measurement           │
└────────────────────────────────────────────────────────────────────┘
```

## How the LLM produces grounded answers

The LLM never sees raw measurements. It sees the operator's question and the
descriptions of five tools, picks one or more to call, and composes its reply
out of the values they return:

```
User:  "Has there been an actuator fault in the last ten seconds, and
        if so on which channel and how severe is it?"
  → LLM emits a call to query_history(Δk = 10000)
        and to get_fault_estimate()
  → Tools return: { events at t = 10s, t = 15s }
                  ( θ̂_u = 0.27 ± 0.02 ,  θ̂_r = 0.23 ± 0.02 )
  → LLM weaves the values into prose:
    "Yes. Two residual anomalies were registered at t = 10s and t = 15s.
     The current actuator-fault estimate is θ̂_u = 0.27 ± 0.02 on the
     surge channel and θ̂_r = 0.23 ± 0.02 on the yaw channel ..."
```

This is not a stylistic preference. The architecture inherits two structural
properties from the paper, which hold whenever the system prompt is respected:

- **Grounding** (Theorem 1 of the paper). Every numerical token in the
  response is computable from the underlying data through a tool in `tools.py`.
- **Consistency** (Theorem 2). Two invocations of the LLM on identical inputs
  produce identical numerical content; the surrounding prose may differ.

Both guarantees are conditional on prompt obedience, which we treat as an
empirical primitive. The repository includes a lightweight digit-matching
verifier that scans each reply against the recorded tool log.

## Why local deployment

The LLM runs locally through [Ollama](https://ollama.ai). We make this
deliberate:

- **Operational continuity.** Many of the assets condition monitoring is meant
  to serve — vessels in transit, offshore installations, remote sites — do not
  enjoy continuous broadband connectivity. A diagnostic stack that falls silent
  when the link drops is of limited use.
- **Data governance.** The tool returns include the raw measurement window,
  the estimator state, and the operator's question in cleartext. In regulated
  industrial settings, exporting any of these to a third-party endpoint is
  often contractually or legally restricted.

## Setup

### Prerequisites

- Python 3.10+
- ~6 GB disk space for the Ollama model weights

### Installation

**Linux / macOS / WSL:**

```bash
git clone <repository-url>
cd LLM-fault-diagnosis

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Install Ollama and pull Qwen3 8B

| Platform     | Installation                                               |
|--------------|------------------------------------------------------------|
| **Windows**  | Download installer from <https://ollama.ai/download>       |
| **macOS**    | `brew install ollama`                                      |
| **Linux/WSL**| `curl -fsSL https://ollama.ai/install.sh \| sh`            |

Then pull the model:

```
ollama pull qwen3:8b
```

Verify with `ollama list`. The choice of Qwen3 8B is not load-bearing for the
architecture; any instruction-tuned model with native tool-calling support can
substitute, and the propositions of the paper hold for any compliant policy.

### Run the simulation

```bash
python -m src.simulate
```

This runs the Otter actuator-fault scenario described in Section 5 of the
paper for 20 s of simulated time, injects two abrupt loss-of-effectiveness
events at $t = 10$ s and $t = 15$ s, and produces the indicator vector
$\mathbf{s}_k$ and event log $\mathcal{H}_k$ that the decision layer consumes.

### Run the operator dialogue

```bash
python -m src.app
```

Open <http://localhost:7860>.

## The Otter

The Otter is a small under-actuated autonomous surface catamaran with two
stern-mounted fixed propellers. Because it has fewer independent control
inputs than degrees of freedom, even a partial loss of effectiveness on one
propulsion channel produces non-trivial coupling between surge, sway and
yaw — which is what makes the platform a useful testbed for fault diagnosis.

| Property                | Value                                       |
|-------------------------|---------------------------------------------|
| Hull form               | Catamaran                                   |
| Actuation               | Two stern-mounted fixed propellers          |
| Control inputs          | $\tau_c = (\tau_u, \tau_r)^\intercal$    |
| Pose measurement        | GNSS/IMU integration ($x, y, \psi$)         |
| Model dimension         | $\mathbf{x} \in \mathbb{R}^6$ (pose + velocity) |
| Sample period (sim)     | $T_s = 10^{-3}$ s                           |
| Simulation length       | 20 s                                        |

The dynamic model used in the simulation is the standard three-degree-of-freedom
manoeuvring model with rotation matrix, mass matrix, Coriolis-centripetal
matrix, and damping matrix; see Section 5 of the paper for the equations and
parameter values.

## Feature-level model

The behaviour model $\mathcal{M}_{\theta}$ is the recursive joint
state-and-fault estimator of Section 3 of the paper. At each step the
estimator combines:

- a **Kalman filter** for the augmented state $\mathbf{x}_k = (\eta^\intercal,
  \nu^\intercal)^\intercal \in \mathbb{R}^6$ (pose and velocity), run as an
  extended-Kalman variant because the Otter dynamics are nonlinear; and
- a **recursive least squares with forgetting factor** $\lambda \in (0, 1]$ for
  the actuator-fault parameter $\theta = (\theta_u, \theta_r)^\intercal$,
  coupled to the state filter through a sensitivity matrix $\Pi_k$.

| Parameter / setting                | Value                            |
|------------------------------------|----------------------------------|
| State dimension $n$                | 6                                |
| Fault dimension $p$                | 2                                |
| Sample period $T_s$                | $10^{-3}$ s                      |
| Forgetting factor $\lambda$        | $0.995$                          |
| Initial state covariance           | $\mathbf{P}_{0\mid 0} = \mathbf{0}$ |
| Initial parameter covariance       | $\Gamma_0 = 0.1\,\mathbf{I}_2$ |
| Initial state-coupling gain        | $\Pi_0 = \mathbf{0}$          |

Each step writes an indicator vector to the shared state $\Xi_k$:

### Fault model

The simulation in this repository injects two abrupt loss-of-effectiveness
events during the 20 s run, the first at $t = 5$ s and the second at
$t = 10$ s, recovering partial losses in the $0.2$–$0.3$ range on each
channel.

## Tool library

Five typed deterministic functions of the shared state $\Xi_k$, called by
the LLM through Ollama's tool interface (`tools.py`). The list matches Table 1
of the paper.

| ID | Tool                        | Argument        | Returns                                                                |
|----|-----------------------------|-----------------|------------------------------------------------------------------------|
| T1 | `get_fault_estimate`        | none            | $(\hat{\theta}_k,\;\operatorname{diag}(\Gamma_k)^{1/2})$         |
| T2 | `get_state_estimate`        | none            | $(\hat{\mathbf{x}}_{k\mid k},\;\operatorname{diag}(\mathbf{P}_{k\mid k})^{1/2})$ |
| T3 | `get_residual`              | none            | $(\tilde{\mathbf{y}}_k,\;\|\tilde{\mathbf{y}}_k\|^2_{\Sigma_k^{-1}})$ |
| T4 | `evaluate_threshold`        | $\sigma > 0$    | per-component boolean significance test at threshold $\sigma$         |
| T5 | `query_history`             | $\Delta k \in \mathbb{N}$ | sublist of $\mathcal{H}_k$ over the window $[k - \Delta k,\,k]$ |

Each function is a pure deterministic map from the shared state and the
tool's argument; none of them invokes the LLM, none of them depends on hidden
state, and none of them reaches outside the host. This is what makes the
grounding and consistency propositions structural.

## Project structure

```
├── requirements.txt
├── src/
│   ├── app.py              # Gradio UI for the operator dialogue
│   ├── llm_agent.py        # Ollama LLM with tool-calling loop (decision-level policy π_LLM)
│   ├── tools.py            # Five tools T1–T5 the LLM can call
│   ├── estimator.py        # Recursive joint state-and-fault estimator (M_θ)
│   ├── otter_model.py      # Otter manoeuvring model and discretisation
│   ├── simulate.py         # 20 s simulation with two injected actuator faults
│   ├── shared_state.py     # Ξ_k = (Y_k, s_k, H_k)
│   └── verifier.py         # Digit-matching prompt-compliance verifier
└── tests/                  # Unit tests for tools and estimator
```

## Troubleshooting

| Problem                       | Solution                                              |
|-------------------------------|-------------------------------------------------------|
| `ModuleNotFoundError`         | `pip install -r requirements.txt`                     |
| `Ollama connection refused`   | `ollama serve` in another terminal                    |
| `Model not found`             | `ollama pull qwen3:8b`                                |

## Citation

## License

See `LICENSE`.
