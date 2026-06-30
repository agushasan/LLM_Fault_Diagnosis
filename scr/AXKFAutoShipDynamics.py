"""
AKF Auto Ship Dynamics - Realistic Operational Scenario
========================================================

Adaptive Kalman Filter (AKF) for joint state and actuator-fault
estimation on the Otter, a 3-DOF under-actuated autonomous catamaran.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from collections import deque

np.random.seed(42)

# ===========================================================================
# 1. Configuration
# ===========================================================================
SIM_DURATION = 60.0
DT           = 1.0e-2
N_STEPS      = int(SIM_DURATION / DT)
TIME         = np.arange(1, N_STEPS + 1) * DT

M_RIGID = 23.8
I_ZZ    = 1.76
X_G     = 0.046

X_UD, Y_VD, Y_RD = -2.0, -10.0, 0.0
N_VD, N_RD       =  0.0,  -1.0

X_U, X_UU = -0.7225, -1.3274
Y_V, Y_VV = -0.8612, -36.2823
Y_R       =  0.1079
N_V       =  0.1052
N_R, N_RR = -0.5,    -1.0

M_MAT = np.array([
    [M_RIGID - X_UD, 0.0,                       0.0],
    [0.0,            M_RIGID - Y_VD,            M_RIGID * X_G - Y_RD],
    [0.0,            M_RIGID * X_G - N_VD,      I_ZZ - N_RD],
])
M_INV = inv(M_MAT)

B_TAU = np.array([
    [1.0, 0.0],
    [0.0, 0.0],
    [0.0, 1.0],
])
B_MAT = DT * np.vstack([np.zeros((3, 2)), M_INV @ B_TAU])

C_MAT = np.hstack([np.eye(3), np.zeros((3, 3))])

R_TRUE = np.diag([0.25, 0.25, (np.pi / 180.0) ** 2])

SIGMA_POS, SIGMA_PSI       = 0.01,  0.005
SIGMA_VEL, SIGMA_YAWRATE   = 0.05,  0.01
Q_DIAG = np.array([
    SIGMA_POS**2, SIGMA_POS**2, SIGMA_PSI**2,
    SIGMA_VEL**2, SIGMA_VEL**2, SIGMA_YAWRATE**2,
])
Q_TRUE = np.diag(Q_DIAG) * DT
Q_FILTER = 4.0 * Q_TRUE
R_FILTER = 1.0 * R_TRUE

CURRENT_NED = np.array([0.15, 0.10])

# ===========================================================================
# 2. Mission profile
# ===========================================================================
MISSION_LEGS = [
    (10.0, 0.80,   0.0, "Leg 1: north"),
    (15.0, 0.60,  90.0, "Turn east"),
    (25.0, 0.80,  90.0, "Leg 2: east"),
    (30.0, 0.60, 180.0, "Turn south"),
    (45.0, 0.80, 180.0, "Leg 3: south"),
    (50.0, 0.60, 270.0, "Turn west"),
    (60.0, 0.80, 270.0, "Leg 4: west"),
]


def reference_setpoint(t: float) -> tuple[float, float]:
    for t_end, u_set, psi_deg, _ in MISSION_LEGS:
        if t <= t_end:
            return u_set, np.deg2rad(psi_deg)
    t_end, u_set, psi_deg, _ = MISSION_LEGS[-1]
    return u_set, np.deg2rad(psi_deg)


# ===========================================================================
# 3. Fault scenario
# ===========================================================================
FAULT_EVENTS = [
    (25.0, "yaw wear onset",       "ramp"),
    (30.0, "yaw wear settles",     "settle"),
    (45.0, "surge debris strike",  "step"),
    (55.0, "yaw partial recovery", "step"),
]


def fault_profile(t: float) -> np.ndarray:
    theta_u, theta_r = 0.0, 0.0
    if 25.0 <= t < 30.0:
        theta_r = 0.25 * (t - 25.0) / 5.0
    elif t >= 30.0:
        theta_r = 0.25
    if t >= 45.0:
        theta_u = 0.55
    if t >= 55.0:
        theta_r = 0.10
    return np.array([theta_u, theta_r])


# ===========================================================================
# 4. Helpers
# ===========================================================================
def rotation_matrix(psi: float) -> np.ndarray:
    cp, sp = np.cos(psi), np.sin(psi)
    return np.array([
        [cp, -sp, 0.0],
        [sp,  cp, 0.0],
        [0.0, 0.0, 1.0],
    ])


def coriolis(nu: np.ndarray) -> np.ndarray:
    m11, m22 = M_MAT[0, 0], M_MAT[1, 1]
    m23, m32 = M_MAT[1, 2], M_MAT[2, 1]
    u, v, r = nu
    c13 = -m22 * v - 0.5 * (m23 + m32) * r
    c23 =  m11 * u
    return np.array([
        [ 0.0,  0.0, c13],
        [ 0.0,  0.0, c23],
        [-c13, -c23, 0.0],
    ])


def damping(nu: np.ndarray) -> np.ndarray:
    u, v, r = nu
    d11 = -X_U - X_UU * abs(u)
    d22 = -Y_V - Y_VV * abs(v)
    d23 = -Y_R
    d32 = -N_V
    d33 = -N_R - N_RR * abs(r)
    return np.array([
        [d11, 0.0, 0.0],
        [0.0, d22, d23],
        [0.0, d32, d33],
    ])


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def vessel_dynamics(eta, nu, tau, theta):
    R = rotation_matrix(eta[2])
    V_c_body = R.T @ np.array([CURRENT_NED[0], CURRENT_NED[1], 0.0])
    nu_r = nu - V_c_body
    tau_eff = (1.0 - theta) * tau
    F_hydro = -(coriolis(nu_r) + damping(nu_r)) @ nu_r
    F_input = B_TAU @ tau_eff
    return R @ nu, M_INV @ (F_hydro + F_input)


def jacobian_dynamics(xhat: np.ndarray) -> np.ndarray:
    psi = xhat[2]
    u, v, r = xhat[3:6]
    cp, sp = np.cos(psi), np.sin(psi)
    Jpose = np.array([
        [0.0, 0.0, -sp * u - cp * v,  cp, -sp, 0.0],
        [0.0, 0.0,  cp * u - sp * v,  sp,  cp, 0.0],
        [0.0, 0.0,                0.0, 0.0,  0.0, 1.0],
    ])
    m11, m22 = M_MAT[0, 0], M_MAT[1, 1]
    m23, m32 = M_MAT[1, 2], M_MAT[2, 1]
    J_nu = -M_INV @ np.array([
        [0.0, 0.0, 0.0,
         -X_U - 2.0 * X_UU * abs(u),
         -m22 * r,
         -m22 * v - (m23 + m32) * r],
        [0.0, 0.0, 0.0,
          Y_R * r + m11 * r,
         -Y_V - 2.0 * Y_VV * abs(v),
          Y_R + m11 * u],
        [0.0, 0.0, 0.0,
          m22 * v + 0.5 * (m23 + m32) * r - m11 * v,
          m22 * u + N_V - m11 * u,
          0.5 * (m23 + m32) * u - N_R + 2.0 * N_RR * abs(r)],
    ])
    return np.eye(6) + DT * np.vstack([Jpose, J_nu])


# ===========================================================================
# 5. Autopilot
# ===========================================================================
class Autopilot:
    K_P_U, K_I_U = 8.0, 0.6
    K_P_R, K_D_R = 4.0, 2.0
    TAU_U_LIMIT  = (-3.0, 6.0)
    TAU_R_LIMIT  = (-4.0, 4.0)

    def __init__(self) -> None:
        self.eu_int = 0.0

    def step(self, u_set, psi_set, eta, nu):
        err_u   = u_set - nu[0]
        self.eu_int = np.clip(self.eu_int + err_u * DT, -2.0, 2.0)
        err_psi = wrap_angle(psi_set - eta[2])
        tau_u = self.K_P_U * err_u + self.K_I_U * self.eu_int
        tau_r = self.K_P_R * err_psi - self.K_D_R * nu[2]
        return np.array([
            np.clip(tau_u, *self.TAU_U_LIMIT),
            np.clip(tau_r, *self.TAU_R_LIMIT),
        ])


# ===========================================================================
# 6. Main simulation loop
# ===========================================================================
print(f"Running {N_STEPS:,}-step simulation over {SIM_DURATION:.0f} s "
      f"at {1.0/DT:.0f} Hz ...")

x        = np.zeros(6)
xhat     = np.zeros(6)
thetahat = np.zeros(2)
P_plus   = 0.1 * np.eye(6)
Gamma    = 0.01 * np.eye(2)
UpsilonPlus = np.zeros((6, 2))
LAMBDA   = 0.995

ap = Autopilot()

log_x         = np.zeros((6, N_STEPS))
log_xhat      = np.zeros((6, N_STEPS))
log_y         = np.zeros((3, N_STEPS))
log_tau       = np.zeros((2, N_STEPS))
log_theta     = np.zeros((2, N_STEPS))
log_thetahat  = np.zeros((2, N_STEPS))
log_theta_std = np.zeros((2, N_STEPS))
log_residual  = np.zeros((3, N_STEPS))
log_nis       = np.zeros(N_STEPS)
log_nis_win   = np.zeros(N_STEPS)
log_u_set     = np.zeros(N_STEPS)
log_psi_set   = np.zeros(N_STEPS)

NIS_WIN  = 100
NIS_BUF  = deque(maxlen=NIS_WIN)
WIN_THRESHOLD = 3.0 + 5.0 * np.sqrt(6.0 / NIS_WIN)
PER_SAMPLE_THRESHOLD = 16.27

PCD_SMOOTH    = 500
PCD_LAG       = 1000
DTHETA_THRESHOLD = 0.15
THETA_BUF = deque(maxlen=PCD_LAG + PCD_SMOOTH)

detection_events: list[tuple[float, float, str]] = []
last_detect_t   = -10.0

L_R_TRUE = np.linalg.cholesky(R_TRUE)
SIG_PROC = np.sqrt(Q_DIAG * DT)

for k in range(N_STEPS):
    t = TIME[k]
    u_set, psi_set = reference_setpoint(t)
    theta_true     = fault_profile(t)

    tau = ap.step(u_set, psi_set, xhat[:3], xhat[3:])

    log_x[:, k]         = x
    log_xhat[:, k]      = xhat
    log_tau[:, k]       = tau
    log_theta[:, k]     = theta_true
    log_thetahat[:, k]  = thetahat
    log_theta_std[:, k] = np.sqrt(np.diag(Gamma))
    log_u_set[k]        = u_set
    log_psi_set[k]      = psi_set

    eta_dot, nu_dot = vessel_dynamics(x[:3], x[3:], tau, theta_true)
    x = x + DT * np.concatenate([eta_dot, nu_dot]) + SIG_PROC * np.random.randn(6)
    y = C_MAT @ x + L_R_TRUE @ np.random.randn(3)
    log_y[:, k] = y

    # AKF predict
    F_k     = jacobian_dynamics(xhat)
    Phi_k   = -B_MAT @ np.diag(tau)
    P_minus = F_k @ P_plus @ F_k.T + Q_FILTER

    eta_dot_h, nu_dot_h = vessel_dynamics(xhat[:3], xhat[3:], tau, thetahat)
    x_predict = xhat + DT * np.concatenate([eta_dot_h, nu_dot_h])
    y_predict = C_MAT @ x_predict
    y_tilde   = y - y_predict

    Sigma_k = C_MAT @ P_minus @ C_MAT.T + R_FILTER
    K_k     = P_minus @ C_MAT.T @ inv(Sigma_k)
    P_plus  = (np.eye(6) - K_k @ C_MAT) @ P_minus

    nis = float(y_tilde @ inv(Sigma_k) @ y_tilde)
    log_residual[:, k] = y_tilde
    log_nis[k] = nis

    NIS_BUF.append(nis)
    nis_windowed   = np.mean(NIS_BUF) if len(NIS_BUF) == NIS_WIN else 3.0
    log_nis_win[k] = nis_windowed

    THETA_BUF.append(thetahat.copy())
    if (t - last_detect_t) > 4.0 and len(THETA_BUF) == THETA_BUF.maxlen:
        recent  = np.mean(np.array(list(THETA_BUF))[-PCD_SMOOTH:], axis=0)
        earlier = np.mean(np.array(list(THETA_BUF))[:PCD_SMOOTH], axis=0)
        dtheta  = recent - earlier
        sigma_theta = np.sqrt(np.diag(Gamma))
        z_score = np.abs(dtheta) / (sigma_theta + 1e-8)
        if np.max(z_score) > 3.0:
            channel = "surge" if z_score[0] > z_score[1] else "yaw"
            detection_events.append((t, float(np.max(z_score)),
                                     f"parameter_change_{channel}"))
            last_detect_t = t

    if (t - last_detect_t) > 0.8:
        if nis > PER_SAMPLE_THRESHOLD:
            detection_events.append((t, nis, "residual_spike"))
            last_detect_t = t
        elif nis_windowed > WIN_THRESHOLD:
            detection_events.append((t, nis_windowed, "windowed_NIS"))
            last_detect_t = t

    Upsilon = (np.eye(6) - K_k @ C_MAT) @ F_k @ UpsilonPlus \
             + (np.eye(6) - K_k @ C_MAT) @ Phi_k
    Omega        = C_MAT @ F_k @ UpsilonPlus + C_MAT @ Phi_k
    Lambda_mat   = LAMBDA * Sigma_k + Omega @ Gamma @ Omega.T
    Lambda_inv   = inv(Lambda_mat)
    Theta_gain   = Gamma @ Omega.T @ Lambda_inv

    Gamma = (Gamma - Gamma @ Omega.T @ Lambda_inv @ Omega @ Gamma) / LAMBDA
    UpsilonPlus = Upsilon

    thetahat = thetahat + Theta_gain @ y_tilde
    xhat     = x_predict + K_k @ y_tilde + Upsilon @ Theta_gain @ y_tilde

rms_pos = np.sqrt(np.mean((log_x[0] - log_xhat[0])**2 + (log_x[1] - log_xhat[1])**2))
rms_psi = np.sqrt(np.mean((log_x[2] - log_xhat[2])**2)) * 180.0 / np.pi

print(f"  RMS position-tracking error : {rms_pos:.3f} m")
print(f"  RMS heading-tracking error  : {rms_psi:.3f} deg")
print(f"  Final theta_hat             : ({log_thetahat[0,-1]:+.4f}, {log_thetahat[1,-1]:+.4f})")
print(f"  Final theta_true            : ({log_theta[0,-1]:+.4f}, {log_theta[1,-1]:+.4f})")
print(f"  Detection events            : {len(detection_events)}")
for (te, st, ty) in detection_events:
    print(f"      t = {te:5.2f} s    type = {ty:<24s}    statistic = {st:6.3f}")

print()
print("Dialogue-time snapshots:")
for t_q in [35.0, 40.0, 50.0, 58.0]:
    k_q = int(t_q / DT) - 1
    th_true = log_theta[:, k_q]
    th_hat  = log_thetahat[:, k_q]
    th_std  = log_theta_std[:, k_q]
    print(f"  t = {t_q:5.1f} s:  true=({th_true[0]:+.4f}, {th_true[1]:+.4f})  "
          f"hat=({th_hat[0]:+.4f}, {th_hat[1]:+.4f})  "
          f"std=({th_std[0]:.4f}, {th_std[1]:.4f})")

# ===========================================================================
# 7. Plotting
# ===========================================================================
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset":   "cm",
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     9,
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "0.7",
    "axes.grid":           True,
    "grid.color":          "0.92",
    "grid.linewidth":      0.5,
    "axes.edgecolor":      "0.35",
    "axes.linewidth":      0.7,
    "xtick.major.width":   0.7,
    "ytick.major.width":   0.7,
    "lines.linewidth":     1.4,
    "savefig.bbox":        "tight",
    "savefig.dpi":         200,
})

C_TRUE   = "#1a1a1a"
C_EST    = "#c0392b"
C_SET    = "#2c5f8d"
C_BAND   = "#c0392b"
C_FAULT  = "#7d3c98"
C_LEG    = "#1e8449"
C_DET    = "#d35400"

LW_TRUE = 1.6
LW_EST  = 1.3
LW_SET  = 1.0

OUT = "/mnt/user-data/outputs"


def shade_legs(ax, alpha=0.06):
    bounds = [0.0] + [leg[0] for leg in MISSION_LEGS]
    for i in range(len(MISSION_LEGS)):
        if i % 2 == 0:
            ax.axvspan(bounds[i], bounds[i + 1], color=C_LEG, alpha=alpha, zorder=0)


def mark_faults(ax):
    for t_e, _, _ in FAULT_EVENTS:
        ax.axvline(t_e, color=C_FAULT, ls=(0, (4, 2)), lw=0.9, alpha=0.65, zorder=1)


# ----- Figure: Mission trajectory (paper fig4) -----
fig1, ax = plt.subplots(figsize=(6.6, 6.0))
ax.plot(log_x[1], log_x[0], color=C_TRUE, lw=LW_TRUE, label="True trajectory")
ax.plot(log_xhat[1], log_xhat[0], color=C_EST, lw=LW_EST,
        linestyle=(0, (5, 2)), label="AKF estimate")
ax.scatter([log_x[1, 0]],  [log_x[0, 0]],  s=80, marker="o",
           facecolor="white", edgecolor=C_TRUE, linewidth=1.6, zorder=5, label="Start")
ax.scatter([log_x[1, -1]], [log_x[0, -1]], s=80, marker="s",
           facecolor="white", edgecolor=C_TRUE, linewidth=1.6, zorder=5, label="End")
for t_e, label, _ in FAULT_EVENTS:
    k_e = int(t_e / DT) - 1
    ax.scatter([log_x[1, k_e]], [log_x[0, k_e]], s=110, marker="X",
               color=C_FAULT, edgecolor="white", linewidth=0.8, zorder=6)
    ax.annotate(f"t = {t_e:.0f} s\n{label}",
                xy=(log_x[1, k_e], log_x[0, k_e]),
                xytext=(log_x[1, k_e] + 0.7, log_x[0, k_e] + 0.9),
                fontsize=8, color=C_FAULT, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=C_FAULT, lw=0.5, alpha=0.6))
ax.set_xlabel("East $y$  (m)")
ax.set_ylabel("North $x$  (m)")
ax.set_title("Otter mission trajectory in the NED frame", pad=10)
ax.set_aspect("equal", adjustable="datalim")
ax.legend(loc="lower right", frameon=True)
xl, yl = ax.get_xlim(), ax.get_ylim()
arrow_origin = (xl[0] + 0.07 * (xl[1] - xl[0]), yl[0] + 0.08 * (yl[1] - yl[0]))
arrow_tip    = (arrow_origin[0] + CURRENT_NED[1] * 8,
                arrow_origin[1] + CURRENT_NED[0] * 8)
ax.annotate("", xy=arrow_tip, xytext=arrow_origin,
            arrowprops=dict(arrowstyle="->", color=C_SET, lw=1.4))
ax.text(arrow_origin[0], arrow_origin[1] - 0.7,
        f"current  {np.linalg.norm(CURRENT_NED):.2f} m/s",
        fontsize=8, color=C_SET, ha="left")
fig1.savefig(f"{OUT}/fig4.pdf"); fig1.savefig(f"{OUT}/fig4.png")
plt.close(fig1)

# ----- Figure: Control inputs (paper fig3) -----
fig2, axes = plt.subplots(2, 1, figsize=(7.6, 4.4), sharex=True)
axes[0].plot(TIME, log_tau[0], color=C_TRUE, lw=LW_TRUE)
axes[0].set_ylabel(r"Surge thrust $\tau_u$ [N]")
axes[0].set_title("Autopilot commands during the 60-second mission", pad=8)
shade_legs(axes[0]); mark_faults(axes[0])
axes[0].annotate("post-fault compensation\n(autopilot saturated)",
                 xy=(47, 5.5), xytext=(51, 4.0),
                 fontsize=8, color=C_FAULT, ha="left",
                 arrowprops=dict(arrowstyle="->", color=C_FAULT, lw=0.7, alpha=0.7))
axes[0].text(1.0, axes[0].get_ylim()[1] * 0.86, "Leg 1: north",
             fontsize=8, color=C_LEG, style="italic")
axes[1].plot(TIME, log_tau[1], color=C_TRUE, lw=LW_TRUE)
axes[1].set_ylabel(r"Yaw moment $\tau_r$  [N$\cdot$m]")
axes[1].set_xlabel(r"Time $t$  (s)")
shade_legs(axes[1]); mark_faults(axes[1])
fig2.tight_layout()
fig2.savefig(f"{OUT}/fig3.pdf"); fig2.savefig(f"{OUT}/fig3.png")
plt.close(fig2)

# ----- Figure: State tracking (paper fig6) -----
fig3, axes = plt.subplots(3, 2, figsize=(8.4, 7.0), sharex=True)
panel_defs = [
    ((0, 0), r"North $x$  (m)",         0, 1.0),
    ((1, 0), r"East $y$  (m)",          1, 1.0),
    ((2, 0), r"Heading $\psi$  (deg)",  2, 180.0 / np.pi),
    ((0, 1), r"Surge $u$  (m/s)",       3, 1.0),
    ((1, 1), r"Sway $v$  (m/s)",        4, 1.0),
    ((2, 1), r"Yaw rate $r$  (deg/s)",  5, 180.0 / np.pi),
]
for (rc, label, idx, scale) in panel_defs:
    ax = axes[rc]
    ax.plot(TIME, scale * log_x[idx],    color=C_TRUE, lw=LW_TRUE, label="True")
    ax.plot(TIME, scale * log_xhat[idx], color=C_EST,  lw=LW_EST,
            linestyle=(0, (5, 2)), label="Estimate")
    if "Surge" in label:
        ax.plot(TIME, log_u_set, color=C_SET, lw=LW_SET, linestyle=":",
                label="Setpoint", alpha=0.85)
    if "Heading" in label:
        ax.plot(TIME, scale * log_psi_set, color=C_SET, lw=LW_SET, linestyle=":",
                label="Setpoint", alpha=0.85)
    ax.set_ylabel(label)
    shade_legs(ax); mark_faults(ax)
    if rc == (0, 0):
        ax.legend(loc="lower right", frameon=True)
    if rc == (0, 1):
        ax.legend(loc="upper right", frameon=True)
axes[2, 0].set_xlabel(r"Time $t$  (s)")
axes[2, 1].set_xlabel(r"Time $t$  (s)")
fig3.suptitle("State tracking:  pose (left)  and  body-fixed velocity (right)",
              y=0.995, fontsize=11)
fig3.tight_layout()
fig3.savefig(f"{OUT}/fig6.pdf"); fig3.savefig(f"{OUT}/fig6.png")
plt.close(fig3)

# ----- Figure: Fault tracking (paper fig5) -----
fig4, axes = plt.subplots(2, 1, figsize=(7.8, 5.0), sharex=True)
for j, (ax, label, ylim) in enumerate([
    (axes[0], r"$\theta_u$  (surge channel)", (-0.10, 0.75)),
    (axes[1], r"$\theta_r$  (yaw channel)",   (-0.10, 0.40)),
]):
    band_low  = log_thetahat[j] - 3 * log_theta_std[j]
    band_high = log_thetahat[j] + 3 * log_theta_std[j]
    ax.fill_between(TIME, band_low, band_high,
                    color=C_BAND, alpha=0.15, linewidth=0,
                    label=r"$\pm 3\sigma$ envelope")
    ax.plot(TIME, log_theta[j], color=C_TRUE, lw=LW_TRUE, label=r"True $\theta$")
    ax.plot(TIME, log_thetahat[j], color=C_EST, lw=LW_EST,
            linestyle=(0, (5, 2)), label=r"Estimate $\hat\theta$")
    ax.set_ylabel(label); ax.set_ylim(ylim)
    shade_legs(ax); mark_faults(ax)
    if j == 0:
        ax.legend(loc="upper left", frameon=True)
axes[0].annotate("debris strike\n(step to 0.55)",
                 xy=(45.2, 0.55), xytext=(50, 0.65),
                 fontsize=8, color=C_FAULT, ha="left",
                 arrowprops=dict(arrowstyle="->", color=C_FAULT, lw=0.6, alpha=0.7))
axes[1].annotate("bearing wear\n(ramp 0 to 0.25)",
                 xy=(27.5, 0.18), xytext=(15.5, 0.32),
                 fontsize=8, color=C_FAULT, ha="left",
                 arrowprops=dict(arrowstyle="->", color=C_FAULT, lw=0.6, alpha=0.7))
axes[1].annotate("partial recovery",
                 xy=(55.4, 0.10), xytext=(45, -0.06),
                 fontsize=8, color=C_FAULT, ha="left",
                 arrowprops=dict(arrowstyle="->", color=C_FAULT, lw=0.6, alpha=0.7))
axes[1].set_xlabel(r"Time $t$  (s)")
fig4.suptitle(r"Actuator-fault parameter tracking with $3\sigma$ uncertainty envelopes",
              y=0.995, fontsize=11)
fig4.tight_layout()
fig4.savefig(f"{OUT}/fig5.pdf"); fig4.savefig(f"{OUT}/fig5.png")
plt.close(fig4)

# ----- Figure: Residual / NIS diagnostics (paper fig7) -----
fig5, axes = plt.subplots(2, 1, figsize=(7.8, 5.0), sharex=True,
                          gridspec_kw=dict(height_ratios=[1.4, 1.0]))
axes[0].semilogy(TIME, log_nis + 1e-3, color=C_TRUE, lw=0.6, alpha=0.30,
                 label="per-sample NIS")
axes[0].semilogy(TIME, log_nis_win + 1e-3, color=C_TRUE, lw=1.3,
                 label=f"windowed NIS  ({NIS_WIN/100:.1f} s mean)")
axes[0].axhline(PER_SAMPLE_THRESHOLD, color=C_EST, ls=(0, (5, 2)), lw=0.9,
                label=fr"per-sample threshold $\chi^2_{{3,0.999}} = {PER_SAMPLE_THRESHOLD:.2f}$")
axes[0].axhline(WIN_THRESHOLD, color=C_DET, ls=(0, (3, 1)), lw=0.9,
                label=f"windowed threshold = {WIN_THRESHOLD:.2f}")
for (t_e, stat, ty) in detection_events:
    axes[0].plot(t_e, stat, marker="v", ms=9, color=C_DET,
                 mec="white", mew=0.7, zorder=5)
axes[0].set_ylabel(r"NIS  $\Vert\tilde{\mathbf{y}}_k\Vert^2_{\Sigma_k^{-1}}$")
axes[0].set_title("Innovation diagnostics and detected events", pad=8)
axes[0].legend(loc="upper left", frameon=True, ncol=2, fontsize=8)
mark_faults(axes[0])
axes[1].plot(TIME, log_residual[0], color="#1f77b4", lw=0.7, label=r"$\tilde y_x$  (m)")
axes[1].plot(TIME, log_residual[1], color="#ff7f0e", lw=0.7, label=r"$\tilde y_y$  (m)")
axes[1].plot(TIME, log_residual[2] * 180.0 / np.pi, color="#2ca02c", lw=0.7,
             label=r"$\tilde y_\psi$  (deg)")
axes[1].set_ylabel("Innovation")
axes[1].set_xlabel(r"Time $t$  (s)")
axes[1].legend(loc="upper right", ncol=3, frameon=True)
mark_faults(axes[1])
fig5.tight_layout()
fig5.savefig(f"{OUT}/fig7.pdf"); fig5.savefig(f"{OUT}/fig7.png")
plt.close(fig5)

print()
print("Figures written to", OUT)
for fn in ("fig3", "fig4", "fig5", "fig6", "fig7"):
    print(f"  {OUT}/{fn}.pdf   {OUT}/{fn}.png")
