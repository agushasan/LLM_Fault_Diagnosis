"""
Python port of AXKFAutoShipDynamics.m

Original research code by Agus Hasan.

Adaptive Extended Kalman Filter (AXKF) for joint state and actuator-fault
estimation on a 3-DOF surface-vessel manoeuvring model. The simulation
injects piece-wise constant changes in the actuator-fault parameter
theta = (theta_u, theta_r) and reconstructs them from pose measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Reproducibility
np.random.seed(0)

# ---------------------------------------------------------------------------
# Time horizon
# ---------------------------------------------------------------------------
tf = 20.0
dt = 0.001
N  = int(tf / dt)
t  = np.arange(1, N + 1) * dt   # MATLAB: t = dt:dt:tf

# ---------------------------------------------------------------------------
# System description
# ---------------------------------------------------------------------------
m   = 23.8
Iz  = 1.76
xg  = 0.046
Xud = -2.0
Yvd = -10.0
Yrd = 0.0
Nvd = 0.0
Nrd = -1.0

A   = np.eye(6)
B_t = np.array([[1.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0]])

M = np.array([[m - Xud, 0.0,           0.0          ],
              [0.0,     m - Yvd,       m * xg - Yrd ],
              [0.0,     m * xg - Nvd,  Iz - Nrd     ]])

# B = dt * [zeros(3,2); inv(M) * B_t]   (6 x 2)
B = dt * np.vstack([np.zeros((3, 2)), inv(M) @ B_t])

C = np.eye(6)
Q = np.eye(6)

# Process / measurement noise covariances (adaptive, will be updated online)
QF = 0.01 * np.eye(6)
RF = 0.04 * np.eye(6)

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
x        = np.zeros(6)        # true state
xbar     = np.zeros(6)        # nonlinear adaptive observer state
xhat     = np.zeros(6)        # AXKF state estimate
theta    = np.zeros(2)        # true fault
thetabar = np.zeros(2)        # observer fault
thetahat = np.zeros(2)        # AXKF fault estimate
temp     = np.zeros(6)
Pplus    = np.eye(6)

# Damping coefficients
Xu  = -0.7225
Xuu = -1.3274
Yv  = -0.8612
Yvv = -36.2823
Yr  =  0.1079
Nv  =  0.1052
Nr  = -0.5
Nrr = -1.0

# Pull mass-matrix entries used inside the loop
m11 = M[0, 0]
m22 = M[1, 1]
m23 = M[1, 2]
m32 = M[2, 1]
m33 = M[2, 2]

# Control input
u = np.array([3.0, 1.5])

# Initial Psi = -B * diag(u)   (6 x 2)
Psi = -B @ np.diag(u)

# RLS-with-forgetting state
S            = 0.1 * np.eye(2)
UpsilonPlus  = np.zeros_like(B)   # 6 x 2
lam          = 0.995
a_alpha      = 0.999

# Observer gain (paper-prescribed L)
L = np.array([
    [ 0.0750,  0.0000, -0.0001, -0.0002, -0.0003,  0.0000],
    [ 0.0000,  0.0750,  0.0000,  0.0003, -0.0002,  0.0000],
    [-0.0001,  0.0000,  0.0750,  0.0000,  0.0000,  0.0002],
    [-0.0002,  0.0003,  0.0000,  0.0750,  0.0000,  0.0000],
    [-0.0003, -0.0002,  0.0000,  0.0000,  0.0750, -0.0009],
    [ 0.0000,  0.0000,  0.0002,  0.0000, -0.0009,  0.0750],
])

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
uArray         = np.zeros((2, N))
xArray         = np.zeros((6, N))
xbarArray      = np.zeros((6, N))
xhatArray      = np.zeros((6, N))
thetaArray     = np.zeros((2, N))
thetabarArray  = np.zeros((2, N))
thetahatArray  = np.zeros((2, N))

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
for i in range(N):
    # MATLAB uses 1-based "i>5000" tests; replicate with the corresponding
    # 0-based equivalent so the switching instants match wall-clock time.
    i_matlab = i + 1

    Psi = -B @ np.diag(u)

    if i_matlab > 5000:
        u = np.array([2.0, 3.0])

    if i_matlab > 5000:
        theta = np.array([0.0, 0.25])

    if i_matlab > 10000:
        theta = np.array([0.3, 0.25])

    if i_matlab > 12000:
        u = np.array([1.0, 1.0])

    if i_matlab > 15000:
        theta = np.array([0.0, 0.141])

    # Log
    uArray[:, i]        = u
    xArray[:, i]        = x
    xhatArray[:, i]     = xhat
    xbarArray[:, i]     = xbar
    thetaArray[:, i]    = theta
    thetabarArray[:, i] = thetabar
    thetahatArray[:, i] = thetahat

    # Coriolis / damping at the true state
    c13 = -m22 * x[4] - ((m23 + m32) / 2.0) * x[5]
    c23 =  m11 * x[3]
    Cv = np.array([[ 0.0, 0.0,  c13],
                   [ 0.0, 0.0,  c23],
                   [-c13, -c23, 0.0]])

    Dv = -np.array([
        [Xu + Xuu * abs(x[3]),       0.0,                       0.0],
        [0.0,                         Yv + Yvv * abs(x[4]),     Yr],
        [0.0,                         Nv,                        Nr + Nrr * abs(x[5])],
    ])

    # True dynamics
    drift_pose = np.array([
        np.cos(x[2]) * x[3] - np.sin(x[2]) * x[4],
        np.sin(x[2]) * x[3] + np.cos(x[2]) * x[4],
        x[5],
    ])
    drift_vel = -inv(M) @ (Cv + Dv) @ x[3:6]
    drift_full = np.concatenate([drift_pose, drift_vel])

    x = (A @ x
         + dt * drift_full
         + B @ u
         + Psi @ theta
         + QF @ (dt * np.random.randn(6)))

    y = C @ x + RF @ (dt * np.random.randn(6))

    # ----- Nonlinear adaptive observer ------------------------------------
    drift_obs_pose = np.array([
        np.cos(y[2]) * y[3] - np.sin(y[2]) * y[4],
        np.sin(y[2]) * y[3] + np.cos(y[2]) * y[4],
        y[5],
    ])
    drift_obs_vel = -inv(M) @ (Cv + Dv) @ y[3:6]
    drift_obs_full = np.concatenate([drift_obs_pose, drift_obs_vel])

    xbar = (A @ xbar
            + dt * drift_obs_full
            + B @ u
            + Psi @ thetabar
            + L @ (y - C @ xbar))

    thetabar = thetabar + 2.0 * Psi.T @ inv(Psi @ Psi.T + Q) @ ((x - xbar) - (A - L) @ temp)
    temp = x - xbar

    # ----- Adaptive Extended Kalman filter --------------------------------
    # Jacobian FX of the one-step nominal map about xbar
    s3 = np.sin(xbar[2]); c3 = np.cos(xbar[2])
    Jpose = np.array([
        [0.0, 0.0, -s3 * xbar[3] - c3 * xbar[4],  c3, -s3, 0.0],
        [0.0, 0.0,  c3 * xbar[3] - s3 * xbar[4],  s3,  c3, 0.0],
        [0.0, 0.0,  0.0,                          0.0, 0.0, 1.0],
    ])

    # Velocity-block Jacobian (note: MATLAB has a leading minus on the whole block)
    Jvel = -inv(M) @ np.array([
        [0.0, 0.0, 0.0,
         Xu + 2.0 * Xuu * abs(xbar[3]),
         -m22 * xbar[5],
         -m22 * xbar[4] - (m23 + m32) * xbar[5]],
        [0.0, 0.0, 0.0,
         Yr * xbar[5] + m11 * xbar[5],
         Yv + 2.0 * Yvv * abs(xbar[4]),
         Yr + m11 * xbar[3]],
        [0.0, 0.0, 0.0,
         m22 * xbar[4] + ((m23 + m32) / 2.0) * xbar[5] - m11 * xbar[4],
         m22 * xbar[3] + Nv - m11 * xbar[3],
         ((m23 + m32) / 2.0) * xbar[3] + Nr + 2.0 * Nrr * abs(xbar[5])],
    ])

    FX = A + dt * np.vstack([Jpose, Jvel])

    Pmin  = FX @ Pplus @ FX.T + QF
    Sigma = C @ Pmin @ C.T + RF
    KF    = Pmin @ C.T @ inv(Sigma)
    Pplus = (np.eye(6) - KF @ C) @ Pmin

    ytilde = y - C @ xhat
    QF = a_alpha * QF + (1.0 - a_alpha) * (KF @ np.outer(ytilde, ytilde) @ KF.T)
    RF = a_alpha * RF + (1.0 - a_alpha) * (np.outer(ytilde, ytilde) + C @ Pmin @ C.T)

    Upsilon = (np.eye(6) - KF @ C) @ FX @ UpsilonPlus + (np.eye(6) - KF @ C) @ Psi
    Omega   = C @ FX @ UpsilonPlus + C @ Psi
    Lambda  = inv(lam * Sigma + Omega @ S @ Omega.T)
    Gamma   = S @ Omega.T @ Lambda
    S       = (1.0 / lam) * S - (1.0 / lam) * S @ Omega.T @ Lambda @ Omega @ S
    UpsilonPlus = Upsilon

    thetahat = thetahat + Gamma @ (y - C @ xhat)

    # State update
    drift_hat_pose = np.array([
        np.cos(xbar[2]) * xbar[3] - np.sin(xbar[2]) * xbar[4],
        np.sin(xbar[2]) * xbar[3] + np.cos(xbar[2]) * xbar[4],
        xbar[5],
    ])
    drift_hat_vel = -inv(M) @ (Cv + Dv) @ xbar[3:6]
    drift_hat_full = np.concatenate([drift_hat_pose, drift_hat_vel])

    xhat = (A @ xhat
            + dt * drift_hat_full
            + B @ u
            + Psi @ thetahat
            + KF @ (y - C @ xhat)
            + Upsilon @ Gamma @ (y - C @ xhat))

# ---------------------------------------------------------------------------
# Plotting (three figures matching the MATLAB layout)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.4,
})

# ---- Figure 1: trajectory + per-axis error
fig1 = plt.figure(figsize=(14, 8))

ax_traj = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax_traj.plot(xArray[0], xArray[1],    'k-',  lw=3, label='true')
ax_traj.plot(xhatArray[0], xhatArray[1], 'r:', lw=3, label='estimated')
ax_traj.plot(xArray[0, 0], xArray[1, 0], 'o', mec='k', mfc='none', ms=12, mew=2, label='start')
ax_traj.plot(xhatArray[0, -1], xhatArray[1, -1], 'o', mec='r', mfc='none', ms=12, mew=2, label='end')
ax_traj.set_xlabel(r'$x$ (m)')
ax_traj.set_ylabel(r'$y$ (m)')
ax_traj.legend(loc='best')
ax_traj.set_title('Trajectory in NED frame')

ax_ex = plt.subplot2grid((2, 2), (0, 1))
ax_ex.plot(t, xArray[0] - xhatArray[0], 'k-', lw=2)
ax_ex.axvline(10, color='b', ls=':', lw=2)
ax_ex.axvline(15, color='b', ls=':', lw=2)
ax_ex.text(12, ax_ex.get_ylim()[1] * 0.5, 'fault')
ax_ex.set_ylabel(r'error $x$')

ax_eu = plt.subplot2grid((2, 2), (1, 1))
ax_eu.plot(t, xArray[3] - xhatArray[3], 'k-', lw=2)
ax_eu.axvline(5, color='b', ls=':', lw=2)
ax_eu.axvline(15, color='b', ls=':', lw=2)
ax_eu.text(12, ax_eu.get_ylim()[1] * 0.5, 'fault')
ax_eu.set_ylabel(r'error $u$')   # MATLAB labelled this "y" but plots index 4 (u)
ax_eu.set_xlabel(r'$t$ (s)')

fig1.tight_layout()
fig1.savefig('fig1_trajectory.png', dpi=130)

# ---- Figure 2: control inputs
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(t, uArray[0], 'm:', lw=2, label=r'$\tau_u$')
ax2.plot(t, uArray[1], 'c:', lw=2, label=r'$\tau_r$')
ax2.set_xlabel(r'$t$ (s)')
ax2.set_ylabel(r'$u$')
ax2.legend()
fig2.tight_layout()
fig2.savefig('fig2_inputs.png', dpi=130)

# ---- Figure 3: fault estimates
fig3, axes3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes3[0].plot(t, thetaArray[0],     'k-',  lw=2.5, label=r'true $\theta$')
axes3[0].plot(t, thetahatArray[0],  'g:',  lw=2.5, label=r'estimated $\theta$')
axes3[0].set_ylabel(r'$\theta_u$')
axes3[0].set_ylim([-0.05, 0.35])
axes3[0].legend(loc='upper right')

axes3[1].plot(t, thetaArray[1],     'k-',  lw=2.5)
axes3[1].plot(t, thetahatArray[1],  'g:',  lw=2.5)
axes3[1].set_ylabel(r'$\theta_r$')
axes3[1].set_xlabel(r'$t$ (s)')
axes3[1].set_ylim([-0.05, 0.30])

fig3.tight_layout()
fig3.savefig('fig3_theta.png', dpi=130)

# Summary numbers worth printing
print(f"Simulated {N} steps over {tf} s with dt = {dt} s")
print(f"Final theta_hat = {thetahatArray[:, -1]}")
print(f"Final theta_true = {thetaArray[:, -1]}")
print(f"Trajectory range: x in [{xArray[0].min():.3f}, {xArray[0].max():.3f}] m, "
      f"y in [{xArray[1].min():.3f}, {xArray[1].max():.3f}] m")

plt.show()
