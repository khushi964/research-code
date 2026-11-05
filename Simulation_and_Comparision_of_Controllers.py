"""
Week 7-8 provisional simulation for hyperchaotic Lu synchronization.

Plant: master-slave system from Wang & Ai (2022), synchronization section.
Parameters: b1=5, b2=35, b3=14, b4=3 (reported simulation example).

IMPORTANT:
- The nonlinear feedback below is a prototype, not the published Wang-Ai
  adaptive backstepping controller and not a verified implementation of the
  2026 ILF controller.
- Numerical convergence is not a fixed-time stability proof.
- The linear controller is an internal numerical benchmark, not the paper baseline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

OUT = Path("week7_week8_outputs")
OUT.mkdir(exist_ok=True)

# Model parameters reported in Wang & Ai (2022)
b1, b2, b3, b4 = 5.0, 35.0, 14.0, 3.0

# Simulation settings
T_FINAL = 10.0
RTOL = 1e-8
ATOL = 1e-10
N_PLOT = 5001
SETTLING_TOL = 1e-3
PERSISTENCE = 0.25

# Shared initial conditions for all controller comparisons
MASTER0 = np.array([1.0, 2.0, 3.0, 4.0])
ERROR0 = np.array([2.0, -1.5, 1.0, -2.0])
SLAVE0 = MASTER0 + ERROR0

# Prototype gains. Tune only after confirming numerical behaviour.
P = 0.5
Q = 1.5
GAINS_NONLINEAR = dict(k0=8.0, k1=3.0, k2=0.5,
                       k3=3.0, k4=0.5, k5=3.0, k6=0.5)
# Linear-feedback comparison gains
GAINS_LINEAR = dict(k0=8.0, d2=5.0, k3=5.0, k4=5.0)


def sigpow(x, exponent):
    """Signed power |x|^exponent sign(x), defined as zero at x=0."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.abs(x) ** exponent


def master_rhs(x):
    x1, x2, x3, x4 = x
    return np.array([
        -b1 * x2,
        b2 * (x3 - x2 + x3 * x4),
        -x2 * x4 + b3 * x3 + x1,
        x2 * x3 - b4 * x4
    ], dtype=float)


def control_law(master, slave, controller="nonlinear"):
    """
    Exact nonlinear cancellation for the actuated slave equations plus
    chosen virtual feedback v2,v3,v4.

    e = slave - master.
    u1 cancels the known terms in e2_dot and inserts -v2.
    u2 cancels the known terms in e3_dot and inserts -v3.
    u3 cancels the known terms in e4_dot and inserts -v4.
    """
    x1, x2, x3, x4 = master
    s1, s2, s3, s4 = slave
    e1, e2, e3, e4 = slave - master

    if controller == "nonlinear":
        g = GAINS_NONLINEAR
        v2 = (-g["k0"] * e1
              + g["k1"] * sigpow(e2, P)
              + g["k2"] * sigpow(e2, Q))
        v3 = g["k3"] * sigpow(e3, P) + g["k4"] * sigpow(e3, Q)
        v4 = g["k5"] * sigpow(e4, P) + g["k6"] * sigpow(e4, Q)

    elif controller == "linear":
        g = GAINS_LINEAR
        v2 = -g["k0"] * e1 + g["d2"] * e2
        v3 = g["k3"] * e3
        v4 = g["k4"] * e4
    else:
        raise ValueError("controller must be 'nonlinear' or 'linear'")

    u1 = -b2 * (e3 - e2 + s3 * s4 - x3 * x4) - v2
    u2 = (s2 * s4 - x2 * x4) - b3 * e3 - e1 - v3
    u3 = -(s2 * s3 - x2 * x3) + b4 * e4 - v4
    return np.array([u1, u2, u3], dtype=float)


def closed_loop_rhs(t, z, controller):
    master = z[:4]
    slave = z[4:]
    u = control_law(master, slave, controller)
    dxm = master_rhs(master)

    s1, s2, s3, s4 = slave
    u1, u2, u3 = u
    dxs = np.array([
        -b1 * s2,
        b2 * (s3 - s2 + s3 * s4) + u1,
        -s2 * s4 + b3 * s3 + s1 + u2,
        s2 * s3 - b4 * s4 + u3
    ], dtype=float)

    return np.concatenate([dxm, dxs])


def run_simulation(controller, master0=MASTER0, error0=ERROR0, t_final=T_FINAL):
    slave0 = master0 + error0
    z0 = np.concatenate([master0, slave0])
    t_eval = np.linspace(0.0, t_final, N_PLOT)

    sol = solve_ivp(
        fun=lambda t, z: closed_loop_rhs(t, z, controller),
        t_span=(0.0, t_final),
        y0=z0,
        t_eval=t_eval,
        method="RK45",
        rtol=RTOL,
        atol=ATOL,
        max_step=0.01
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed for {controller}: {sol.message}")

    master = sol.y[:4].T
    slave = sol.y[4:].T
    error = slave - master
    controls = np.vstack([
        control_law(master[i], slave[i], controller)
        for i in range(len(sol.t))
    ])
    return sol.t, master, slave, error, controls


def settling_time(t, error_norm, tol=SETTLING_TOL, persistence=PERSISTENCE):
    """First sampled time after which error stays below tolerance for persistence."""
    dt = float(t[1] - t[0])
    n_persist = max(1, int(np.ceil(persistence / dt)))
    below = error_norm <= tol
    for i in range(len(t) - n_persist + 1):
        if np.all(below[i:i+n_persist]):
            return float(t[i])
    return np.nan


def calculate_metrics(t, error, controls):
    err_norm = np.linalg.norm(error, axis=1)
    control_sq = np.sum(controls ** 2, axis=1)
    metrics = {
        "final_error_norm": float(err_norm[-1]),
        "max_error_norm": float(np.max(err_norm)),
        "settling_time_at_tolerance": settling_time(t, err_norm),
        "control_energy_integral": float(np.trapezoid(control_sq, t)),
        "IAE_L1_error": float(np.trapezoid(np.sum(np.abs(error), axis=1), t)),
        "ISE_squared_L2_error": float(np.trapezoid(err_norm ** 2, t)),
        "max_abs_control_u1": float(np.max(np.abs(controls[:, 0]))),
        "max_abs_control_u2": float(np.max(np.abs(controls[:, 1]))),
        "max_abs_control_u3": float(np.max(np.abs(controls[:, 2]))),
    }
    return metrics, err_norm


def save_state_plot(t, master, slave, suffix):
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, master[:, i], label=f"Master x{i+1}", linewidth=1.4)
        ax.plot(t, slave[:, i], "--", label=f"Slave xbar{i+1}", linewidth=1.2)
        ax.set_ylabel(f"State {i+1}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=2)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Master-slave state trajectories — {suffix}")
    fig.tight_layout()
    fig.savefig(OUT / f"01_state_trajectories_{suffix}.png", dpi=180)
    plt.close(fig)


def save_error_plot(t, error, err_norm, suffix):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i in range(4):
        axes[0].plot(t, error[:, i], label=f"e{i+1}")
    axes[0].set_ylabel("State error")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=4)

    axes[1].semilogy(t, np.maximum(err_norm, 1e-14), label="||e(t)||2")
    axes[1].axhline(SETTLING_TOL, linestyle="--", label=f"Tolerance {SETTLING_TOL:g}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error norm (log scale)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"Synchronization errors — {suffix}")
    fig.tight_layout()
    fig.savefig(OUT / f"02_error_trajectories_{suffix}.png", dpi=180)
    plt.close(fig)


def save_control_plot(t, controls, suffix):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, controls[:, i], linewidth=1.2)
        ax.set_ylabel(f"u{i+1}(t)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Control inputs — {suffix}")
    fig.tight_layout()
    fig.savefig(OUT / f"03_control_inputs_{suffix}.png", dpi=180)
    plt.close(fig)


def main():
    results = {}
    metric_rows = []

    for controller in ["nonlinear", "linear"]:
        t, master, slave, error, controls = run_simulation(controller)
        metrics, err_norm = calculate_metrics(t, error, controls)
        results[controller] = (t, master, slave, error, controls, metrics, err_norm)

        save_state_plot(t, master, slave, controller)
        save_error_plot(t, error, err_norm, controller)
        save_control_plot(t, controls, controller)

        row = {"controller": controller, **metrics}
        metric_rows.append(row)

        df = pd.DataFrame({
            "time_s": t,
            **{f"master_x{i+1}": master[:, i] for i in range(4)},
            **{f"slave_xbar{i+1}": slave[:, i] for i in range(4)},
            **{f"error_e{i+1}": error[:, i] for i in range(4)},
            "error_norm": err_norm,
            **{f"u{i+1}": controls[:, i] for i in range(3)}
        })
        df.to_csv(OUT / f"timeseries_{controller}.csv", index=False)

    # Comparison of error norms
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, data in results.items():
        t, _, _, _, _, _, err_norm = data
        ax.semilogy(t, np.maximum(err_norm, 1e-14), label=name)
    ax.axhline(SETTLING_TOL, linestyle="--", label=f"Tolerance {SETTLING_TOL:g}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||e(t)||2 (log scale)")
    ax.set_title("Prototype controller comparison (not paper-baseline comparison)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "04_controller_error_comparison.png", dpi=180)
    plt.close(fig)

    # Initial-condition sensitivity for the nonlinear prototype
    initial_errors = [
        np.array([0.5, -0.5, 0.5, -0.5]),
        np.array([1.0, -1.0, 1.0, -1.0]),
        np.array([2.0, -1.5, 1.0, -2.0]),
        np.array([-3.0, 2.0, -2.0, 1.0]),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ic_rows = []
    for j, e0 in enumerate(initial_errors, start=1):
        t, _, _, error, controls = run_simulation(
            "nonlinear", master0=MASTER0, error0=e0, t_final=T_FINAL
        )
        err_norm = np.linalg.norm(error, axis=1)
        ax.semilogy(t, np.maximum(err_norm, 1e-14), label=f"IC set {j}")
        ic_rows.append({
            "initial_error": str(e0.tolist()),
            "initial_error_norm": float(np.linalg.norm(e0)),
            "final_error_norm": float(err_norm[-1]),
            "settling_time_at_tolerance": settling_time(t, err_norm),
            "control_energy_integral": float(np.trapezoid(np.sum(controls**2, axis=1), t))
        })
    ax.axhline(SETTLING_TOL, linestyle="--", label=f"Tolerance {SETTLING_TOL:g}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||e(t)||2 (log scale)")
    ax.set_title("Initial-condition sensitivity — nonlinear prototype")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "05_initial_condition_comparison.png", dpi=180)
    plt.close(fig)

    pd.DataFrame(metric_rows).to_csv(OUT / "simulation_metrics.csv", index=False)
    pd.DataFrame(ic_rows).to_csv(OUT / "initial_condition_metrics.csv", index=False)

    print(f"Saved figures, time-series CSVs, and metrics to: {OUT.resolve()}")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nNOTE: If the error does not converge, do not hide the result.")
    print("Tune gains systematically and investigate the coupled e1/e2 dynamics.")
    print("These tests do not establish fixed-time stability or reproduce either paper controller.")


if __name__ == "__main__":
    main()
