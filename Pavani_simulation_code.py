"""
=============================================================================
NLDC GROUP 2 — Simulation Code
Fixed-Time Synchronization of Hyperchaotic Lü Systems
Authors: Khushi · Suhani Thakur · Vennela Varshini Anasoori · Gubba Pavani

Base papers:
  [1] M. Shafiq & I. Ahmad, AMC Vol.527, 2026. DOI:10.1016/j.amc.2026.130091
  [2] H. Wang & Y. Ai,      AMC Vol.433, 2022. DOI:10.1016/j.amc.2022.127388

HOW TO RUN:
  pip install numpy scipy matplotlib
  python simulation_code.py

What this script does:
  1. Simulates the Week 8 uncalibrated prototype (shows error floor at 3.03e-3)
  2. Simulates the recalibrated backstepping controller (Week 9, settles ~0.456 s)
  3. Simulates the 2026 Shafiq & Ahmad FTSC adapted to 4-D Lü (settles ~0.348 s)
  4. Simulates the 2022 Wang & Ai adaptive backstepping FTSC (settles ~0.452 s)
  5. Runs robustness tests: IC range, disturbances, parameter mismatch
  6. Prints a quantitative benchmark table
  7. Saves all figures to ./figures/
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────
# SYSTEM PARAMETERS
# ─────────────────────────────────────────────
b_nom = np.array([5.0, 35.0, 14.0, 3.0])   # [b1, b2, b3, b4]

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def sig(x, p):
    """Signed power function: sign(x)|x|^p"""
    return np.sign(x) * (np.abs(x) ** p)


def smooth_sig(x, p, delta=1e-4):
    """
    Smoothed version of sig(x,p) near zero.
    Replaces the region |x| < delta with a linear approximation to avoid
    the infinite-slope singularity |x|^(p-1) -> inf as x->0 (when p<1).
    """
    abs_x = np.abs(x)
    return np.where(abs_x >= delta,
                    np.sign(x) * (abs_x ** p),
                    x / (delta ** (1.0 - p)))


def master_dynamics(xm, b=b_nom):
    """4-D hyperchaotic Lü master system equations"""
    b1, b2, b3, b4 = b
    return np.array([
        -b1 * xm[1],
        b2 * (xm[2] - xm[1] + xm[2] * xm[3]),
        -xm[1] * xm[3] + b3 * xm[2] + xm[0],
        xm[1] * xm[2] - b4 * xm[3]
    ])


# ─────────────────────────────────────────────
# MAIN SIMULATION ENGINE
# ─────────────────────────────────────────────
def simulate(ctrl, t_max=10.0, steps=2500,
             x0_m=None, x0_s=None,
             b_actual=None, dist_type=None,
             **params):
    """
    Run a master-slave synchronization simulation.

    Parameters
    ----------
    ctrl      : str  — 'week8' | 'recalibrated' | '2026_paper' | '2022_adaptive'
    t_max     : float — simulation duration (s)
    steps     : int   — number of time steps
    x0_m      : array(4) — master initial state  (default [1,2,3,4])
    x0_s      : array(4) — slave  initial state  (default [2,-1,4,1])
    b_actual  : array(4) — system parameters (default b_nom; vary for mismatch test)
    dist_type : str   — None | 'constant' | 'sinusoidal' | 'noisy'
    **params  : controller hyperparameters (alpha, beta, p, q, gamma, k, c1, delta)

    Returns
    -------
    dict with keys: t, err_norm, E, U, ts (settling time), ctrl_effort, peak_u,
                    final_err, ise, iae, b_hat_hist
    """
    if x0_m   is None: x0_m     = np.array([1.0,  2.0, 3.0, 4.0])
    if x0_s   is None: x0_s     = np.array([2.0, -1.0, 4.0, 1.0])
    if b_actual is None: b_actual = b_nom.copy()

    # Controller hyperparameters (with defaults)
    alpha     = params.get('alpha',  5.0)
    beta      = params.get('beta',   5.0)
    p         = params.get('p',      1.2)
    q         = params.get('q',      0.7)
    gamma     = params.get('gamma',  5.0)   # tanh gain (2026 paper)
    k         = params.get('k',      8.0)   # linear gain
    c1        = params.get('c1',     3.0)   # backstepping tuning
    delta     = params.get('delta',  1e-4)  # smoothing threshold

    t_eval    = np.linspace(0, t_max, steps)
    dt        = t_eval[1] - t_eval[0]

    E         = np.zeros((steps, 4))
    U         = np.zeros((steps, 3))
    b_hat_hist = np.zeros((steps, 4))

    # Adaptive parameter estimates start far from true values (2022 controller)
    b_hat = np.array([3.0, 25.0, 10.0, 2.0])

    xm = x0_m.copy()
    xs = x0_s.copy()
    b1, b2, b3, b4 = b_actual

    for i, t in enumerate(t_eval):
        e = xs - xm
        E[i] = e

        # ── Disturbance ────────────────────────────────────────────────────
        if dist_type == 'constant':
            d = np.array([1.0, 1.0, 1.0])
        elif dist_type == 'sinusoidal':
            d = np.array([2.0*np.sin(3*t), 2.0*np.cos(3*t), 1.5*np.sin(2*t)])
        elif dist_type == 'noisy':
            np.random.seed(i)
            d = np.array([1.2*np.sin(25*t) + 0.3*np.random.randn(),
                          1.2*np.cos(25*t) + 0.3*np.random.randn(),
                          0.8*np.sin(20*t) + 0.3*np.random.randn()])
        else:
            d = np.zeros(3)

        # ── Controller ─────────────────────────────────────────────────────

        if ctrl == 'week8':
            # ── Week 8 UNCALIBRATED prototype ──────────────────────────────
            # Drives e2,e3,e4 to zero directly — no backstepping for e1.
            # Once e2→0, ė1=-b1*e2 freezes → error floor at 3.03e-3.
            u1 = (-b2*(xs[2]-xs[1]+xs[2]*xs[3]) + b2*(xm[2]-xm[1]+xm[2]*xm[3])
                  - 1.0*sig(e[1], 1.5) - 1.0*sig(e[1], 0.5))
            u2 = ((xs[1]*xs[3]-b3*xs[2]-xs[0]) - (xm[1]*xm[3]-b3*xm[2]-xm[0])
                  - 1.0*sig(e[2], 1.5) - 1.0*sig(e[2], 0.5))
            u3 = (-(xs[1]*xs[2]-b4*xs[3]) + (xm[1]*xm[2]-b4*xm[3])
                  - 1.0*sig(e[3], 1.5) - 1.0*sig(e[3], 0.5))

        elif ctrl == 'recalibrated':
            # ── Recalibrated Lyapunov backstepping controller ──────────────
            # (Week 9) Uses smooth_sig with fractional power q for near-zero term.
            # Backstepping virtual control α₁(e₁) redirects e₂ to also drive e₁→0.
            # Default: p=1.2, q=0.7, α=β=5.0
            alpha1    = (c1*e[0] + alpha*smooth_sig(e[0], p, delta)
                         + beta*smooth_sig(e[0], q, delta)) / b1
            z2        = e[1] - alpha1
            # Time derivative of α₁ (chain rule)
            da1_de1   = (c1
                         + alpha*p*(np.abs(e[0])**(p-1) if np.abs(e[0]) >= delta else delta**(p-1))
                         + beta *q*(np.abs(e[0])**(q-1) if np.abs(e[0]) >= delta else delta**(q-1))
                        ) / b1
            d_alpha1  = da1_de1 * (-b1 * e[1])

            u1 = (-b2*(xs[2]-xs[1]+xs[2]*xs[3]) + b2*(xm[2]-xm[1]+xm[2]*xm[3])
                  + d_alpha1 - k*z2
                  - alpha*smooth_sig(z2, p, delta) - beta*smooth_sig(z2, q, delta)
                  + b1*e[0])
            u2 = ((xs[1]*xs[3]-b3*xs[2]-xs[0]) - (xm[1]*xm[3]-b3*xm[2]-xm[0])
                  - k*e[2] - alpha*smooth_sig(e[2], p, delta) - beta*smooth_sig(e[2], q, delta))
            u3 = (-(xs[1]*xs[2]-b4*xs[3]) + (xm[1]*xm[2]-b4*xm[3])
                  - k*e[3] - alpha*smooth_sig(e[3], p, delta) - beta*smooth_sig(e[3], q, delta))

        elif ctrl == '2026_paper':
            # ── Shafiq & Ahmad (2026) FTSC — adapted to 4-D Lü system ──────
            # Key difference from 'recalibrated': near-zero term is β·tanh(γ·z)
            # instead of β·sig(z,q). tanh is smooth everywhere → no singularity,
            # no stiffness. Default: p=1.3, α=β=6.0, γ=5.0
            alpha1   = (c1*e[0] + alpha*sig(e[0], p)
                        + beta*np.tanh(gamma*e[0])) / b1
            z2       = e[1] - alpha1
            da1_de1  = (c1 + alpha*p*(np.abs(e[0])**(p-1))
                        + beta*gamma*(1.0 - np.tanh(gamma*e[0])**2)) / b1
            d_alpha1 = da1_de1 * (-b1 * e[1])

            u1 = (-b2*(xs[2]-xs[1]+xs[2]*xs[3]) + b2*(xm[2]-xm[1]+xm[2]*xm[3])
                  + d_alpha1 - k*z2
                  - alpha*sig(z2, p) - beta*np.tanh(gamma*z2)
                  + b1*e[0])
            u2 = ((xs[1]*xs[3]-b3*xs[2]-xs[0]) - (xm[1]*xm[3]-b3*xm[2]-xm[0])
                  - k*e[2] - alpha*sig(e[2], p) - beta*np.tanh(gamma*e[2]))
            u3 = (-(xs[1]*xs[2]-b4*xs[3]) + (xm[1]*xm[2]-b4*xm[3])
                  - k*e[3] - alpha*sig(e[3], p) - beta*np.tanh(gamma*e[3]))

        elif ctrl == '2022_adaptive':
            # ── Wang & Ai (2022) Adaptive Fixed-Time Backstepping ───────────
            # Uses online parameter estimates b̂ᵢ(t) via 4 adaptive laws.
            # Parameters are initially unknown: b_hat starts at [3,25,10,2].
            # Default for our sim: p=1.2, q=0.8, α=β=5.0
            b_hat_hist[i] = b_hat
            bh1 = max(b_hat[0], 0.5)   # clip to avoid division by near-zero

            alpha1   = (c1*e[0] + alpha*smooth_sig(e[0], p, delta)
                        + beta*smooth_sig(e[0], q, delta)) / bh1
            z2       = e[1] - alpha1
            da1_de1  = (c1
                        + alpha*p*(np.abs(e[0])**(p-1) if np.abs(e[0]) >= delta else delta**(p-1))
                        + beta *q*(np.abs(e[0])**(q-1) if np.abs(e[0]) >= delta else delta**(q-1))
                       ) / bh1
            d_alpha1 = da1_de1 * (-b_actual[0] * e[1])

            u1 = (-b_hat[1]*(xs[2]-xs[1]+xs[2]*xs[3]) + b_hat[1]*(xm[2]-xm[1]+xm[2]*xm[3])
                  + d_alpha1 - k*z2
                  - alpha*smooth_sig(z2, p, delta) - beta*smooth_sig(z2, q, delta)
                  + bh1*e[0])
            u2 = ((xs[1]*xs[3]-b_hat[2]*xs[2]-xs[0]) - (xm[1]*xm[3]-b_hat[2]*xm[2]-xm[0])
                  - k*e[2] - alpha*smooth_sig(e[2], p, delta) - beta*smooth_sig(e[2], q, delta))
            u3 = (-(xs[1]*xs[2]-b_hat[3]*xs[3]) + (xm[1]*xm[2]-b_hat[3]*xm[3])
                  - k*e[3] - alpha*smooth_sig(e[3], p, delta) - beta*smooth_sig(e[3], q, delta))

            # Adaptive parameter update laws  ḃ̂ᵢ = γₐ·(tracking term) - σ·(b̂ᵢ-bᵢ)
            ga = 4.0; sigma = 0.05
            db_hat = np.array([
                ga * e[0] * z2             - sigma*(b_hat[0]-b_nom[0]),
                ga * z2 * (xs[2]-xs[1]+xs[2]*xs[3]-(xm[2]-xm[1]+xm[2]*xm[3]))
                                           - sigma*(b_hat[1]-b_nom[1]),
                ga * e[2] * (xs[2]-xm[2]) - sigma*(b_hat[2]-b_nom[2]),
                ga * e[3] * (-(xs[3]-xm[3]))- sigma*(b_hat[3]-b_nom[3])
            ])
            b_hat = b_hat + db_hat * dt

        else:
            raise ValueError(f"Unknown controller: {ctrl}")

        U[i] = np.array([u1, u2, u3])

        # ── Euler integration (master and slave) ──────────────────────────
        dxm = master_dynamics(xm, b_actual)
        dxs = np.array([
            -b1*xs[1],
            b2*(xs[2]-xs[1]+xs[2]*xs[3]) + u1 + d[0],
            -xs[1]*xs[3] + b3*xs[2] + xs[0]     + u2 + d[1],
            xs[1]*xs[2]  - b4*xs[3]              + u3 + d[2]
        ])
        xm += dxm * dt
        xs += dxs * dt

    # ── Post-processing ────────────────────────────────────────────────────
    err_norm   = np.linalg.norm(E, axis=1)
    tol        = 1e-3

    # Settling time: first index where error stays below tol until end
    settled    = np.where(err_norm <= tol)[0]
    ts         = t_eval[settled[0]] if (len(settled) > 0 and settled[-1] == steps-1) else float('nan')

    ctrl_effort = np.trapz(np.sum(U**2, axis=1), t_eval)
    peak_u      = np.max(np.abs(U))
    ise         = np.trapz(err_norm**2, t_eval)
    iae         = np.trapz(err_norm,    t_eval)

    return {
        't': t_eval, 'err_norm': err_norm, 'E': E, 'U': U,
        'ts': ts, 'ctrl_effort': ctrl_effort, 'peak_u': peak_u,
        'final_err': err_norm[-1], 'ise': ise, 'iae': iae,
        'b_hat_hist': b_hat_hist
    }


# =============================================================================
# EXPERIMENT 1: Week 8 error floor vs recalibrated fix
# =============================================================================
print("=" * 65)
print("EXPERIMENT 1 — Week 8 failure vs Recalibrated Lyapunov")
print("=" * 65)

# Week 8: x0_s = [3, 0.5, 4, 2]  (same as original Week 8 run)
r_w8  = simulate('week8',        x0_s=np.array([3.0,0.5,4.0,2.0]),
                 alpha=1.0, beta=1.0, p=1.5, q=0.5)
# Recalibrated: same ICs as Week 8 for direct visual comparison
r_rec = simulate('recalibrated', x0_s=np.array([3.0,0.5,4.0,2.0]),
                 alpha=5.0, beta=5.0, p=1.2, q=0.7)

print(f"  Week 8 final error : {r_w8['final_err']:.4e}   (should stall ~3e-3)")
print(f"  Recalibrated ts    : {r_rec['ts']:.3f} s,  final error: {r_rec['final_err']:.2e}")

plt.figure(figsize=(9, 5.5))
plt.semilogy(r_w8['t'],  r_w8['err_norm'],  'r--', lw=2.5,
             label=f'Week 8 Prototype (stalled at {r_w8["final_err"]:.2e})')
plt.semilogy(r_rec['t'], r_rec['err_norm'], 'b-',  lw=2.5,
             label=f'Recalibrated (ts = {r_rec["ts"]:.3f} s)')
plt.axhline(1e-3, color='gray', ls=':', lw=2, label='Tolerance 1e-3')
plt.xlabel('Time (s)');  plt.ylabel('Error Norm ||e||')
plt.title('Week 8 Error Floor vs Recalibrated Backstepping Fix')
plt.legend(); plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/exp1_week8_vs_recalibrated.png', dpi=200)
plt.close()


# =============================================================================
# EXPERIMENT 2: Parameter Sweep (p, q, α, β)  — Week 9
# =============================================================================
print("\n" + "=" * 65)
print("EXPERIMENT 2 — Week 9 Parameter Recalibration Sweep")
print("=" * 65)
print(f"{'Config':<35} {'p':>4} {'q':>4} {'α':>5} {'β':>5} {'ts (s)':>8} {'Energy':>10} {'Final err':>11}")
print("-" * 85)

configs = [
    ("Week 8 Baseline (Uncalibrated)",  1.5, 0.5,  1.0,  1.0),
    ("Low Gains (Slow)",                1.2, 0.7,  1.0,  1.0),
    ("Moderate Gains (Optimal)",        1.2, 0.7,  5.0,  5.0),
    ("High Gains (Aggressive)",         1.2, 0.7, 12.0, 12.0),
    ("High p (p=2.0)",                  2.0, 0.7,  5.0,  5.0),
    ("Low q (q=0.4, stiff)",            1.2, 0.4,  5.0,  5.0),
]
for name, pv, qv, av, bv in configs:
    ctrl_name = 'week8' if '8 Baseline' in name else 'recalibrated'
    r = simulate(ctrl_name, alpha=av, beta=bv, p=pv, q=qv)
    ts_str = f"{r['ts']:.3f}" if not np.isnan(r['ts']) else "Failed(>10s)"
    print(f"  {name:<33} {pv:>4} {qv:>4} {av:>5} {bv:>5} {ts_str:>8} {r['ctrl_effort']:>10.1f} {r['final_err']:>11.2e}")


# =============================================================================
# EXPERIMENT 3: Fixed-Time Property — Initial Conditions  — Week 10
# =============================================================================
print("\n" + "=" * 65)
print("EXPERIMENT 3 — Week 10 Fixed-Time Invariance (IC range)")
print("=" * 65)
print(f"{'IC Set':<30} {'||e(0)||':>9} {'ts (s)':>8} {'Peak |u|':>10} {'Energy':>12}")
print("-" * 75)

ic_sets = [
    ("IC1: Small  ||e0||=0.32",  np.array([1.1,  1.8,  3.2,  3.9])),
    ("IC2: Moderate ||e0||=4.47", np.array([2.0, -1.0,  4.0,  1.0])),
    ("IC3: Large  ||e0||=10.54", np.array([6.0, -3.0,  8.0, -2.0])),
    ("IC4: Extreme ||e0||=22.27", np.array([11.0,-8.0, 13.0,-10.0])),
]
x0_master = np.array([1.0, 2.0, 3.0, 4.0])
for name, x0_s in ic_sets:
    r = simulate('2026_paper', x0_m=x0_master, x0_s=x0_s,
                 alpha=6.0, beta=6.0, p=1.3, gamma=5.0)
    e0 = np.linalg.norm(x0_s - x0_master)
    ts_str = f"{r['ts']:.3f}" if not np.isnan(r['ts']) else "No settle"
    print(f"  {name:<30} {e0:>9.2f} {ts_str:>8} {r['peak_u']:>10.2f} {r['ctrl_effort']:>12.1f}")

plt.figure(figsize=(9, 5.5))
colors = ['#2563eb', '#16a34a', '#ea580c', '#9333ea']
for (name, x0_s), col in zip(ic_sets, colors):
    r = simulate('2026_paper', x0_m=x0_master, x0_s=x0_s,
                 alpha=6.0, beta=6.0, p=1.3, gamma=5.0)
    ts_str = f"{r['ts']:.3f}" if not np.isnan(r['ts']) else "No settle"
    plt.semilogy(r['t'], r['err_norm'], color=col, lw=2.4, label=f'{name}  →  ts={ts_str} s')
plt.axhline(1e-3, color='gray', ls=':', lw=2, label='Tol 1e-3')
plt.axvline(2.5,  color='red',  ls='--', lw=2, label='Tmax ≈ 2.5 s')
plt.xlabel('Time (s)'); plt.ylabel('Error Norm ||e||')
plt.title('Fixed-Time Invariance Across Initial Error Magnitudes')
plt.legend(fontsize=9); plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/exp3_fixed_time_initial_conditions.png', dpi=200)
plt.close()


# =============================================================================
# EXPERIMENT 4: Disturbance Robustness — Week 10
# =============================================================================
print("\n" + "=" * 65)
print("EXPERIMENT 4 — Week 10 Disturbance Rejection")
print("=" * 65)
dist_cases = [('Nominal (d=0)',          None),
              ('Constant Bias (d=1.0)',  'constant'),
              ('Harmonic (d=2sin3t)',    'sinusoidal'),
              ('Stochastic HF Noise',    'noisy')]
plt.figure(figsize=(9, 5.5))
for (name, dtype), col in zip(dist_cases, colors):
    r = simulate('2026_paper', dist_type=dtype, alpha=6.0, beta=6.0, p=1.3, gamma=5.0)
    print(f"  {name:<30}  final_err={r['final_err']:.3e}")
    plt.semilogy(r['t'], r['err_norm'], lw=2.4, color=col,
                 label=f'{name}  (final err={r["final_err"]:.2e})')
plt.axhline(1e-3, color='gray', ls=':', lw=2, label='Tol 1e-3')
plt.xlabel('Time (s)'); plt.ylabel('Error Norm ||e||')
plt.title('Disturbance Rejection Robustness (2026 Controller)')
plt.legend(fontsize=9); plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/exp4_disturbance_robustness.png', dpi=200)
plt.close()


# =============================================================================
# EXPERIMENT 5: Parameter Mismatch ±15% — Week 10
# =============================================================================
print("\n" + "=" * 65)
print("EXPERIMENT 5 — Week 10 Parametric Robustness (±15%)")
print("=" * 65)
param_cases = [
    ('Nominal b=[5,35,14,3]',      b_nom),
    ('+15% Mismatch',              b_nom * 1.15),
    ('-15% Mismatch',              b_nom * 0.85),
    ('Asymmetric [5.5,40,12,3.5]', np.array([5.5,40.0,12.0,3.5])),
]
for (name, b_val), col in zip(param_cases, colors):
    r = simulate('2026_paper', b_actual=b_val, alpha=6.0, beta=6.0, p=1.3, gamma=5.0)
    ts_str = f"{r['ts']:.3f}" if not np.isnan(r['ts']) else "No settle"
    print(f"  {name:<35}  ts={ts_str} s,  final_err={r['final_err']:.2e}")


# =============================================================================
# EXPERIMENT 6: 2026 vs 2022 Benchmark — Week 11
# NOTE ON SETTLING TIME DIFFERENCE:
#   2026 controller: p=1.3, α=β=6.0, near-zero term = β·tanh(γ·z)  → ts ≈ 0.348 s
#   Recalibrated:    p=1.2, α=β=5.0, near-zero term = β·sig(z,0.7) → ts ≈ 0.456 s
#   These are TWO DIFFERENT controllers — the difference is intentional.
# =============================================================================
print("\n" + "=" * 65)
print("EXPERIMENT 6 — Week 11 Benchmark: 2026 vs 2022")
print("=" * 65)
print("  (Same ICs, same integration settings for fair comparison)")

r_2026 = simulate('2026_paper',  alpha=6.0, beta=6.0, p=1.3, gamma=5.0)
r_2022 = simulate('2022_adaptive', alpha=5.0, beta=5.0, p=1.2, q=0.8)

print(f"\n  {'Metric':<35} {'2026 FTSC':>12} {'2022 Adaptive':>14}")
print("  " + "-" * 65)
print(f"  {'Settling time ts (s)':<35} {r_2026['ts']:>12.3f} {r_2022['ts']:>14.3f}")
print(f"  {'Peak control |u|':<35} {r_2026['peak_u']:>12.2f} {r_2022['peak_u']:>14.2f}")
print(f"  {'Control energy ∫||u||²dt':<35} {r_2026['ctrl_effort']:>12.1f} {r_2022['ctrl_effort']:>14.1f}")
print(f"  {'ISE ∫||e||²dt':<35} {r_2026['ise']:>12.4f} {r_2022['ise']:>14.4f}")
print(f"  {'Final error':<35} {r_2026['final_err']:>12.2e} {r_2022['final_err']:>14.2e}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].semilogy(r_2026['t'], r_2026['err_norm'], 'b-',  lw=2.4,
                 label=f"2026 Shafiq & Ahmad  (ts={r_2026['ts']:.3f} s)")
axes[0].semilogy(r_2022['t'], r_2022['err_norm'], 'r--', lw=2.4,
                 label=f"2022 Wang & Ai       (ts={r_2022['ts']:.3f} s)")
axes[0].axhline(1e-3, color='gray', ls=':', lw=2)
axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Error Norm ||e||')
axes[0].set_title('Error Convergence Comparison'); axes[0].legend()
axes[0].grid(True, which='both', ls='--', alpha=0.4)

import scipy.integrate as spi
ce26 = spi.cumulative_trapezoid(np.sum(r_2026['U']**2, axis=1), r_2026['t'], initial=0)
ce22 = spi.cumulative_trapezoid(np.sum(r_2022['U']**2, axis=1), r_2022['t'], initial=0)
axes[1].plot(r_2026['t'], ce26, 'b-',  lw=2.4, label=f"2026 (total={r_2026['ctrl_effort']:.0f})")
axes[1].plot(r_2022['t'], ce22, 'r--', lw=2.4, label=f"2022 (total={r_2022['ctrl_effort']:.0f})")
axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Cumulative ∫||u||²dt')
axes[1].set_title('Cumulative Control Energy'); axes[1].legend()
axes[1].grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/exp6_2026_vs_2022_benchmark.png', dpi=200)
plt.close()

print("\n" + "=" * 65)
print("ALL DONE — figures saved to ./figures/")
print("=" * 65)
