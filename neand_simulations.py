from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt



def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def steps_to_years(steps: np.ndarray, *, N1: int, tau_years: float) -> np.ndarray:
    return steps.astype(float) * (tau_years / float(N1))



class NeutralParams:
    N1: int = 50
    M2: float = 0.15
    p_est: float | None = None   # default = 1/N1
    tau_years: float = 25.0
    max_steps: int = 600_000
    seed: int = 0


def simulate_trajectory(
    *, N1: int, M2: float, p_est: float, max_steps: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
  
    rng = np.random.default_rng(seed)
    i = 0
    frac = [0.0]

    for t in range(1, max_steps + 1):
        migrant_attempt = rng.random() < M2

        dead_is_modern = rng.random() < (i / N1)
        i_after = i - (1 if dead_is_modern else 0)

        if migrant_attempt and (rng.random() < p_est):
            replacement_modern = True
        else:
            replacement_modern = rng.random() < (i_after / (N1 - 1)) if N1 > 1 else False

        i = i_after + (1 if replacement_modern else 0)
        frac.append(i / N1)

        if i == N1:
            break

    return np.arange(len(frac), dtype=int), np.asarray(frac, dtype=float)


def simulate_fixation_times(
    *, N1: int, M2: float, p_est: float, n_runs: int, max_steps: int, seed: int
) -> np.ndarray:

    rng = np.random.default_rng(seed)
    i = np.zeros(n_runs, dtype=np.int32)
    t_fix = -np.ones(n_runs, dtype=np.int32)

    if N1 == 1:
        t_fix[:] = 1
        return t_fix

    denom = float(N1 - 1)

    for t in range(1, max_steps + 1):
        active = (t_fix < 0)
        if not np.any(active):
            break

        idx = np.where(active)[0]
        ii = i[idx].astype(np.float64)

        migrant_attempt = rng.random(idx.size) < M2
        dead_is_modern = rng.random(idx.size) < (ii / N1)
        i_after = ii - dead_is_modern.astype(np.float64)

        migrant_est = migrant_attempt & (rng.random(idx.size) < p_est)
        local_mod = rng.random(idx.size) < (i_after / denom)
        replacement_modern = migrant_est | (~migrant_est & local_mod)

        i[idx] = (i_after + replacement_modern.astype(np.float64)).astype(np.int32)

        fixed_now = (i[idx] == N1)
        t_fix[idx[fixed_now]] = t

    return t_fix


def neutral_make_figures(params: NeutralParams, outdir: Path) -> None:
    ensure_dir(outdir)

    N1, M2 = params.N1, params.M2
    p_est = params.p_est if params.p_est is not None else 1.0 / N1
    tau = params.tau_years

    plt.figure(figsize=(8, 5))
    n_traj = 30
    max_years = 20_000
    for k in range(n_traj):
        t_steps, frac = simulate_trajectory(N1=N1, M2=M2, p_est=p_est, max_steps=params.max_steps, seed=params.seed + k)
        years = steps_to_years(t_steps, N1=N1, tau_years=tau)
        mask = years <= max_years
        plt.plot(years[mask], frac[mask], linewidth=1)

    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Modern in Europe (i/N1)")
    plt.title(f"Trajectories (N1={N1}, M2={M2}, p_est={p_est:.4g}, tau={tau:g}y)")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "trajs.png", dpi=200)
    plt.close()

    n_runs = 3000
    t_fix = simulate_fixation_times(N1=N1, M2=M2, p_est=p_est, n_runs=n_runs, max_steps=params.max_steps, seed=params.seed + 111)
    ok = t_fix[t_fix > 0]
    years = steps_to_years(ok, N1=N1, tau_years=tau)

    plt.figure(figsize=(8, 5))
    plt.hist(years, bins=60)
    med = float(np.median(years))
    plt.axvline(med, linestyle="--", linewidth=2, label=f"Median = {med:.0f} years")
    plt.xlabel("Fixation time (years)")
    plt.ylabel("Count")
    plt.title(f"Fixation-time distribution (N1={N1}, M2={M2}, p_est={p_est:.4g}, runs={n_runs})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "fixation_hist.png", dpi=200)
    plt.close()
    yrs_sorted = np.sort(years)
    cdf = np.arange(1, yrs_sorted.size + 1) / float(yrs_sorted.size)
    plt.figure(figsize=(8, 5))
    plt.plot(yrs_sorted, cdf, linewidth=2)
    plt.xlabel("Time (years)")
    plt.ylabel("P(fixation by time)")
    plt.title("CDF of fixation times")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "fixation_cdf.png", dpi=200)
    plt.close()
    N1_values = [10, 20, 30, 50, 80, 120, 200]
    means, medians = [], []
    runs_per_N1 = 1500

    for N1v in N1_values:
        p_est_v = 1.0 / N1v
        t_fix_v = simulate_fixation_times(
            N1=N1v, M2=M2, p_est=p_est_v, n_runs=runs_per_N1, max_steps=params.max_steps, seed=params.seed + 1000 + N1v
        )
        okv = t_fix_v[t_fix_v > 0]
        yrsv = steps_to_years(okv, N1=N1v, tau_years=tau)
        means.append(float(np.mean(yrsv)))
        medians.append(float(np.median(yrsv)))

    plt.figure(figsize=(8, 5))
    plt.plot(N1_values, means, marker="o", label="Mean fixation time (years)")
    plt.plot(N1_values, medians, marker="o", label="Median fixation time (years)")
    plt.xlabel("N1 (number of bands in Europe)")
    plt.ylabel("Fixation time (years)")
    plt.title(f"Fixation time vs N1 (M2={M2}, p_est=1/N1, runs={runs_per_N1}, tau={tau:g}y)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fixation_vs_N1.png", dpi=200)
    plt.close()
    M2_vals = np.linspace(0.01, 0.30, 15)
    p_vals = np.linspace(0.002, 0.08, 15)
    runs_per_cell = 250
    med_years = np.full((p_vals.size, M2_vals.size), np.nan, dtype=float)

    for ip, p in enumerate(p_vals):
        for im, m in enumerate(M2_vals):
            t_fix_h = simulate_fixation_times(
                N1=params.N1, M2=float(m), p_est=float(p),
                n_runs=runs_per_cell, max_steps=params.max_steps,
                seed=params.seed + 5000 + 1000*ip + im
            )
            okh = t_fix_h[t_fix_h > 0]
            if okh.size == 0:
                continue
            yrsh = steps_to_years(okh, N1=params.N1, tau_years=tau)
            med_years[ip, im] = float(np.median(yrsh))

    plt.figure(figsize=(9, 6))
    im = plt.imshow(
        med_years,
        aspect="auto",
        origin="lower",
        extent=[M2_vals.min(), M2_vals.max(), p_vals.min(), p_vals.max()],
    )
    plt.colorbar(im, label="Median fixation time (years)")
    plt.xlabel("M2 (migrant attempt probability per step)")
    plt.ylabel("p_est (P(establish | arrival))")
    plt.title(f"Sensitivity heatmap (N1={params.N1}, runs/cell={runs_per_cell}, tau={tau:g}y)")
    plt.tight_layout()
    plt.savefig(outdir / "fig_migration_sensitivity_heatmap.png", dpi=200)
    plt.close()

class PDEParams:
    nx: int = 300
    L: float = 1.0
    dt_years: float = 5.0
    t_end_years: float = 15000.0
    K0: float = 3000.0
    rN: float = 0.0020
    rS: float = 0.0025
    DN: float = 1.0e-6
    DS: float = 1.2e-6
    alpha: float = 1.25
    beta: float = 0.85
    sapiens_seed_frac: float = 0.02
    sapiens_seed_width: float = 0.03


def laplacian_neumann(u: np.ndarray, dx: float) -> np.ndarray:
    lap = np.empty_like(u)
    lap[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx*dx)
    lap[0] = 2*(u[1] - u[0]) / (dx*dx)
    lap[-1] = 2*(u[-2] - u[-1]) / (dx*dx)
    return lap


def simulate_pde(params: PDEParams, snapshot_years: list[float]) -> dict:
    x = np.linspace(0.0, params.L, params.nx)
    dx = x[1] - x[0]
    K = params.K0 * np.ones_like(x)

    nN = K.copy()  # Neanderthals initially at K everywhere
    seed = params.sapiens_seed_frac * params.K0 * np.exp(-(x/params.sapiens_seed_width)**2)
    nS = seed.copy()

    dt = params.dt_years
    n_steps = int(np.ceil(params.t_end_years / dt))

    snapshot_years = sorted(snapshot_years)
    snap_idx = 0
    snaps_N, snaps_S = [], []

    times, totN, totS = [], [], []
    initial_totalN = float(np.trapz(nN, x))
    ext_threshold = 0.01 * initial_totalN
    ext_time = None

    for step in range(n_steps + 1):
        t = step * dt

        while snap_idx < len(snapshot_years) and t >= snapshot_years[snap_idx] - 0.5*dt:
            snaps_N.append((snapshot_years[snap_idx], nN.copy()))
            snaps_S.append((snapshot_years[snap_idx], nS.copy()))
            snap_idx += 1

        times.append(t)
        totN.append(float(np.trapz(nN, x)))
        totS.append(float(np.trapz(nS, x)))

        if ext_time is None and totN[-1] <= ext_threshold:
            ext_time = t

        if step == n_steps:
            break

        lapN = laplacian_neumann(nN, dx)
        lapS = laplacian_neumann(nS, dx)

        growN = params.rN * nN * (1.0 - (nN + params.alpha*nS) / K)
        growS = params.rS * nS * (1.0 - (nS + params.beta*nN) / K)

        nN = nN + dt * (params.DN * lapN + growN)
        nS = nS + dt * (params.DS * lapS + growS)

        nN = np.clip(nN, 0.0, None)
        nS = np.clip(nS, 0.0, None)

    return {
        "x": x,
        "K0": params.K0,
        "snaps_N": snaps_N,
        "snaps_S": snaps_S,
        "times": np.asarray(times),
        "totN": np.asarray(totN),
        "totS": np.asarray(totS),
        "ext_time": ext_time,
    }


def pde_make_figures(params: PDEParams, outdir: Path) -> None:
    ensure_dir(outdir)
    snapshot_years = [0, 3000, 6000, 9000, 12000, 15000]
    res = simulate_pde(params, snapshot_years)

    x = res["x"]
    K0 = res["K0"]

    def plot_snaps(snaps, title, ylabel, outfile):
        plt.figure(figsize=(10, 4))
        for t, u in snaps:
            plt.plot(x, u / K0, label=f"t={int(t)} yr")
        plt.xlabel("x in [0,1]  (0 = Levant, 1 = Europe)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.ylim(-0.02, 1.02)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3, fontsize=8, frameon=True)
        plt.tight_layout()
        plt.savefig(outdir / outfile, dpi=200)
        plt.close()

    plot_snaps(res["snaps_N"], "Neanderthals density profiles (snapshots)", "nN(x,t)/K", "density.png")
    plot_snaps(res["snaps_S"], "Sapiens density profiles (snapshots)", "nS(x,t)/K", "density2.png")

    plt.figure(figsize=(10, 4))
    plt.plot(res["times"], res["totN"], label=r"Total Neanderthals  $\int n_N\,dx$")
    plt.plot(res["times"], res["totS"], label=r"Total Sapiens  $\int n_S\,dx$")
    if res["ext_time"] is not None:
        plt.axvline(res["ext_time"], linestyle="--", label=f"Extinction threshold ~ {res['ext_time']:.0f} yr")
    plt.xlabel("time (years)")
    plt.ylabel("Total population (integral over space)")
    plt.title("Global dynamics (extinction in the model)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "global_dynamics.png", dpi=200)
    plt.close()


def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = f(t + dt, y + dt*k3)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def simulate_rk4(f, y0, t0, t1, dt):
    n = int(np.ceil((t1 - t0)/dt))
    t = np.linspace(t0, t0 + n*dt, n+1)
    y = np.zeros((n+1, len(y0)), dtype=float)
    y[0] = y0
    for k in range(n):
        y[k+1] = rk4_step(f, t[k], y[k], dt)
        y[k+1] = np.clip(y[k+1], 0.0, None)
    return t, y

class AlleeParams:
    r: float = 0.03
    K: float = 100.0
    A: float = 15.0
    t_end: float = 300.0
    dt: float = 0.2


@dataclass(frozen=True)
class HybridParams:
    rN: float = 0.02
    rS: float = 0.03
    K: float = 100.0
    alpha: float = 1.1
    beta: float = 0.9
    gamma: float = 0.03
    t_end: float = 400.0
    dt: float = 0.2


def ode_make_figures(allee: AlleeParams, hybrid: HybridParams, outdir: Path) -> None:
    ensure_dir(outdir)

    def f_allee(t, y):
        N = y[0]
        return np.array([allee.r * N * (1 - N/allee.K) * ((N - allee.A) / (N + allee.A))])

    def f_logistic(t, y):
        N = y[0]
        return np.array([allee.r * N * (1 - N/allee.K)])

    t, y_above = simulate_rk4(f_allee, np.array([40.0]), 0.0, allee.t_end, allee.dt)
    _, y_below = simulate_rk4(f_allee, np.array([10.0]), 0.0, allee.t_end, allee.dt)
    _, y_log = simulate_rk4(f_logistic, np.array([10.0]), 0.0, allee.t_end, allee.dt)

    plt.figure(figsize=(8, 5))
    plt.plot(t, y_above[:, 0], label="Allee: start above A")
    plt.plot(t, y_below[:, 0], label="Allee: start below A")
    plt.plot(t, y_log[:, 0], linestyle="--", label="Logistic (no Allee)")
    plt.axhline(allee.A, linestyle=":", linewidth=2, label=f"A = {allee.A:g}")
    plt.xlabel("Time (years)")
    plt.ylabel("Population size N")
    plt.title("Impact of the Allee effect")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_allee_threshold.png", dpi=200)
    plt.close()
    def f_hybrid(t, y):
        N, S, H = y
        dN = hybrid.rN * N * (1 - (N + hybrid.alpha*S)/hybrid.K) - hybrid.gamma*N*S
        dS = hybrid.rS * S * (1 - (S + hybrid.beta*N)/hybrid.K)
        dH = hybrid.gamma*N*S
        return np.array([dN, dS, dH])

    y0 = np.array([80.0, 5.0, 0.0])
    t2, y2 = simulate_rk4(f_hybrid, y0, 0.0, hybrid.t_end, hybrid.dt)
    N, S, H = y2[:, 0], y2[:, 1], y2[:, 2]

    plt.figure(figsize=(8, 5))
    plt.plot(t2, N/hybrid.K, label="Neanderthals N/K")
    plt.plot(t2, S/hybrid.K, label="Sapiens S/K")
    plt.plot(t2, H/hybrid.K, label="Hybrid/Introgressed H/K")
    plt.xlabel("Time (years)")
    plt.ylabel("Relative population density")
    plt.title(f"Replacement with interbreeding (gamma={hybrid.gamma:g})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_hybrid_dynamics.png", dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="outputs_all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--pde", action="store_true")
    ap.add_argument("--ode", action="store_true")

    ap.add_argument("--N1", type=int, default=50)
    ap.add_argument("--M2", type=float, default=0.15)
    ap.add_argument("--p_est", type=float, default=None)
    ap.add_argument("--tau", type=float, default=25.0)
    ap.add_argument("--max_steps", type=int, default=600000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--nx", type=int, default=300)
    ap.add_argument("--dt_years", type=float, default=5.0)
    ap.add_argument("--t_end_years", type=float, default=15000.0)

    args = ap.parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    if args.all:
        args.neutral = args.pde = args.ode = True

    if args.neutral:
        neutral_dir = outdir / "neutral"
        neutral_make_figures(
            NeutralParams(
                N1=args.N1, M2=args.M2, p_est=args.p_est,
                tau_years=args.tau, max_steps=args.max_steps, seed=args.seed
            ),
            neutral_dir
        )

    if args.pde:
        pde_dir = outdir / "pde"
        pde_make_figures(
            PDEParams(nx=args.nx, dt_years=args.dt_years, t_end_years=args.t_end_years),
            pde_dir
        )

    if args.ode:
        ode_dir = outdir / "ode"
        ode_make_figures(AlleeParams(), HybridParams(), ode_dir)

    print(f"Done. Outputs saved under: {outdir.resolve()}")


if __name__ == "__main__":
    main()
