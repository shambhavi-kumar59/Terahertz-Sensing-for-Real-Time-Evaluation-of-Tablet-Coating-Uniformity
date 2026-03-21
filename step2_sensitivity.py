"""
Step 2 — Sensitivity Analysis: Robustness to Wrong Refractive Index
====================================================================
Shows WBTM-Net is far more robust when assumed n is wrong.
Saves: step2_sensitivity.png + step2_results.npy
Run:   python step2_sensitivity.py
"""
import numpy as np
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")

from thz_simulation import (
    SimulationParams, TabletSample, MATERIALS, Material,
    THzPulseGenerator, SampleSignalBuilder, CoatingThicknessAnalyzer,
)
from wbtm_net import WaveletDenoiser, BayesianEstimator

SAVE_PATH = r"step2_sensitivity.png"

def run():
    print("=" * 55)
    print("  Step 2 — Sensitivity Analysis (wrong n)")
    print("=" * 55)

    coating_key   = "hpmc"
    substrate_key = "tablet_core"
    coating       = MATERIALS[coating_key]
    core          = MATERIALS[substrate_key]
    true_d        = 150.0
    true_n        = coating.refractive_index   # 1.52
    snr_db        = 35.0
    n_offsets     = np.linspace(-0.10, 0.10, 11)  # assumed n = true_n + offset

    params   = SimulationParams(snr_db=snr_db)
    gen      = THzPulseGenerator(params)
    ref      = gen.reference_pulse()
    denoiser = WaveletDenoiser(n_levels=4)

    # Ground truth signal (with true n)
    sample  = TabletSample(coating, core, true_d)
    sig     = SampleSignalBuilder(gen, sample).build()
    sig_dn  = denoiser.denoise(sig)

    results = {"n_offset": n_offsets.tolist(),
               "peak_err": [], "bayes_err": []}

    print(f"  True n = {true_n:.4f}   True d = {true_d} um")
    print(f"  {'n_offset':>10}  {'Assumed n':>10}  {'Peak err':>10}  {'Bayes err':>10}")

    for offset in n_offsets:
        assumed_n = true_n + offset

        # Peak detection: uses assumed_n directly in d = c*Dt/(2n)
        analyzer_wrong = CoatingThicknessAnalyzer(
            gen,
            Material("Wrong n", assumed_n, coating.absorption_coeff))
        pk = analyzer_wrong.peak_detection(sig_dn, ref)
        d_pk = pk["thickness_um"] if pk["thickness_um"] else 0.0

        # Bayesian: initialised with assumed_n but estimates n simultaneously
        bayes = BayesianEstimator(gen.dt_ps, coating_key, substrate_key,
                                  n_samples=400, burn_in=100)
        b = bayes.run(sig_dn, ref, d_init_um=true_d*0.9,
                      n_init=assumed_n, d_prior_std_um=70.0, verbose=False)
        d_bay = b["d_mean_um"]

        results["peak_err"].append(abs(d_pk - true_d))
        results["bayes_err"].append(abs(d_bay - true_d))
        print(f"  {offset:>+10.3f}  {assumed_n:>10.4f}  {abs(d_pk-true_d):>10.2f}  {abs(d_bay-true_d):>10.2f}")

    np.save("step2_results.npy", results)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
    fig.suptitle("Step 2 — Sensitivity: Error when Refractive Index is Wrong",
                 color="#e6edf3", fontsize=13, fontweight="bold")

    def S(ax, title, xl, yl):
        ax.set_facecolor("#161b22"); ax.tick_params(colors="#8b949e", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.set_title(title, color="#e6edf3", fontsize=10)
        ax.set_xlabel(xl, color="#8b949e", fontsize=9)
        ax.set_ylabel(yl, color="#8b949e", fontsize=9)
        ax.grid(True, color="#21262d", lw=0.5, ls="--")

    ax1 = axes[0]
    ax1.plot(n_offsets, results["peak_err"],  "o-", color="#3fb950", lw=2, ms=7,
             label="Peak detection (assumes fixed n)")
    ax1.plot(n_offsets, results["bayes_err"], "s-", color="#bc8cff", lw=2, ms=7,
             label="Bayesian WBTM-Net (estimates n)")
    ax1.axvline(0, color="#ffa657", lw=1.2, ls="--", label="True n (offset=0)")
    ax1.axhline(5, color="#ff6e6e", lw=1.0, ls=":", label="5 um target")
    S(ax1, "Error vs assumed n offset (true d=150 um, SNR=35 dB)",
      "n offset (assumed - true)", "Absolute error (um)")
    ax1.legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")

    ax2 = axes[1]
    ratio = [p/(b+1e-3) for p, b in zip(results["peak_err"], results["bayes_err"])]
    ax2.bar(n_offsets, ratio, width=0.016, color="#58a6ff", edgecolor="none", alpha=0.85)
    ax2.axhline(1.0, color="#ffa657", lw=1.2, ls="--", label="No improvement (ratio=1)")
    ax2.axhline(4.0, color="#3fb950", lw=1.0, ls=":", label="4x improvement line")
    S(ax2, "Robustness ratio: Peak error / Bayesian error\n(higher = Bayesian more robust)",
      "n offset", "Error ratio (peak / Bayesian)")
    ax2.legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Plot saved -> {SAVE_PATH}")
    plt.show()
    print("  [Step 2 Done]")
    return results

if __name__ == "__main__":
    run()
