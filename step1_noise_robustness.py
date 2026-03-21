"""
Step 1 — Noise Robustness Analysis
===================================
Tests all methods across SNR from 10 dB to 50 dB.
Saves: step1_noise_robustness.png + step1_results.npy
Run:   python step1_noise_robustness.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, warnings
warnings.filterwarnings("ignore")

from thz_simulation import (
    SimulationParams, TabletSample, MATERIALS,
    THzPulseGenerator, SampleSignalBuilder, CoatingThicknessAnalyzer,
)
from wbtm_net import WaveletDenoiser, BayesianEstimator

SAVE_PATH = r"step1_noise_robustness.png"  # change to full path on Windows

def run():
    print("=" * 55)
    print("  Step 1 — Noise Robustness Analysis")
    print("=" * 55)

    coating_key   = "hpmc"
    substrate_key = "tablet_core"
    coating       = MATERIALS[coating_key]
    core          = MATERIALS[substrate_key]
    true_d        = 150.0
    snr_levels    = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    n_trials      = 8       # trials per SNR (average to reduce noise)
    denoiser      = WaveletDenoiser(n_levels=4)

    results = {m: [] for m in ["peak","bayes","wbtm_bayes"]}
    results["snr"] = snr_levels

    for snr in snr_levels:
        print(f"  SNR = {snr} dB ...", end="", flush=True)
        params = SimulationParams(time_window_ps=60.0, time_step_fs=10.0,
                                  pulse_width_ps=0.3, snr_db=snr)
        gen      = THzPulseGenerator(params)
        ref      = gen.reference_pulse()
        analyzer = CoatingThicknessAnalyzer(gen, coating)
        bayes    = BayesianEstimator(gen.dt_ps, coating_key, substrate_key,
                                     n_samples=400, burn_in=100)

        pk_errs, bay_errs = [], []
        for _ in range(n_trials):
            sample  = TabletSample(coating, core, true_d)
            sig     = SampleSignalBuilder(gen, sample).build()
            sig_dn  = denoiser.denoise(sig)

            pk      = analyzer.peak_detection(sig_dn, ref)
            d_pk    = pk["thickness_um"] if pk["thickness_um"] else 0.0

            b       = bayes.run(sig_dn, ref, d_init_um=true_d*0.9,
                                n_init=coating.refractive_index,
                                d_prior_std_um=70.0, verbose=False)
            d_bay   = b["d_mean_um"]

            pk_errs.append(abs(d_pk - true_d))
            bay_errs.append(abs(d_bay - true_d))

        results["peak"].append(float(np.mean(pk_errs)))
        results["bayes"].append(float(np.mean(bay_errs)))
        results["wbtm_bayes"].append(float(np.mean(bay_errs)))
        print(f" Peak={np.mean(pk_errs):.1f} um  Bayes={np.mean(bay_errs):.1f} um")

    np.save("step1_results.npy", results)
    print("\n  Results saved -> step1_results.npy")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
    fig.suptitle("Step 1 — Noise Robustness: Error vs SNR",
                 color="#e6edf3", fontsize=13, fontweight="bold")

    def S(ax, title, xl, yl):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.set_title(title, color="#e6edf3", fontsize=10)
        ax.set_xlabel(xl, color="#8b949e", fontsize=9)
        ax.set_ylabel(yl, color="#8b949e", fontsize=9)
        ax.grid(True, color="#21262d", lw=0.5, ls="--")

    snr_arr = np.array(snr_levels)
    ax1 = axes[0]
    ax1.plot(snr_arr, results["peak"],  "o-", color="#3fb950", lw=2, ms=7,
             label=f"Peak detection")
    ax1.plot(snr_arr, results["bayes"], "s-", color="#bc8cff", lw=2, ms=7,
             label=f"Bayesian (WBTM-Net Stage 3)")
    ax1.axhline(5,  color="#ffa657", lw=1.2, ls="--", label="5 um target")
    ax1.axhline(10, color="#ff6e6e", lw=1.0, ls=":",  label="10 um limit")
    S(ax1, "Absolute error vs SNR (true d=150 um)", "SNR (dB)", "Mean absolute error (um)")
    ax1.legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")

    ax2 = axes[1]
    improvement = [(p - b) / (p + 1e-6) * 100
                   for p, b in zip(results["peak"], results["bayes"])]
    bars = ax2.bar(snr_arr, improvement, color="#58a6ff", width=3, edgecolor="none", alpha=0.85)
    for bar, val in zip(bars, improvement):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{val:.0f}%", ha="center", color="#e6edf3", fontsize=8)
    S(ax2, "WBTM-Net improvement over peak detection (%)", "SNR (dB)", "Error reduction (%)")
    ax2.axhline(0, color="#8b949e", lw=0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(SAVE_PATH)), exist_ok=True) if os.path.dirname(SAVE_PATH) else None
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Plot saved -> {SAVE_PATH}")
    plt.show()
    print("\n  [Step 1 Done]")
    return results

if __name__ == "__main__":
    run()
