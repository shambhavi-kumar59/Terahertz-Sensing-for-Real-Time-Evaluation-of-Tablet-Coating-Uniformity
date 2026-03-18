"""
Terahertz (THz) Pulse Simulation for Tablet Coating Thickness Analysis
=======================================================================
Simulates THz time-domain spectroscopy (TDS) to measure tablet coating
thickness using time delay between surface and coating-substrate reflections.

Physics:
  - THz pulse reflects at air/coating interface (surface reflection)
  - Second reflection occurs at coating/tablet-core interface
  - Time delay Δt between the two echoes → thickness d = (c * Δt) / (2 * n)
    where n is the refractive index of the coating material
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# ── Physical constants ──────────────────────────────────────────────────────
C_LIGHT = 2.998e8          # m/s — speed of light in vacuum
FS_TO_PS = 1e-3            # femtosecond → picosecond
PS_TO_S  = 1e-12           # picosecond → second


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class Material:
    """Optical properties of a material in the THz range."""
    name: str
    refractive_index: float        # real part of n
    absorption_coeff: float        # cm⁻¹ at 1 THz
    description: str = ""

    @property
    def complex_n(self) -> complex:
        """Complex refractive index: n + iκ.  κ ≈ α·c/(4π·f)."""
        f_ref = 1.0e12                           # reference frequency 1 THz
        kappa = (self.absorption_coeff * 1e2 * C_LIGHT) / (4 * np.pi * f_ref)
        return complex(self.refractive_index, kappa)


@dataclass
class SimulationParams:
    """Parameters controlling the THz simulation."""
    time_window_ps:  float = 60.0    # total time window [ps]
    time_step_fs:    float = 10.0    # time resolution [fs]
    pulse_width_ps:  float = 0.3     # Gaussian FWHM of THz pulse [ps]
    center_freq_THz: float = 1.0     # center frequency [THz]
    noise_level:     float = 1e-4    # additive Gaussian noise amplitude
    snr_db:          float = 40.0    # signal-to-noise ratio [dB]


@dataclass
class TabletSample:
    """Describes a coated tablet sample."""
    coating: Material
    core:    Material
    coating_thickness_um: float      # true thickness in micrometres
    surface_reflectance:  float = 0.05   # Fresnel reflection at air/coating

    @property
    def coating_thickness_m(self) -> float:
        return self.coating_thickness_um * 1e-6

    @property
    def expected_delay_ps(self) -> float:
        """Expected round-trip time delay through the coating [ps]."""
        d = self.coating_thickness_m
        n = self.coating.refractive_index
        return (2 * n * d / C_LIGHT) * 1e12

    @property
    def core_reflectance(self) -> float:
        """Fresnel reflectance at coating/core interface (normal incidence)."""
        n1 = self.coating.refractive_index
        n2 = self.core.refractive_index
        r  = (n1 - n2) / (n1 + n2)
        return r ** 2


# ── Material library ─────────────────────────────────────────────────────────
MATERIALS = {
    "hpmc":        Material("HPMC",          1.52, 2.1,  "Hydroxypropyl methylcellulose"),
    "eudragit_l":  Material("Eudragit L",    1.50, 1.8,  "Methacrylic acid copolymer"),
    "ec":          Material("Ethylcellulose", 1.48, 3.0,  "Ethylcellulose polymer"),
    "pvp":         Material("PVP",            1.55, 4.5,  "Polyvinylpyrrolidone"),
    "tablet_core": Material("Tablet Core",   1.60, 8.0,  "Compressed lactose/MCC core"),
    "mcc":         Material("MCC",            1.58, 6.0,  "Microcrystalline cellulose"),
}


# ── THz Pulse generator ───────────────────────────────────────────────────────
class THzPulseGenerator:
    """Generates a realistic THz time-domain pulse."""

    def __init__(self, params: SimulationParams):
        self.p = params
        self.dt_ps = params.time_step_fs * FS_TO_PS          # [ps]
        self.t = np.arange(0,
                           params.time_window_ps,
                           self.dt_ps)                        # time axis [ps]
        self.N = len(self.t)

    def reference_pulse(self, t0_ps: float = 5.0) -> np.ndarray:
        """
        Single-cycle THz pulse: first derivative of a Gaussian (realistic
        photoconductive antenna shape).
        """
        t    = self.t
        σ    = self.p.pulse_width_ps / (2 * np.sqrt(2 * np.log(2)))
        gauss = np.exp(-0.5 * ((t - t0_ps) / σ) ** 2)
        pulse = -(t - t0_ps) / (σ ** 2) * gauss          # 1st derivative
        pulse /= np.max(np.abs(pulse))                    # normalise
        return pulse

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add Gaussian white noise scaled to the desired SNR."""
        rms   = np.sqrt(np.mean(signal ** 2))
        noise_rms = rms / (10 ** (self.p.snr_db / 20))
        noise = np.random.normal(0, noise_rms, signal.shape)
        return signal + noise


# ── Sample signal builder ────────────────────────────────────────────────────
class SampleSignalBuilder:
    """
    Constructs the reflected THz signal from a coated tablet.

    Reflections modelled:
      1. Air / coating surface  → main echo (sign: positive Fresnel)
      2. Coating / core         → delayed echo (sign depends on Δn)
      3. Multiple internal reflections (Fabry–Pérot, attenuated)
    """

    def __init__(self, generator: THzPulseGenerator, sample: TabletSample):
        self.gen    = generator
        self.sample = sample

    def build(self, t0_ref_ps: float = 5.0,
              n_multiples: int = 2) -> np.ndarray:
        """
        Returns the time-domain reflected signal.

        Parameters
        ----------
        t0_ref_ps : float
            Arrival time of the surface reflection peak [ps].
        n_multiples : int
            Number of Fabry–Pérot multiple reflections to include.
        """
        t      = self.gen.t
        ref    = self.gen.reference_pulse(t0_ref_ps)
        Δt     = self.sample.expected_delay_ps
        s      = self.sample

        # Reflection amplitudes
        r_surf = np.sqrt(s.surface_reflectance)     # air→coating reflection
        r_core = np.sqrt(s.core_reflectance)         # coating→core reflection
        # Transmission factors (energy conservation: T = 1 - R)
        T_ac   = np.sqrt(1 - s.surface_reflectance)  # air→coating transmit
        T_ca   = T_ac                                 # coating→air transmit

        # Absorption loss through coating (Beer–Lambert, round trip)
        α_m   = s.coating.absorption_coeff * 1e2       # convert cm⁻¹ → m⁻¹
        d     = s.coating_thickness_m
        attn  = np.exp(-α_m * 2 * d)

        # ── Echo 1: surface reflection ──
        signal = r_surf * ref

        # ── Echo 2: coating/core reflection ──
        t2   = t0_ref_ps + Δt
        ref2 = self._shift_pulse(ref, t, t0_ref_ps, t2)
        amp2 = T_ac * r_core * T_ca * attn
        signal += amp2 * ref2

        # ── Multiple reflections (Fabry–Pérot) ──
        r_ac_internal = -r_surf          # reflected back into coating
        for k in range(1, n_multiples + 1):
            tk  = t0_ref_ps + (2 * k + 1) * Δt
            rfk = self._shift_pulse(ref, t, t0_ref_ps, tk)
            amp = (T_ac * (r_ac_internal ** (2 * k - 1)) *
                   r_core * T_ca * attn ** (2 * k + 1))
            signal += amp * rfk

        signal = self.gen.add_noise(signal)
        return signal

    @staticmethod
    def _shift_pulse(pulse: np.ndarray,
                     t: np.ndarray,
                     t_orig: float,
                     t_new:  float) -> np.ndarray:
        """Shift a pulse in time by interpolation."""
        return np.interp(t - (t_new - t_orig),
                         t, pulse, left=0, right=0)


# ── Thickness analyser ────────────────────────────────────────────────────────
class CoatingThicknessAnalyzer:
    """
    Extracts coating thickness from a measured THz waveform using:
      1. Peak detection in the time domain
      2. Deconvolution / frequency-domain analysis
      3. Cross-correlation with the reference pulse
    """

    def __init__(self, generator: THzPulseGenerator,
                 coating_material: Material):
        self.gen  = generator
        self.mat  = coating_material
        self.t    = generator.t
        self.dt   = generator.dt_ps * 1e-12    # [s]

    # ── Method 1: Direct peak detection ──────────────────────────────────────
    def peak_detection(self, signal: np.ndarray,
                       ref_pulse: np.ndarray,
                       min_prominence: float = 0.03
                       ) -> dict:
        """
        Find peaks in the absolute value of the signal envelope.
        Δt = time between 1st and 2nd dominant peaks.
        """
        env = np.abs(signal)
        peaks, props = find_peaks(env,
                                  prominence=min_prominence * env.max(),
                                  distance=int(0.5 / self.gen.dt_ps))

        result = {"method": "Peak Detection", "peaks_ps": [], "delta_t_ps": None,
                  "thickness_um": None, "confidence": 0.0}

        if len(peaks) >= 2:
            t_peaks = self.t[peaks]
            result["peaks_ps"] = t_peaks[:4].tolist()
            Δt = t_peaks[1] - t_peaks[0]
            d  = self._thickness_from_delay(Δt)
            result.update({"delta_t_ps": Δt,
                           "thickness_um": d,
                           "confidence": float(min(props["prominences"][:2]) /
                                               env.max())})
        return result

    # ── Method 2: Cross-correlation ───────────────────────────────────────────
    def cross_correlation(self, signal: np.ndarray,
                          ref_pulse: np.ndarray) -> dict:
        """
        Cross-correlate measured signal with reference pulse.
        Peaks in cross-correlation correspond to echoes.
        """
        xcorr = np.correlate(signal, ref_pulse, mode="full")
        lags  = np.arange(-(len(ref_pulse) - 1), len(signal)) * self.gen.dt_ps

        # Only look at positive lags (causal)
        pos        = lags >= 0
        xcorr_pos  = xcorr[pos]
        lags_pos   = lags[pos]
        norm_xcorr = xcorr_pos / xcorr_pos.max()

        peaks, props = find_peaks(norm_xcorr,
                                  prominence=0.05,
                                  distance=int(0.5 / self.gen.dt_ps))

        result = {"method": "Cross-Correlation", "peaks_ps": [],
                  "delta_t_ps": None, "thickness_um": None, "confidence": 0.0}

        if len(peaks) >= 2:
            t_pk = lags_pos[peaks]
            Δt   = t_pk[1] - t_pk[0]
            d    = self._thickness_from_delay(Δt)
            result.update({"peaks_ps": t_pk[:4].tolist(),
                           "delta_t_ps": Δt, "thickness_um": d,
                           "confidence": float(props["prominences"][1] /
                                               norm_xcorr.max())})
        return result, xcorr_pos, lags_pos

    # ── Method 3: Frequency-domain phase analysis ─────────────────────────────
    def frequency_domain(self, signal: np.ndarray,
                         ref_pulse: np.ndarray) -> dict:
        """
        Extract coating thickness from the phase of the transfer function
        H(f) = S(f) / E_ref(f).
        The coating introduces a periodic phase modulation: Δφ = 2π·f·Δt.
        """
        S   = fft(signal,  n=self.gen.N)
        E   = fft(ref_pulse, n=self.gen.N)
        f   = fftfreq(self.gen.N, d=self.dt)        # [Hz]

        # Avoid division by near-zero
        mask = np.abs(E) > 0.01 * np.abs(E).max()
        H    = np.where(mask, S / E, 0 + 0j)
        φ    = np.unwrap(np.angle(H))

        # Positive frequencies 0.2–3 THz
        fTHz    = f * 1e-12
        band    = (fTHz > 0.2) & (fTHz < 3.0)
        f_band  = fTHz[band]
        φ_band  = φ[band]

        # Linear fit: φ ≈ -2π·f·Δt  (slope = -2π·Δt)
        result = {"method": "Frequency Domain", "delta_t_ps": None,
                  "thickness_um": None, "confidence": 0.0,
                  "slope": None}
        if f_band.size > 10:
            coeffs  = np.polyfit(f_band, φ_band, 1)
            slope   = coeffs[0]           # ps / THz
            Δt_fit  = -slope / (2 * np.pi)   # [ps]
            if Δt_fit > 0:
                d = self._thickness_from_delay(Δt_fit)
                r2 = self._r_squared(f_band, φ_band, np.polyval(coeffs, f_band))
                result.update({"delta_t_ps": Δt_fit,
                               "thickness_um": d,
                               "confidence": r2,
                               "slope": slope})
        return result, f_band, φ_band

    def _thickness_from_delay(self, delta_t_ps: float) -> float:
        """d [µm] = (c · Δt) / (2n)"""
        n = self.mat.refractive_index
        d_m = (C_LIGHT * delta_t_ps * 1e-12) / (2 * n)
        return d_m * 1e6   # → µm

    @staticmethod
    def _r_squared(x, y, y_fit) -> float:
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ── Batch thickness sweep ─────────────────────────────────────────────────────
def thickness_sweep(thicknesses_um: list[float],
                    coating: Material,
                    core: Material,
                    params: SimulationParams) -> list[dict]:
    """
    Run the simulation for multiple true coating thicknesses
    and return measured vs. true comparison data.
    """
    results = []
    for d_true in thicknesses_um:
        sample   = TabletSample(coating, core, d_true)
        gen      = THzPulseGenerator(params)
        ref      = gen.reference_pulse()
        builder  = SampleSignalBuilder(gen, sample)
        sig      = builder.build()
        analyzer = CoatingThicknessAnalyzer(gen, coating)

        pk  = analyzer.peak_detection(sig, ref)
        cc_res, _, _ = analyzer.cross_correlation(sig, ref)
        fd_res, _, _ = analyzer.frequency_domain(sig, ref)

        results.append({
            "true_um":       d_true,
            "peak_det_um":   pk["thickness_um"],
            "xcorr_um":      cc_res["thickness_um"],
            "freq_dom_um":   fd_res["thickness_um"],
            "expected_dt":   sample.expected_delay_ps,
        })
    return results


# ── Visualisation ─────────────────────────────────────────────────────────────
def plot_results(sample: TabletSample,
                 params: SimulationParams,
                 save_path: Optional[str] = None):
    """
    Comprehensive 6-panel plot:
      1. Reference THz pulse
      2. Reflected signal with echo annotations
      3. Cross-correlation
      4. Frequency spectrum
      5. Transfer-function phase
      6. Thickness sweep accuracy
    """
    gen      = THzPulseGenerator(params)
    ref      = gen.reference_pulse(t0_ps=5.0)
    builder  = SampleSignalBuilder(gen, sample)
    sig      = builder.build()
    analyzer = CoatingThicknessAnalyzer(gen, sample.coating)

    pk            = analyzer.peak_detection(sig, ref)
    cc_res, xcorr, lags = analyzer.cross_correlation(sig, ref)
    fd_res, f_band, phi = analyzer.frequency_domain(sig, ref)

    # Sweep
    thicknesses = np.linspace(20, 500, 25).tolist()
    sweep = thickness_sweep(thicknesses, sample.coating, sample.core, params)
    t_true = [r["true_um"] for r in sweep]
    t_pk   = [r["peak_det_um"] or 0 for r in sweep]
    t_cc   = [r["xcorr_um"]    or 0 for r in sweep]
    t_fd   = [r["freq_dom_um"] or 0 for r in sweep]

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
    fig.suptitle("THz Pulse Simulation — Tablet Coating Thickness Analysis",
                 fontsize=16, fontweight="bold", color="#e6edf3", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97,
                           top=0.93, bottom=0.06)

    axes_styles = dict(facecolor="#161b22",
                       labelcolor="#8b949e",
                       titlecolor="#e6edf3")

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(axes_styles["facecolor"])
        ax.tick_params(colors="#8b949e", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.set_title(title,  color=axes_styles["titlecolor"], fontsize=9, pad=6)
        ax.set_xlabel(xlabel, color=axes_styles["labelcolor"], fontsize=8)
        ax.set_ylabel(ylabel, color=axes_styles["labelcolor"], fontsize=8)
        ax.grid(True, color="#21262d", linewidth=0.5, linestyle="--")

    # Panel 1 — Reference pulse
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(gen.t, ref, color="#58a6ff", lw=1.5, label="Reference pulse")
    ax1.axvline(5.0, color="#3fb950", lw=0.8, linestyle=":", alpha=0.7)
    style_ax(ax1, "THz Reference Pulse", "Time (ps)", "Amplitude (norm.)")
    ax1.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3",
               edgecolor="#30363d")

    # Panel 2 — Reflected signal with echo markers
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.plot(gen.t, sig, color="#f0883e", lw=1.2, label="Reflected signal", zorder=3)
    ax2.plot(gen.t, ref * 0.15, color="#58a6ff", lw=0.8, alpha=0.5,
             linestyle="--", label="Reference (scaled)")
    # Mark detected peaks
    if pk["peaks_ps"]:
        for i, tp in enumerate(pk["peaks_ps"][:2]):
            ax2.axvline(tp, color="#3fb950" if i == 0 else "#bc8cff",
                        lw=1.2, linestyle="--", alpha=0.85)
            ax2.text(tp + 0.3, sig.max() * 0.75,
                     f"Echo {i+1}\n{tp:.2f} ps",
                     color="#3fb950" if i == 0 else "#bc8cff",
                     fontsize=7)
        if pk["delta_t_ps"]:
            mid = (pk["peaks_ps"][0] + pk["peaks_ps"][1]) / 2
            ax2.annotate("", xy=(pk["peaks_ps"][1], sig.max() * 0.6),
                         xytext=(pk["peaks_ps"][0], sig.max() * 0.6),
                         arrowprops=dict(arrowstyle="<->", color="#ffa657", lw=1.5))
            ax2.text(mid, sig.max() * 0.63,
                     f"Δt={pk['delta_t_ps']:.3f} ps",
                     ha="center", color="#ffa657", fontsize=8)
    style_ax(ax2, f"Reflected THz Signal — Coating: {sample.coating.name}, "
                  f"True thickness: {sample.coating_thickness_um:.0f} µm",
             "Time (ps)", "Amplitude (a.u.)")
    ax2.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3",
               edgecolor="#30363d")

    # Panel 3 — Cross-correlation
    ax3 = fig.add_subplot(gs[1, 0])
    norm = xcorr.max() if xcorr.max() > 0 else 1
    ax3.plot(lags, xcorr / norm, color="#bc8cff", lw=1.2)
    if cc_res["peaks_ps"]:
        for i, tp in enumerate(cc_res["peaks_ps"][:2]):
            ax3.axvline(tp, color=["#3fb950", "#f0883e"][i],
                        lw=1.0, linestyle=":", alpha=0.9)
    style_ax(ax3, "Cross-Correlation", "Lag (ps)", "Norm. Xcorr")

    # Panel 4 — Frequency spectrum
    ax4 = fig.add_subplot(gs[1, 1])
    from scipy.fft import fft, fftfreq
    f_ax  = fftfreq(gen.N, d=gen.dt_ps * 1e-12) * 1e-12   # THz
    S_abs = np.abs(fft(sig))
    R_abs = np.abs(fft(ref))
    pos   = f_ax > 0
    ax4.semilogy(f_ax[pos], R_abs[pos], color="#58a6ff",
                 lw=1.2, label="Reference")
    ax4.semilogy(f_ax[pos], S_abs[pos], color="#f0883e",
                 lw=1.0, alpha=0.8, label="Signal")
    ax4.set_xlim(0, 4)
    style_ax(ax4, "Frequency Spectrum", "Frequency (THz)", "Amplitude (a.u.)")
    ax4.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3",
               edgecolor="#30363d")

    # Panel 5 — Transfer-function phase
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(f_band, phi, s=3, color="#58a6ff", alpha=0.5, label="Phase")
    if fd_res["slope"] is not None:
        phi_fit = np.polyfit(f_band, phi, 1)
        ax5.plot(f_band, np.polyval(phi_fit, f_band),
                 color="#ffa657", lw=1.5, label=f"Linear fit (R²={fd_res['confidence']:.3f})")
    style_ax(ax5, "Transfer Function Phase", "Frequency (THz)", "Phase (rad)")
    ax5.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3",
               edgecolor="#30363d")

    # Panel 6 — Thickness sweep accuracy
    ax6 = fig.add_subplot(gs[2, :])
    ax6.plot(t_true, t_true, color="#8b949e", lw=1.5, linestyle="--",
             label="Ideal (true = measured)")
    ax6.plot(t_true, t_pk, "o-", color="#3fb950", ms=4, lw=1.2,
             label="Peak Detection")
    ax6.plot(t_true, t_cc, "s-", color="#bc8cff", ms=4, lw=1.2,
             label="Cross-Correlation")
    ax6.plot(t_true, t_fd, "^-", color="#ffa657", ms=4, lw=1.2,
             label="Frequency Domain")
    ax6.set_xlim(0, max(thicknesses) * 1.05)
    style_ax(ax6,
             f"Coating Thickness Measurement Accuracy — {sample.coating.name} / {sample.core.name}",
             "True Thickness (µm)", "Measured Thickness (µm)")
    ax6.legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3",
               edgecolor="#30363d", ncol=4)

    # Summary box
    summary = (
        f"Peak Detection : {pk['thickness_um'] or 0:.1f} µm  (Δt = {pk['delta_t_ps'] or 0:.3f} ps)\n"
        f"Cross-Correlation: {cc_res['thickness_um'] or 0:.1f} µm  (Δt = {cc_res['delta_t_ps'] or 0:.3f} ps)\n"
        f"Frequency Domain: {fd_res['thickness_um'] or 0:.1f} µm  (Δt = {fd_res['delta_t_ps'] or 0:.3f} ps)\n"
        f"True thickness  : {sample.coating_thickness_um:.1f} µm  (expected Δt = {sample.expected_delay_ps:.3f} ps)"
    )
    fig.text(0.5, 0.01, summary, ha="center", va="bottom",
             fontsize=8.5, color="#e6edf3",
             bbox=dict(facecolor="#161b22", edgecolor="#30363d",
                       boxstyle="round,pad=0.4"))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[✓] Figure saved → {save_path}")
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" THz Pulse Simulation — Tablet Coating Analysis")
    print("=" * 60)

    # ── Simulation parameters ──
    params = SimulationParams(
        time_window_ps  = 60.0,
        time_step_fs    = 10.0,
        pulse_width_ps  = 0.3,
        center_freq_THz = 1.0,
        snr_db          = 40.0,
    )

    # ── Sample definition ──
    coating = MATERIALS["hpmc"]       # swap with "eudragit_l", "ec", "pvp" …
    core    = MATERIALS["tablet_core"]
    sample  = TabletSample(coating, core, coating_thickness_um=150.0)

    print(f"\n  Coating material : {coating.name}  (n={coating.refractive_index})")
    print(f"  Core material    : {core.name}  (n={core.refractive_index})")
    print(f"  True thickness   : {sample.coating_thickness_um:.1f} µm")
    print(f"  Expected Δt      : {sample.expected_delay_ps:.4f} ps")
    print(f"  Core reflectance : {sample.core_reflectance:.4f}")

    # ── Run simulation ──
    gen      = THzPulseGenerator(params)
    ref      = gen.reference_pulse()
    builder  = SampleSignalBuilder(gen, sample)
    sig      = builder.build()
    analyzer = CoatingThicknessAnalyzer(gen, coating)

    print("\n  ── Analysis results ──")
    pk = analyzer.peak_detection(sig, ref)
    print(f"  [Peak Detection]    Δt={pk['delta_t_ps']:.4f} ps  →  "
          f"{pk['thickness_um']:.2f} µm  (confidence={pk['confidence']:.3f})")

    cc_res, _, _ = analyzer.cross_correlation(sig, ref)
    cc_dt = cc_res['delta_t_ps'] or 0.0
    cc_th = cc_res['thickness_um'] or 0.0
    print(f"  [Cross-Correlation] Δt={cc_dt:.4f} ps  →  "
          f"{cc_th:.2f} µm  (confidence={cc_res['confidence']:.3f})")

    fd_res, _, _ = analyzer.frequency_domain(sig, ref)
    fd_dt = fd_res['delta_t_ps'] or 0.0
    fd_th = fd_res['thickness_um'] or 0.0
    print(f"  [Frequency Domain]  Δt={fd_dt:.4f} ps  →  "
          f"{fd_th:.2f} µm  (R²={fd_res['confidence']:.4f})")

    # ── Batch sweep ──
    print("\n  Running thickness sweep (20–500 µm) …")
    sweep = thickness_sweep(np.linspace(20, 500, 10).tolist(),
                            coating, core, params)
    print(f"  {'True (µm)':>10}  {'Peak Det':>10}  {'XCorr':>10}  {'Freq Dom':>10}")
    for r in sweep:
        print(f"  {r['true_um']:>10.1f}  "
              f"{(r['peak_det_um'] or 0):>10.1f}  "
              f"{(r['xcorr_um']    or 0):>10.1f}  "
              f"{(r['freq_dom_um'] or 0):>10.1f}")

    # ── Plot ──
    print("\n  Generating plots …")
    fig = plot_results(sample, params,
                   save_path=r"C:\Users\shamb\Desktop\THz_project\thz_results.png")
    plt.show()
    print("\n[Done]")


if __name__ == "__main__":
    main()
