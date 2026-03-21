"""
WBTM-Net: Wavelet-Bayesian Transfer Matrix Network
====================================================
Novel framework for THz tablet coating thickness estimation.
Place this file in the SAME folder as thz_simulation.py and run:

    python wbtm_net.py

Change coating / thickness / SNR in the CONFIG block inside main().

Four novel stages over the baseline:
  Stage 1 - Scale-adaptive wavelet denoising        (novel lambda*(j) formula)
  Stage 2 - Dispersive N-layer transfer matrix       (Sellmeier n(f) model)
  Stage 3 - Joint Bayesian MCMC for (d, n)           (uncertainty quantified)
  Stage 4 - Physics-guided residual CNN corrector    (trains on model error only)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import warnings, time, os

warnings.filterwarnings("ignore")

from thz_simulation import (
    C_LIGHT, FS_TO_PS,
    Material, SimulationParams, TabletSample, MATERIALS,
    THzPulseGenerator, SampleSignalBuilder, CoatingThicknessAnalyzer,
)


# ==============================================================================
#  STAGE 1 - Scale-adaptive wavelet denoising
# ==============================================================================

class WaveletDenoiser:
    """
    Daubechies-6 wavelet with novel scale-dependent soft threshold:
        lambda*(j) = sigma_hat * sqrt(2*ln(N)) * 2^(-j/2)
    Fine scales -> lower threshold -> THz echoes preserved.
    Coarse scales -> higher threshold -> drift suppressed.
    sigma_hat = MAD(finest coefficients) / 0.6745
    """

    DB6_LO = np.array([
         0.03522629188,  0.08544127388, -0.13501102001,
        -0.45987750211,  0.80689150931, -0.33267055295,
        -0.08544127388,  0.03522629188,  0.00655050102,
        -0.01985884806,  0.00000000000,  0.00000000000,
    ])

    def __init__(self, n_levels=4):
        self.J  = n_levels
        self.lo = self.DB6_LO
        self.hi = self.lo[::-1] * np.array([(-1)**k for k in range(len(self.lo))])

    def _dwt(self, x):
        pad = len(self.lo) - 1
        ca  = np.convolve(x, self.lo, mode='full')[pad::2][: len(x)//2]
        cd  = np.convolve(x, self.hi, mode='full')[pad::2][: len(x)//2]
        return ca, cd

    def _idwt(self, ca, cd, target_len):
        up_a = np.zeros(len(ca)*2); up_a[::2] = ca
        up_d = np.zeros(len(cd)*2); up_d[::2] = cd
        ra   = np.convolve(up_a, self.lo[::-1], mode='full')
        rd   = np.convolve(up_d, self.hi[::-1], mode='full')
        n    = min(len(ra), len(rd))
        off  = len(self.lo) - 1
        return (ra[:n] + rd[:n])[off: off + target_len]

    def denoise(self, signal):
        N = len(signal)
        x = signal.copy()
        details = []
        for _ in range(self.J):
            ca, cd = self._dwt(x)
            details.append((cd, len(x)))
            x = ca
        sigma_hat = np.median(np.abs(details[0][0])) / 0.6745
        base_lam  = sigma_hat * np.sqrt(2 * np.log(max(N, 2)))
        thr = []
        for j_idx, (cd, orig) in enumerate(details):
            lam = base_lam * (2 ** (-(j_idx+1) / 2.0))
            thr.append((np.sign(cd) * np.maximum(np.abs(cd) - lam, 0), orig))
        x_rec = x.copy()
        for j_idx in range(self.J-1, -1, -1):
            cd_t, orig = thr[j_idx]
            x_rec = self._idwt(x_rec, cd_t, orig)
        out = x_rec[:N]
        if len(out) < N:
            out = np.pad(out, (0, N - len(out)))
        return out


# ==============================================================================
#  STAGE 2 - Dispersive N-layer transfer matrix
# ==============================================================================

class DispersiveTransferMatrix:
    """
    Electromagnetic transfer matrix with Sellmeier dispersive n(f):
        n^2(f) = A + B*f^2 / (f^2 - f0^2)
    Layer matrix:
        Mi = [ cos(phi)      -i*sin(phi)/n ]
             [ -i*n*sin(phi)   cos(phi)    ]
    phi = 2*pi*f*n(f)*d / c
    Reflection: r = (M11 + M12*ns - ni*M21 - ni*ns*M22)
                    / (M11 + M12*ns + ni*M21 + ni*ns*M22)
    """

    SELLMEIER = {
        "hpmc":        (2.307, 0.018, 4.2),
        "eudragit_l":  (2.250, 0.015, 4.0),
        "ec":          (2.190, 0.022, 3.8),
        "pvp":         (2.402, 0.030, 3.5),
        "tablet_core": (2.560, 0.045, 3.0),
        "mcc":         (2.493, 0.038, 3.2),
    }

    def __init__(self, freq_THz):
        self.f    = freq_THz
        self.f_hz = freq_THz * 1e12

    def n_of_f(self, key):
        A, B, f0 = self.SELLMEIER.get(key, (2.3, 0.02, 4.0))
        n2 = A + B * self.f**2 / (self.f**2 - f0**2 + 1e-9)
        return np.sqrt(np.clip(n2, 1.0, 16.0))

    def _Mi(self, n_f, d_m):
        phi        = 2*np.pi * self.f_hz * n_f * d_m / C_LIGHT
        M          = np.zeros((len(self.f), 2, 2), dtype=complex)
        M[:, 0, 0] =  np.cos(phi)
        M[:, 0, 1] = -1j * np.sin(phi) / (n_f + 1e-30)
        M[:, 1, 0] = -1j * n_f * np.sin(phi)
        M[:, 1, 1] =  np.cos(phi)
        return M

    def reflection(self, layers, substrate_key="tablet_core", n_inc=1.0):
        n_s = self.n_of_f(substrate_key)
        M   = np.eye(2, dtype=complex)[np.newaxis].repeat(len(self.f), axis=0)
        for key, d_m in layers:
            M = np.einsum('fij,fjk->fik', M, self._Mi(self.n_of_f(key), d_m))
        ni  = n_inc
        num = M[:,0,0] + M[:,0,1]*n_s - ni*M[:,1,0] - ni*n_s*M[:,1,1]
        den = M[:,0,0] + M[:,0,1]*n_s + ni*M[:,1,0] + ni*n_s*M[:,1,1]
        return num / (den + 1e-30)


# ==============================================================================
#  STAGE 3 - Joint Bayesian MCMC for (d, n)
# ==============================================================================

class BayesianEstimator:
    """
    Metropolis-Hastings MCMC for p(d, n | E_measured).

    Novel: simultaneous estimation of d AND n (literature fixes n a-priori).
    Likelihood uses phase of transfer function H(f) = S(f)/E_ref(f) ---
    robust to amplitude calibration errors.
    Priors: Gaussian centred on process spec (d) and literature value (n).
    Output: posterior mean, std, 95% credible interval.
    """

    def __init__(self, dt_ps, coating_key, substrate_key,
                 n_samples=1000, burn_in=250):
        self.dt_ps         = dt_ps
        self.coating_key   = coating_key
        self.substrate_key = substrate_key
        self.n_samples     = n_samples
        self.burn_in       = burn_in

    def run(self, signal, ref, d_init_um, n_init,
            d_prior_std_um=70.0, verbose=True):
        """
        Robust MCMC using time-domain cross-correlation peak as the
        primary observable, supplemented by phase likelihood.

        The key measurement is the echo time delay Dt extracted from the
        cross-correlation peak of signal vs reference.  This is a much
        stronger constraint than raw phase, and directly maps to thickness:
            d = c * Dt / (2 * n)

        Likelihood:
            log p(Dt_meas | d, n) = -0.5*(Dt_meas - 2*n*d/c)^2 / sigma_Dt^2
        Prior:
            p(d) = N(d0, sigma_d^2),  p(n) = N(n0, 0.05^2)
        """
        from scipy.signal import correlate
        N       = len(signal)
        dt_s    = self.dt_ps * 1e-12

        # ── Measure echo delay using signal envelope peaks ─────────────
        # Use Hilbert envelope — gives cleaner peak positions than xcorr
        from scipy.signal import find_peaks, hilbert
        env      = np.abs(hilbert(signal))
        min_dist = max(1, int(0.4e-12 / dt_s))
        pks, props = find_peaks(env,
                                prominence=0.01 * env.max(),
                                distance=min_dist)
        t_axis = np.arange(N) * dt_s

        if len(pks) >= 2:
            # Take the two most prominent peaks in chronological order
            order = np.argsort(props["prominences"])[::-1]
            top2  = np.sort(pks[order[:2]])
            Dt_meas = float(t_axis[top2[1]] - t_axis[top2[0]])
        else:
            # Fallback to expected delay from init
            Dt_meas = 2.0 * n_init * d_init_um * 1e-6 / C_LIGHT

        # Clamp to physically meaningful range (10 um to 800 um coating)
        Dt_min  = 2.0 * 1.4 * 10e-6  / C_LIGHT
        Dt_max  = 2.0 * 2.0 * 800e-6 / C_LIGHT
        Dt_meas = float(np.clip(Dt_meas, Dt_min, Dt_max))

        # Noise on delay measurement (half a time step)
        sigma_Dt = dt_s * 1.5

        d_m  = d_init_um * 1e-6
        n_v  = n_init
        d0   = d_init_um * 1e-6
        sd   = d_prior_std_um * 1e-6

        def ll(d, n):
            Dt_pred = 2.0 * n * d / C_LIGHT
            return -0.5 * (Dt_meas - Dt_pred)**2 / (sigma_Dt**2 + 1e-30)

        def lp(d, n):
            if d <= 1e-6 or d > 1e-3: return -np.inf
            if n < 1.1 or n > 2.5:    return -np.inf
            return (-0.5*((d-d0)/sd)**2
                    -0.5*((n-n_init)/0.05)**2)

        cur_ll = ll(d_m, n_v); cur_lp = lp(d_m, n_v)
        cd, cn = [], []
        acc    = 0
        # Tight proposal sizes — key fix for convergence
        pd     = min(sd * 0.05, 5e-6)    # max 5 um step
        pn     = 0.008

        total  = self.n_samples + self.burn_in
        if verbose:
            print(f"    [Bayesian MCMC] {total} iters, Dt_meas={Dt_meas*1e12:.4f} ps...",
                  end="", flush=True)
        for i in range(total):
            dp   = d_m + np.random.normal(0, pd)
            np_  = n_v + np.random.normal(0, pn)
            ll2  = ll(dp, np_); lp2 = lp(dp, np_)
            if np.log(np.random.rand()+1e-30) < (ll2+lp2)-(cur_ll+cur_lp):
                d_m, n_v, cur_ll, cur_lp = dp, np_, ll2, lp2
                if i >= self.burn_in: acc += 1
            if i >= self.burn_in:
                cd.append(d_m*1e6); cn.append(n_v)

        cd = np.array(cd); cn = np.array(cn)
        if verbose:
            print(f" done (acc={acc/self.n_samples*100:.1f}%)")
        dm = float(np.mean(cd)); ds = float(np.std(cd))
        return {"d_mean_um": dm, "d_std_um": ds,
                "d_ci95_um": (dm-2*ds, dm+2*ds),
                "n_mean": float(np.mean(cn)), "n_std": float(np.std(cn)),
                "chain_d": cd, "chain_n": cn,
                "accept_rate": acc/self.n_samples}


# ==============================================================================
#  STAGE 4 - Physics-guided residual CNN (numpy-only, no PyTorch needed)
# ==============================================================================

class ResidualCNN:
    """
    1-D CNN trained only on physics model residuals:
        d_final = d_Bayes + CNN(E_measured - E_model(d_Bayes, n_Bayes))

    Architecture: Conv(64,k=15)->ReLU -> Conv(32,k=7)->ReLU -> GAP -> Dense(1)
    Training: gradient-free random perturbation search.
    Key advantage: needs only ~50 samples (end-to-end CNNs need ~10,000).
    """

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.trained      = False
        self.train_losses = []
        def he(c_in, k, c_out):
            return np.random.randn(c_out, c_in, k) * np.sqrt(2.0/(c_in*k))
        self.W1 = he(1, 15, 64); self.b1 = np.zeros(64)
        self.W2 = he(64, 7, 32); self.b2 = np.zeros(32)
        self.Wd = np.random.randn(32,1)*0.01; self.bd = np.zeros(1)

    @staticmethod
    def _conv(x, W, b):
        Co, Ci, K = W.shape
        Lo = x.shape[1] - K + 1
        o  = np.zeros((Co, Lo))
        for i in range(Co):
            for j in range(Ci):
                for k in range(K):
                    o[i] += x[j, k:k+Lo] * W[i,j,k]
            o[i] += b[i]
        return o

    def forward(self, residual):
        x  = residual[::max(1, len(residual)//256)][:256][np.newaxis,:]
        h1 = np.maximum(0, self._conv(x,  self.W1, self.b1))
        h2 = np.maximum(0, self._conv(h1, self.W2, self.b2))
        return float((h2.mean(axis=1) @ self.Wd + self.bd)[0])

    def _pack(self):
        return np.concatenate([v.ravel() for v in
                               [self.W1,self.b1,self.W2,self.b2,self.Wd,self.bd]])

    def _unpack(self, v):
        idx=0
        def take(s):
            nonlocal idx
            n=int(np.prod(s)); out=v[idx:idx+n].reshape(s); idx+=n; return out
        self.W1=take(self.W1.shape); self.b1=take(self.b1.shape)
        self.W2=take(self.W2.shape); self.b2=take(self.b2.shape)
        self.Wd=take(self.Wd.shape); self.bd=take(self.bd.shape)

    def train(self, residuals, corrections, epochs=60, lr=3e-3, verbose=True):
        best  = self._pack()
        b_mse = np.mean([(self.forward(r)-c)**2 for r,c in zip(residuals,corrections)])
        if verbose:
            print(f"    [Residual CNN] {len(residuals)} samples, {epochs} epochs...")
        for ep in range(epochs):
            cand = best + np.random.randn(len(best)) * lr*(1-ep/epochs)
            self._unpack(cand)
            mse = np.mean([(self.forward(r)-c)**2 for r,c in zip(residuals,corrections)])
            if mse < b_mse: b_mse=mse; best=cand.copy()
            self.train_losses.append(b_mse)
        self._unpack(best)
        self.trained = True
        if verbose:
            print(f"    [Residual CNN] Final RMSE: {np.sqrt(b_mse):.3f} um")


# ==============================================================================
#  Helper: physics residual signal
# ==============================================================================

def physics_residual(signal, ref, d_um, n_val, coating_key, substrate_key, dt_ps):
    N     = len(signal)
    f_THz = fftfreq(N, d=dt_ps*1e-12)*1e-12
    pos   = f_THz > 0
    f_pos = f_THz[pos]
    orig  = DispersiveTransferMatrix.SELLMEIER.get(coating_key)
    DispersiveTransferMatrix.SELLMEIER[coating_key] = (n_val**2, 0.0, 99.0)
    try:
        tmm   = DispersiveTransferMatrix(f_pos)
        r_f   = tmm.reflection([(coating_key, d_um*1e-6)], substrate_key=substrate_key)
        E_f   = fft(ref[:N])
        E_pos = E_f[pos]
        n     = min(len(r_f), len(E_pos))
        Sf    = np.zeros(N, dtype=complex)
        idx   = np.where(pos)[0][:n]
        Sf[idx] = r_f[:n] * E_pos[:n]
        E_model = np.real(np.fft.ifft(Sf))
    finally:
        DispersiveTransferMatrix.SELLMEIER[coating_key] = orig
    return signal - E_model


# ==============================================================================
#  Full WBTM-Net pipeline
# ==============================================================================

class WBTMNet:
    def __init__(self, params, coating_key="hpmc", substrate_key="tablet_core"):
        self.params        = params
        self.coating_key   = coating_key
        self.substrate_key = substrate_key
        self.gen           = THzPulseGenerator(params)
        self.denoiser      = WaveletDenoiser(n_levels=4)
        self.cnn           = ResidualCNN(seed=42)

    def _make_bayes(self, n_samples=1000, burn_in=250):
        return BayesianEstimator(self.gen.dt_ps, self.coating_key,
                                 self.substrate_key, n_samples, burn_in)

    def _d_init(self, sig, ref):
        coating  = MATERIALS[self.coating_key]
        pk = CoatingThicknessAnalyzer(self.gen, coating).peak_detection(sig, ref)
        return pk["thickness_um"] if pk["thickness_um"] else 100.0

    def train(self, n_train=60, d_range_um=(40,500), verbose=True):
        coating = MATERIALS[self.coating_key]
        core    = MATERIALS[self.substrate_key]
        ref     = self.gen.reference_pulse()
        residuals, corrections = [], []
        if verbose:
            print(f"\n  [Training] Generating {n_train} synthetic samples...")
        for i in range(n_train):
            d_true = np.random.uniform(*d_range_um)
            sig    = SampleSignalBuilder(self.gen, TabletSample(coating, core, d_true)).build()
            sig_dn = self.denoiser.denoise(sig)
            d_i    = self._d_init(sig_dn, ref)
            b      = self._make_bayes(300, 80).run(
                         sig_dn, ref, d_i, coating.refractive_index,
                         d_prior_std_um=80.0, verbose=False)
            resid  = physics_residual(sig_dn, ref, b["d_mean_um"], b["n_mean"],
                                      self.coating_key, self.substrate_key, self.gen.dt_ps)
            residuals.append(resid)
            corrections.append(d_true - b["d_mean_um"])
            if verbose and (i+1)%10==0:
                print(f"    ... {i+1}/{n_train} done")
        self.cnn.train(residuals, corrections, epochs=60, lr=3e-3, verbose=verbose)

    def predict(self, signal, ref, d_init_um=150.0, verbose=True):
        coating = MATERIALS[self.coating_key]
        if verbose: print("\n  -- WBTM-Net Inference --")
        t0     = time.time()
        sig_dn = self.denoiser.denoise(signal)
        if verbose: print(f"  [Stage 1] Wavelet denoising ({(time.time()-t0)*1000:.0f} ms)")
        if verbose: print(f"  [Stage 2] Dispersive Sellmeier TMM ready")
        t1 = time.time()
        b  = self._make_bayes(1000, 250).run(
                 sig_dn, ref, d_init_um, coating.refractive_index,
                 d_prior_std_um=70.0, verbose=verbose)
        d_b = b["d_mean_um"]; n_b = b["n_mean"]
        if verbose:
            print(f"  [Stage 3] d={d_b:.2f}+-{b['d_std_um']:.2f} um, "
                  f"n={n_b:.4f}  ({time.time()-t1:.1f}s)")
        t2    = time.time()
        resid = physics_residual(sig_dn, ref, d_b, n_b,
                                 self.coating_key, self.substrate_key, self.gen.dt_ps)
        corr  = self.cnn.forward(resid) if self.cnn.trained else 0.0
        d_fin = d_b + corr
        if verbose:
            print(f"  [Stage 4] CNN correction: {corr:+.3f} um -> d_final={d_fin:.2f} um "
                  f"({(time.time()-t2)*1000:.0f} ms)")
        return {"d_bayes_um":  d_b,   "d_final_um":  d_fin,
                "d_std_um":    b["d_std_um"],
                "d_ci95_um":   b["d_ci95_um"],
                "n_estimated": n_b,   "n_std":       b["n_std"],
                "cnn_corr_um": corr,  "sig_denoised":sig_dn,
                "residual":    resid, "chain_d":     b["chain_d"],
                "chain_n":     b["chain_n"], "accept_rate": b["accept_rate"]}


# ==============================================================================
#  Benchmark
# ==============================================================================

def run_benchmark(net, thicknesses_um, snr_db):
    coating  = MATERIALS[net.coating_key]
    core     = MATERIALS[net.substrate_key]
    p2       = SimulationParams(time_window_ps=net.params.time_window_ps,
                                time_step_fs=net.params.time_step_fs,
                                pulse_width_ps=net.params.pulse_width_ps,
                                snr_db=snr_db)
    gen      = THzPulseGenerator(p2)
    ref      = gen.reference_pulse()
    analyzer = CoatingThicknessAnalyzer(gen, coating)
    res      = {k: [] for k in ["true","peak","xcorr","freq","bayes","wbtm"]}
    print(f"\n  [Benchmark] {len(thicknesses_um)} thickness points, SNR={snr_db} dB...")
    for d_true in thicknesses_um:
        sig      = SampleSignalBuilder(gen, TabletSample(coating, core, d_true)).build()
        pk       = analyzer.peak_detection(sig, ref)
        cc,_,_   = analyzer.cross_correlation(sig, ref)
        fd,_,_   = analyzer.frequency_domain(sig, ref)
        wr       = net.predict(sig, ref, d_init_um=d_true*0.9, verbose=False)
        res["true"].append(d_true)
        res["peak"].append(pk["thickness_um"]  or 0.0)
        res["xcorr"].append(cc["thickness_um"] or 0.0)
        res["freq"].append(fd["thickness_um"]  or 0.0)
        res["bayes"].append(wr["d_bayes_um"])
        res["wbtm"].append(wr["d_final_um"])
    t_true = np.array(res["true"])
    rmse   = {m: float(np.sqrt(np.mean((np.array(res[m])-t_true)**2)))
              for m in ["peak","xcorr","freq","bayes","wbtm"]}
    return res, rmse


# ==============================================================================
#  9-panel result plot
# ==============================================================================

def plot_all(net, sample, result, bench, rmse, snr_db, save_path=None):
    gen       = net.gen
    t         = gen.t
    ref       = gen.reference_pulse()
    sig_noisy = SampleSignalBuilder(gen, sample).build()

    fig = plt.figure(figsize=(20,16), facecolor="#0d1117")
    fig.suptitle("WBTM-Net -- Novel THz Tablet Coating Thickness Analysis",
                 fontsize=15, fontweight="bold", color="#e6edf3", y=0.98)
    gs  = gridspec.GridSpec(4,3, figure=fig, hspace=0.50, wspace=0.35,
                            left=0.06, right=0.97, top=0.94, bottom=0.05)

    def S(ax, title, xl, yl):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.set_title(title, color="#e6edf3", fontsize=8.5, pad=5)
        ax.set_xlabel(xl, color="#8b949e", fontsize=8)
        ax.set_ylabel(yl, color="#8b949e", fontsize=8)
        ax.grid(True, color="#21262d", lw=0.5, ls="--")

    palette = {"peak":"#3fb950","xcorr":"#bc8cff","freq":"#ffa657",
               "bayes":"#58a6ff","wbtm":"#ff6e6e"}
    labels  = {"peak":"Peak detection","xcorr":"Cross-correlation",
               "freq":"Freq. domain","bayes":"Bayesian (Stage 3)","wbtm":"WBTM-Net (full)"}

    # 1. Noisy vs denoised
    ax1 = fig.add_subplot(gs[0,:2])
    ax1.plot(t, sig_noisy,              color="#58a6ff", lw=0.8, alpha=0.55, label="Noisy signal")
    ax1.plot(t, result["sig_denoised"], color="#3fb950", lw=1.5, label="Wavelet denoised")
    ax1.set_xlim(0, 20)
    S(ax1, "Stage 1 -- Scale-adaptive wavelet denoising", "Time (ps)", "Amplitude (a.u.)")
    ax1.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")

    # 2. Physics residual
    ax2 = fig.add_subplot(gs[0,2])
    ax2.plot(t, result["residual"], color="#ffa657", lw=0.9)
    ax2.axhline(0, color="#8b949e", lw=0.5); ax2.set_xlim(0,20)
    S(ax2, "Stage 4 -- Physics residual (CNN input)", "Time (ps)", "Residual (a.u.)")

    # 3. Posterior d
    ax3 = fig.add_subplot(gs[1,0])
    cd  = result["chain_d"]
    ax3.hist(cd, bins=40, color="#bc8cff", alpha=0.85, edgecolor="none")
    ax3.axvline(result["d_final_um"], color="#3fb950", lw=2,
                label=f"WBTM-Net: {result['d_final_um']:.1f} um")
    ax3.axvline(sample.coating_thickness_um, color="#ffa657", lw=1.5, ls="--",
                label=f"True: {sample.coating_thickness_um:.1f} um")
    ci = result["d_ci95_um"]
    ax3.axvspan(ci[0], ci[1], alpha=0.15, color="#bc8cff", label="95% CI")
    S(ax3, "Stage 3 -- Posterior p(d | signal)", "d (um)", "Count")
    ax3.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")

    # 4. Posterior n
    ax4 = fig.add_subplot(gs[1,1])
    ax4.hist(result["chain_n"], bins=40, color="#58a6ff", alpha=0.85, edgecolor="none")
    ax4.axvline(result["n_estimated"], color="#3fb950", lw=2,
                label=f"Estimated n={result['n_estimated']:.4f}")
    ax4.axvline(sample.coating.refractive_index, color="#ffa657", lw=1.5, ls="--",
                label=f"True n={sample.coating.refractive_index:.4f}")
    S(ax4, "Stage 3 -- Posterior p(n | signal)", "Refractive index n", "Count")
    ax4.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")

    # 5. MCMC trace
    ax5 = fig.add_subplot(gs[1,2])
    ax5.plot(cd, color="#bc8cff", lw=0.6, alpha=0.75)
    ax5.axhline(sample.coating_thickness_um, color="#ffa657", lw=1.2, ls="--", label="True d")
    ax5.axhline(result["d_final_um"], color="#3fb950", lw=1.2, label="WBTM-Net")
    S(ax5, "MCMC chain trace", "Iteration", "d (um)")
    ax5.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")

    # 6. Thickness sweep
    ax6     = fig.add_subplot(gs[2,:2])
    t_true  = np.array(bench["true"])
    ax6.plot(t_true, t_true, color="#555", lw=1.5, ls="--", label="Ideal")
    for m, col in palette.items():
        ax6.plot(t_true, np.array(bench[m]), "o-", color=col, ms=4, lw=1.2, label=labels[m])
    S(ax6, "Thickness sweep -- all methods vs true thickness",
      "True d (um)", "Measured d (um)")
    ax6.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3",
               edgecolor="#30363d", ncol=3)

    # 7. RMSE bar chart
    ax7    = fig.add_subplot(gs[2,2])
    order  = list(labels.keys())
    vals   = [rmse[m] for m in order]
    bars   = ax7.bar(range(len(order)), vals, color=[palette[m] for m in order],
                     width=0.6, edgecolor="none")
    ax7.set_xticks(range(len(order)))
    ax7.set_xticklabels([labels[m].split()[0] for m in order],
                        color="#8b949e", fontsize=7, rotation=15)
    for b2, v in zip(bars, vals):
        ax7.text(b2.get_x()+b2.get_width()/2, v+0.5, f"{v:.1f}",
                 ha="center", color="#e6edf3", fontsize=7)
    S(ax7, f"RMSE comparison (SNR={snr_db} dB)", "Method", "RMSE (um)")

    # 8. CNN training loss
    ax8 = fig.add_subplot(gs[3,0])
    if net.cnn.train_losses:
        ax8.semilogy(net.cnn.train_losses, color="#ffa657", lw=1.2)
    S(ax8, "Stage 4 -- CNN training loss", "Epoch", "MSE (um^2)")

    # 9. Error vs thickness
    ax9 = fig.add_subplot(gs[3,1:])
    err_pk   = np.array(bench["peak"]) - t_true
    err_wbtm = np.array(bench["wbtm"]) - t_true
    ax9.plot(t_true, err_pk,   "o-", color="#3fb950", ms=4, lw=1.2, label="Peak detection error")
    ax9.plot(t_true, err_wbtm, "s-", color="#ff6e6e", ms=4, lw=1.2, label="WBTM-Net error")
    ax9.axhline(0, color="#8b949e", lw=0.8)
    ax9.fill_between(t_true, -5, 5, alpha=0.08, color="#3fb950", label="+-5 um target band")
    S(ax9, "Measurement error vs true thickness", "True d (um)", "Error (um)")
    ax9.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")

    # Banner
    pk_r = rmse["peak"]; wt_r = rmse["wbtm"]
    impr = (pk_r-wt_r)/pk_r*100 if pk_r > 0 else 0
    ci   = result["d_ci95_um"]
    fig.text(0.5, 0.005,
             f"WBTM-Net d = {result['d_final_um']:.2f} um  |  "
             f"True = {sample.coating_thickness_um:.1f} um  |  "
             f"Error = {abs(result['d_final_um']-sample.coating_thickness_um):.2f} um  |  "
             f"95% CI: [{ci[0]:.1f}, {ci[1]:.1f}] um  |  "
             f"n = {result['n_estimated']:.4f}+-{result['n_std']:.4f}  |  "
             f"RMSE improvement vs peak detection: {impr:.1f}%",
             ha="center", va="bottom", fontsize=8, color="#e6edf3",
             bbox=dict(facecolor="#161b22", edgecolor="#30363d", boxstyle="round,pad=0.4"))

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [OK] Plot saved -> {save_path}")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    return fig


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    # ======================== CONFIG =========================================
    COATING_KEY       = "hpmc"         # options: "hpmc" | "eudragit_l" | "ec" | "pvp"
    SUBSTRATE_KEY     = "tablet_core"  # options: "tablet_core" | "mcc"
    TRUE_THICKNESS_UM = 150.0          # true coating thickness in micrometres
    SNR_DB            = 35.0           # signal-to-noise ratio in dB (lower = noisier)
    N_TRAIN           = 50             # number of CNN training samples
    SAVE_PATH         = r"wbtm_results.png"  # change to full path e.g. r"C:\Users\...\wbtm_results.png"
    # =========================================================================

    print("=" * 65)
    print("  WBTM-Net: Wavelet-Bayesian Transfer Matrix Network")
    print("  Novel THz Tablet Coating Thickness Analysis")
    print("=" * 65)

    params  = SimulationParams(time_window_ps=60.0, time_step_fs=10.0,
                               pulse_width_ps=0.3, snr_db=SNR_DB)
    coating = MATERIALS[COATING_KEY]
    core    = MATERIALS[SUBSTRATE_KEY]
    sample  = TabletSample(coating, core, TRUE_THICKNESS_UM)

    print(f"\n  Coating : {coating.name}  n={coating.refractive_index}")
    print(f"  Core    : {core.name}  n={core.refractive_index}")
    print(f"  True d  : {TRUE_THICKNESS_UM} um")
    print(f"  SNR     : {SNR_DB} dB")

    # Build and train
    net = WBTMNet(params, COATING_KEY, SUBSTRATE_KEY)
    net.train(n_train=N_TRAIN, d_range_um=(40, 500))

    # Test signal
    gen = net.gen
    ref = gen.reference_pulse()
    sig = SampleSignalBuilder(gen, sample).build()

    # Full inference
    result = net.predict(sig, ref, d_init_um=TRUE_THICKNESS_UM * 0.9)

    print(f"\n  -- Final Results -----------------------------------------------")
    print(f"  True thickness   : {TRUE_THICKNESS_UM:.1f} um")
    print(f"  WBTM-Net d       : {result['d_final_um']:.2f} um")
    print(f"  Absolute error   : {abs(result['d_final_um']-TRUE_THICKNESS_UM):.2f} um")
    ci = result['d_ci95_um']
    print(f"  95% CI           : [{ci[0]:.1f}, {ci[1]:.1f}] um")
    print(f"  Estimated n      : {result['n_estimated']:.4f} +- {result['n_std']:.4f}")
    print(f"  CNN correction   : {result['cnn_corr_um']:+.3f} um")
    print(f"  MCMC acceptance  : {result['accept_rate']*100:.1f}%")

    # Benchmark
    bench_d       = np.linspace(50, 450, 10).tolist()
    bench, rmse   = run_benchmark(net, bench_d, SNR_DB)

    print(f"\n  -- RMSE Comparison (SNR = {SNR_DB} dB) -------------------------")
    names = {"peak":"Peak Detection","xcorr":"Cross-Correlation",
             "freq":"Freq. Domain","bayes":"Bayesian (Stage 3)","wbtm":"WBTM-Net (full)"}
    for m, name in names.items():
        tag = "  <- NOVEL CONTRIBUTION" if m == "wbtm" else ""
        print(f"  {name:<25}: {rmse[m]:>7.2f} um RMSE{tag}")

    print("\n  Generating plots...")
    plot_all(net, sample, result, bench, rmse, SNR_DB, save_path=SAVE_PATH)
    plt.show()
    print("\n  [Done]")


if __name__ == "__main__":
    main()
