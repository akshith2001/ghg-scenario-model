"""Synthetic feeder energy data: simulated radial network, AC load flow, generative arms,
metrics and a bridge to the GHG accounting model.

The network is simulated; Studies 2 and 3 use real daily building loads. NumPy/SciPy only for the generators and load flow; scikit-learn
(optional extra ``synthetic``) is needed only for the train-synthetic-test-real metric.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from .model import Activity

# ----------------------------------------------------------------------------------------
# Network and load flow
# ----------------------------------------------------------------------------------------
N_BUS = 9                      # slack (0) + 8 load buses
N_LOAD = N_BUS - 1
T = 24                         # hourly resolution, one sample = one day
PARENT = np.array([-1, 0, 1, 2, 3, 1, 5, 6, 7])
R = np.array([0, 0.012, 0.020, 0.020, 0.025, 0.020, 0.025, 0.030, 0.035])   # pu
X = np.array([0, 0.008, 0.013, 0.013, 0.016, 0.013, 0.016, 0.019, 0.022])   # pu
IMPEDANCE_SCALE = 0.43   # calibrated on simulator data only, before any model was trained
Z = (R + 1j * X) * IMPEDANCE_SCALE
TAN_PHI = 0.3                  # reactive/active ratio for consumption
V_MIN = 0.95


def _sweep(p: np.ndarray, tan_phi: float, iters: int):
    """Forward-backward sweep. Returns (complex bus voltages, branch currents, converged)."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != N_LOAD or not len(p) or not np.isfinite(p).all():
        raise ValueError("loads must be a nonempty finite (samples, 8) array")
    if not isinstance(iters, (int, np.integer)) or iters < 1 or not np.isfinite(tan_phi):
        raise ValueError("positive integer iterations and finite reactive ratio required")
    s = p + 1j * tan_phi * np.maximum(p, 0.0)   # PV assumed unity power factor
    m = p.shape[0]
    v = np.ones((m, N_BUS), dtype=complex)
    branch = np.zeros((m, N_BUS), dtype=complex)
    converged = False
    for _ in range(iters):
        branch = np.zeros((m, N_BUS), dtype=complex)
        branch[:, 1:] = np.conj(s / v[:, 1:])     # load currents at buses 1..8
        for i in range(N_BUS - 1, 0, -1):         # leaves -> root
            if PARENT[i] >= 1:
                branch[:, PARENT[i]] += branch[:, i]
        v_new = np.ones((m, N_BUS), dtype=complex)
        for i in range(1, N_BUS):                 # root -> leaves
            v_new[:, i] = v_new[:, PARENT[i]] - Z[i] * branch[:, i]
        delta = np.max(np.abs(v_new - v))
        v = v_new
        if delta < 1e-10:
            converged = True
            break
    return v, branch, converged


def load_flow(p: np.ndarray, tan_phi: float = TAN_PHI, iters: int = 40) -> np.ndarray:
    """AC power flow on the radial feeder by forward-backward sweep.

    p: net active injections in pu, shape (M, 8), positive = consumption.
    Returns bus voltage magnitudes, shape (M, 9), including the slack at 1.0 pu.
    Raises if the sweep has not converged (for example for implausibly large injections).
    """
    v, _, converged = _sweep(p, tan_phi, iters)
    if not converged:
        raise RuntimeError("load_flow did not converge; refusing unreliable voltages")
    return np.abs(v)


def slack_power(p: np.ndarray, tan_phi: float = TAN_PHI) -> np.ndarray:
    """Active power drawn at the slack bus, including line losses (used in tests)."""
    _, branch, converged = _sweep(p, tan_phi, 60)
    if not converged:
        raise RuntimeError("slack_power did not converge")
    into_feeder = sum(branch[:, c] for c in range(1, N_BUS) if PARENT[c] == 0)
    return np.real(np.conj(into_feeder))


def day_voltages(P: np.ndarray, t: int = T) -> np.ndarray:
    """P: (N, 8*t) injections bus-major -> voltages (N, 8*t) for buses 1..8 (bus-major).

    t is the number of time steps per sample (24 for the simulated study, 1 for daily means).
    """
    P = _validate_days(P, t)
    n = P.shape[0]
    p3 = P.reshape(n, N_LOAD, t)
    pp = p3.transpose(0, 2, 1).reshape(n * t, N_LOAD)
    v = load_flow(pp)[:, 1:]
    return v.reshape(n, t, N_LOAD).transpose(0, 2, 1).reshape(n, N_LOAD * t)


def violation_days(V: np.ndarray, vmin: float = V_MIN) -> float:
    """Share of days with any bus below vmin."""
    return float(np.mean(V.min(axis=1) < vmin))


# ----------------------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------------------
KIND = np.array(["res", "res", "com", "res", "com", "res", "com", "res"])   # buses 1..8
PEAK = np.array([0.30, 0.38, 0.42, 0.34, 0.46, 0.30, 0.40, 0.32])          # pu
PV_BUSES = {2: 0.35, 5: 0.30, 7: 0.28}                                       # index (0-based) -> pu cap


def _shape(kind, h):
    if kind == "res":
        return 0.38 + 0.55 * np.exp(-((h - 19.0) / 2.6) ** 2) + 0.22 * np.exp(-((h - 7.5) / 1.6) ** 2)
    return 0.22 + 0.80 * np.exp(-((h - 13.0) / 4.2) ** 2)


def simulate(n_days: int, seed: int) -> np.ndarray:
    """Simulate n_days of net injections. Returns array (n_days, 8*T), bus-major, in pu."""
    rng = np.random.default_rng(seed)
    h = np.arange(T)
    base = np.stack([_shape(k, h) for k in KIND])           # (8, T)
    out = np.zeros((n_days, N_LOAD, T))
    for d in range(n_days):
        season = np.cos(2 * np.pi * d / 365.0)              # +1 winter, -1 summer
        temp = rng.normal(0, 1)
        cloud = rng.beta(5, 2)
        weekend = (d % 7) >= 5
        day_factor = 1.0 + 0.10 * season + 0.07 * temp
        for b in range(N_LOAD):
            kind = KIND[b]
            wk = (1.08 if kind == "res" else 0.62) if weekend else 1.0
            amp = PEAK[b] * day_factor * wk * np.exp(rng.normal(0, 0.08))
            noise = np.zeros(T)
            e = rng.normal(0, 0.05, T)
            for t in range(T):
                noise[t] = (0.7 * noise[t - 1] if t else 0.0) + e[t]
            load = amp * base[b] * (1 + noise)
            if b in PV_BUSES:
                sun = np.maximum(0.0, np.sin(np.pi * (h - 6.0) / 13.0)) ** 1.2
                sun = sun * (0.85 - 0.25 * season) * cloud * (1 + rng.normal(0, 0.05, T))
                load = load - PV_BUSES[b] * np.maximum(sun, 0.0)
            out[d, b] = load
    return out.reshape(n_days, N_LOAD * T)


def split(seed: int, n: int = 1500):
    data = simulate(n, seed)
    return data[:800], data[800:900], data[900:]


# ----------------------------------------------------------------------------------------
# Standardisation
# ----------------------------------------------------------------------------------------
class Scaler:
    def fit(self, a):
        self.mu = a.mean(0)
        self.sd = a.std(0) + 1e-8
        return self

    def tf(self, a):
        return (a - self.mu) / self.sd

    def inv(self, a):
        return a * self.sd + self.mu


# ----------------------------------------------------------------------------------------
# NumPy VAE with optional physics penalty
# ----------------------------------------------------------------------------------------
class VAE:
    """One-hidden-layer tanh VAE trained with Adam; manual backprop.

    If `n_p` and `surrogate` are given, a penalty lam * 0.5 * ||v - (p A + b)||^2 is added,
    where the first n_p outputs are injections and the rest are voltages (standardised).
    """

    def __init__(self, d_in, hidden=128, latent=8, seed=0, lam=0.0, n_p=None, surrogate=None):
        rng = np.random.default_rng(seed)
        self.d, self.h, self.k, self.lam, self.n_p = d_in, hidden, latent, lam, n_p
        self.surrogate = surrogate
        s = lambda a, b: rng.normal(0, np.sqrt(1.0 / a), (a, b))
        self.p = {
            "W1": s(d_in, hidden), "b1": np.zeros(hidden),
            "Wm": s(hidden, latent) * 0.1, "bm": np.zeros(latent),
            "Wl": s(hidden, latent) * 0.1, "bl": np.zeros(latent),
            "W2": s(latent, hidden), "b2": np.zeros(hidden),
            "W3": s(hidden, d_in), "b3": np.zeros(d_in),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0
        self.rng = rng

    # -- forward / loss / gradients ------------------------------------------------------
    def _forward(self, x, eps, p=None):
        p = p or self.p
        h = np.tanh(x @ p["W1"] + p["b1"])
        mu = h @ p["Wm"] + p["bm"]
        lv = np.clip(h @ p["Wl"] + p["bl"], -8, 4)
        z = mu + np.exp(0.5 * lv) * eps
        g = np.tanh(z @ p["W2"] + p["b2"])
        xr = g @ p["W3"] + p["b3"]
        return h, mu, lv, z, g, xr

    def _penalty_terms(self, xr):
        if not self.lam or self.surrogate is None:
            return 0.0, None
        A, b = self.surrogate
        n_p = self.n_p
        po, vo = xr[:, :n_p], xr[:, n_p:]
        r = vo - po @ A - b
        pen = self.lam * 0.5 * np.sum(r ** 2) / xr.shape[0]
        return pen, r

    def loss_and_grads(self, x, eps):
        B = x.shape[0]
        h, mu, lv, z, g, xr = self._forward(x, eps)
        rec = 0.5 * np.sum((xr - x) ** 2) / B
        kl = -0.5 * np.sum(1 + lv - mu ** 2 - np.exp(lv)) / B
        pen, r = self._penalty_terms(xr)
        dxr = (xr - x) / B
        if r is not None:
            A, _ = self.surrogate
            n_p = self.n_p
            dxr[:, n_p:] += self.lam * r / B
            dxr[:, :n_p] += -self.lam * (r @ A.T) / B
        gr = {}
        gr["W3"] = g.T @ dxr
        gr["b3"] = dxr.sum(0)
        dg = dxr @ self.p["W3"].T
        dza = dg * (1 - g ** 2)
        gr["W2"] = z.T @ dza
        gr["b2"] = dza.sum(0)
        dz = dza @ self.p["W2"].T
        dmu = dz + mu / B
        dlv = dz * eps * 0.5 * np.exp(0.5 * lv) + 0.5 * (np.exp(lv) - 1) / B
        dlv = dlv * ((h @ self.p["Wl"] + self.p["bl"] > -8) & (h @ self.p["Wl"] + self.p["bl"] < 4))
        gr["Wm"] = h.T @ dmu
        gr["bm"] = dmu.sum(0)
        gr["Wl"] = h.T @ dlv
        gr["bl"] = dlv.sum(0)
        dh = dmu @ self.p["Wm"].T + dlv @ self.p["Wl"].T
        dha = dh * (1 - h ** 2)
        gr["W1"] = x.T @ dha
        gr["b1"] = dha.sum(0)
        return rec + kl + pen, gr

    def _adam(self, gr, lr=1e-3, b1=0.9, b2=0.999):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * gr[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * gr[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def val_loss(self, x):
        eps = np.zeros((x.shape[0], self.k))
        h, mu, lv, z, g, xr = self._forward(x, eps)
        rec = 0.5 * np.sum((xr - x) ** 2) / x.shape[0]
        kl = -0.5 * np.sum(1 + lv - mu ** 2 - np.exp(lv)) / x.shape[0]
        pen, _ = self._penalty_terms(xr)
        return rec + kl + pen

    def fit(self, xtr, xva, epochs=300, batch=64, lr=1e-3):
        best, best_p = np.inf, None
        n = xtr.shape[0]
        for ep in range(epochs):
            idx = self.rng.permutation(n)
            for s in range(0, n, batch):
                xb = xtr[idx[s:s + batch]]
                eps = self.rng.normal(size=(xb.shape[0], self.k))
                _, gr = self.loss_and_grads(xb, eps)
                self._adam(gr, lr=lr)
            if ep % 10 == 9 or ep == epochs - 1:
                vl = self.val_loss(xva)
                if vl < best:
                    best, best_p = vl, {k: v.copy() for k, v in self.p.items()}
        self.p = best_p
        # per-feature residual std on validation (decoder noise used when sampling)
        eps = np.zeros((xva.shape[0], self.k))
        xr = self._forward(xva, eps)[-1]
        self.res_sd = np.std(xva - xr, axis=0)
        return best

    def sample(self, n, rng):
        z = rng.normal(size=(n, self.k))
        g = np.tanh(z @ self.p["W2"] + self.p["b2"])
        xr = g @ self.p["W3"] + self.p["b3"]
        return xr + rng.normal(size=xr.shape) * self.res_sd


# ----------------------------------------------------------------------------------------
# Arms
# ----------------------------------------------------------------------------------------
def fit_surrogate(p_std, v_std, ridge=1e-2):
    """Linear map v_std ~ p_std A + b fitted on training data (used only in the soft penalty)."""
    Xb = np.hstack([p_std, np.ones((p_std.shape[0], 1))])
    W = np.linalg.solve(Xb.T @ Xb + ridge * np.eye(Xb.shape[1]), Xb.T @ v_std)
    return W[:-1], W[-1]


def run_arm(name: str, train: np.ndarray, val: np.ndarray, n_syn: int, seed: int, epochs: int = 300,
            t: int = T, hidden: int = 128, latent: int = 8, k: int = 24):
    """Return synthetic (injections P_syn, voltages V_reported or None) in physical units.

    Defaults reproduce Study 1 (24 hourly steps). Study 2 uses t=1, hidden=32, latent=3, k=8.
    """
    rng = np.random.default_rng(10_000 + seed)
    sp = Scaler().fit(train)
    ptr, pva = sp.tf(train), sp.tf(val)
    if name == "GAUSS":
        u, s, vt = np.linalg.svd(ptr - ptr.mean(0), full_matrices=False)
        comp = vt[:k]
        sd = s[:k] / np.sqrt(ptr.shape[0] - 1)
        coef = rng.normal(size=(n_syn, k)) * sd
        res = ptr - (ptr - ptr.mean(0)) @ comp.T @ comp - ptr.mean(0)
        syn = coef @ comp + ptr.mean(0) + rng.normal(size=(n_syn, ptr.shape[1])) * res.std(0)
        return sp.inv(syn), None
    if name == "GMM":
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError as exc:  # pragma: no cover
            raise ImportError("GMM needs scikit-learn: pip install -e .[synthetic]") from exc
        best = None
        for n_comp in range(1, 5):                 # components chosen by BIC on the train window
            g = GaussianMixture(n_comp, covariance_type="full", reg_covar=1e-4, n_init=3,
                                random_state=seed).fit(ptr)
            if best is None or g.bic(ptr) < best.bic(ptr):
                best = g
        return sp.inv(best.sample(n_syn)[0]), None
    vtr, vva = day_voltages(train, t), day_voltages(val, t)
    if name == "VAE-HARD":
        vae = VAE(ptr.shape[1], hidden=hidden, latent=latent, seed=seed)
        vae.fit(ptr, pva, epochs=epochs)
        return sp.inv(vae.sample(n_syn, rng)), None
    sv = Scaler().fit(vtr)
    xtr = np.hstack([ptr, sv.tf(vtr)])
    xva = np.hstack([pva, sv.tf(vva)])
    if name == "VAE-FREE":
        vae = VAE(xtr.shape[1], hidden=hidden, latent=latent, seed=seed)
    elif name == "VAE-SOFT":
        A, b = fit_surrogate(ptr, sv.tf(vtr))
        vae = VAE(xtr.shape[1], hidden=hidden, latent=latent, seed=seed, lam=1.0, n_p=ptr.shape[1],
                  surrogate=(A, b))
    else:
        raise ValueError(name)
    vae.fit(xtr, xva, epochs=epochs)
    syn = vae.sample(n_syn, rng)
    n_p = ptr.shape[1]
    return sp.inv(syn[:, :n_p]), sv.inv(syn[:, n_p:])


# ----------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------
def pf_residual(P_syn, V_reported, t=T):
    """RMSE (pu) between reported voltages and the exact load-flow voltages of the injections."""
    solved = day_voltages(P_syn, t)  # validates loads and requires solver convergence
    if V_reported is None:
        # Voltages are defined by this solver: consistency by construction, not a learned result.
        return 0.0
    V_reported = np.asarray(V_reported, dtype=float)
    if V_reported.shape != solved.shape or not np.isfinite(V_reported).all():
        raise ValueError("reported voltages must match the finite solved array")
    return float(np.sqrt(np.mean((V_reported - solved) ** 2)))


def physics_diagnostics(P, t=T):
    """Independently reconstruct nodal power from complex voltages and impedances.

    This checks numerical balance, not agreement with a real feeder. Voltage
    limits are separate from solvability; no thermal/current limits are modelled.
    """
    P = _validate_days(P, t)
    pp = P.reshape(-1, N_LOAD, t).transpose(0, 2, 1).reshape(-1, N_LOAD)
    v, _, converged = _sweep(pp, TAN_PHI, 60)
    if not converged:
        raise RuntimeError("Cannot report physical diagnostics for an unconverged solve")
    currents = (v[:, PARENT[1:]] - v[:, 1:]) / Z[1:]
    net = currents.copy()
    for child in range(1, N_BUS):
        parent = PARENT[child]
        if parent > 0:
            net[:, parent - 1] -= currents[:, child - 1]
    reconstructed = v[:, 1:] * np.conj(net)
    requested = pp + 1j * TAN_PHI * np.maximum(pp, 0.0)
    volts = np.abs(v[:, 1:]).reshape(-1, t, N_LOAD)
    return dict(nodal_balance_max_pu=float(np.max(np.abs(reconstructed - requested))),
                voltage_limit_violation_share=float(np.mean(np.any((volts < V_MIN) | (volts > 1.05), axis=(1, 2)))))


def violation_gap(P_syn, P_real, t=T):
    return abs(violation_days(day_voltages(P_syn, t)) - violation_days(day_voltages(P_real, t)))


def fidelity(P_syn, P_real, ref=None):
    """Mean per-feature 1-D Wasserstein distance and correlation-matrix error against P_real.

    By default features are standardised by P_real's own mean and sd (Studies 1 and 2). If `ref`
    is given (Study 3: the real train window) it supplies the mean and sd instead, and the
    correlation error uses only features that vary in P_real, so a flat meter cannot blow up
    the metric.
    """
    base = P_real if ref is None else ref
    mu = base.mean(0)
    sd = base.std(0) + 1e-8
    a, b = (P_syn - mu) / sd, (P_real - mu) / sd
    w1 = float(np.mean([wasserstein_distance(a[:, j], b[:, j]) for j in range(a.shape[1])]))
    if ref is not None:
        keep = P_real.std(0) > 1e-9
        a, b = a[:, keep], b[:, keep]
    ca, cb = np.corrcoef(a, rowvar=False), np.corrcoef(b, rowvar=False)
    ca, cb = np.nan_to_num(ca), np.nan_to_num(cb)
    return w1, float(np.linalg.norm(ca - cb, "fro") / ca.shape[0])


def _tstr_xy(P):
    P = _validate_days(P, T)
    n = P.shape[0]
    p3 = P.reshape(n, N_LOAD, T)
    xs = p3[:, :, :12].reshape(n, -1)
    y = p3[:, :, 17:22].sum(axis=(1, 2))
    return xs, y


def tstr_mae(P_train, P_test, seed=0):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover
        raise ImportError("TSTR needs scikit-learn: pip install -e .[synthetic]") from exc
    x, y = _tstr_xy(P_train)
    xt, yt = _tstr_xy(P_test)
    m = HistGradientBoostingRegressor(max_iter=200, random_state=seed).fit(x, y)
    return float(np.mean(np.abs(m.predict(xt) - yt)))


def memorisation(P_syn, P_train):
    sd = P_train.std(0) + 1e-8
    mu = P_train.mean(0)
    a, b = (P_syn - mu) / sd, (P_train - mu) / sd

    def nn(x, y, same=False):
        d = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
        if same:
            np.fill_diagonal(d, np.inf)
        return np.sqrt(d.min(1))

    thr = np.percentile(nn(b, b, same=True), 5)
    return float(np.mean(nn(a, b) < thr))


# ----------------------------------------------------------------------------------------
# Bridge to the GHG accounting model
# ----------------------------------------------------------------------------------------
KWH_PER_PU_HOUR = 1000.0       # 1 pu = 1 MW on the simulated feeder, so 1 pu for 1 h = 1000 kWh
DEFAULT_FACTOR = 0.18          # kg CO2e/kWh; the repo's illustrative sample electricity factor
# ASSUMED time-of-day shape (not data): dirtier evening peak, cleaner morning; mean over hours = 1.
TOU_SHAPE = 1.0 + 0.35 * np.cos(2 * np.pi * (np.arange(T) - 19.0) / 24.0)


def _validate_days(P, t):
    if not isinstance(t, (int, np.integer)) or t < 1:
        raise ValueError("t must be a positive integer")
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != N_LOAD * t or not len(P) or not np.isfinite(P).all():
        raise ValueError("expected nonempty finite (days, 8*t) loads")
    return P



def feeder_import_kwh(P_days, t=T):
    """Grid-side import including line losses, (days,t), over equal steps spanning 24h.

    Daily-resolution inputs are mean power; exports receive no credit.
    """
    P_days = _validate_days(P_days, t)
    n = len(P_days)
    loads = P_days.reshape(n, N_LOAD, t).transpose(0, 2, 1).reshape(n*t, N_LOAD)
    net = slack_power(loads).reshape(n, t)
    return np.maximum(net, 0.0) * KWH_PER_PU_HOUR * (24.0 / t)


def daily_emissions_kg(P_days, factor=DEFAULT_FACTOR, time_of_use=False, t=T, shape=None):
    """Daily emissions from grid imports; hourly factors require hourly input."""
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("factor must be finite and nonnegative")
    if (time_of_use or shape is not None) and t != T:
        raise ValueError("time-of-use factors require 24 hourly values")
    if shape is not None:
        shape = np.asarray(shape, dtype=float)
        if shape.shape != (T,) or not np.isfinite(shape).all() or np.any(shape < 0) or shape.mean() <= 0:
            raise ValueError("shape must contain 24 finite nonnegative values with positive mean")
        f = factor * shape / shape.mean()
    else:
        f = factor * (TOU_SHAPE if time_of_use else 1.0)
    return (feeder_import_kwh(P_days, t) * f).sum(axis=1)


def electricity_activity(P_days, activity_id="SG001", factor=DEFAULT_FACTOR, days_per_year=365, t=T):
    """Turn a set of (real or synthetic) feeder days into a GHG-model `Activity` record.

    activity_value is the mean daily import scaled to a year. activity_uncertainty_pct is the
    relative standard error of that mean over the supplied days (sampling uncertainty only).
    """
    daily = feeder_import_kwh(P_days, t).sum(axis=1)
    mean = float(daily.mean())
    rel_se = float(daily.std(ddof=1) / np.sqrt(len(daily)) / mean) if len(daily) > 1 and mean > 0 else 0.0
    return Activity(
        activity_id=activity_id,
        sector="electricity",
        activity_name="simulated feeder electricity import",
        activity_value=mean * days_per_year,
        activity_unit="kWh",
        emission_factor=factor,
        factor_unit="kg CO2e/kWh",
        factor_year="illustrative",
        factor_source="Synthetic demonstration factor - replace before real use",
        activity_uncertainty_pct=100.0 * rel_se,
        factor_uncertainty_pct=5.0,
    )


def emissions_error(P_syn, P_real, time_of_use=False, factor=DEFAULT_FACTOR, t=T, shape=None):
    """Relative error of mean daily emissions, synthetic vs real (exploratory metric)."""
    a = daily_emissions_kg(P_syn, factor, time_of_use, t, shape).mean()
    b = daily_emissions_kg(P_real, factor, time_of_use, t, shape).mean()
    if b == 0:
        raise ValueError("relative emissions error undefined for zero reference emissions")
    return float(abs(a - b) / b)


# ----------------------------------------------------------------------------------------
# Experiment and command line
# ----------------------------------------------------------------------------------------
ARMS = ["GAUSS", "VAE-FREE", "VAE-SOFT", "VAE-HARD"]
CONTRASTS = [("VAE-SOFT", "VAE-FREE"), ("VAE-HARD", "VAE-FREE"), ("VAE-HARD", "GAUSS"), ("VAE-FREE", "GAUSS")]
METRICS = ["pf_residual", "violation_gap", "w1", "corr_err", "tstr_ratio", "memorisation",
           "emis_err_flat", "emis_err_tou", "nodal_balance_max_pu", "voltage_limit_violation_share"]          # last two are exploratory


def run_experiment(seeds=(0, 1, 2, 3, 4), epochs=300, verbose=True):
    rows = []
    for seed in seeds:
        tr, va, te = split(seed)
        trtr = tstr_mae(tr, te, seed)
        for arm in ARMS:
            P, V = run_arm(arm, tr, va, n_syn=800, seed=seed, epochs=epochs)
            w1, corr = fidelity(P, te)
            row = dict(seed=seed, arm=arm, pf_residual=pf_residual(P, V),
                       violation_gap=violation_gap(P, te), w1=w1, corr_err=corr,
                       tstr_ratio=tstr_mae(P, te, seed) / trtr, memorisation=memorisation(P, tr),
                       emis_err_flat=emissions_error(P, te), emis_err_tou=emissions_error(P, te, True))
            row.update(physics_diagnostics(P, 1 if P.shape[1] == N_LOAD else T))
            rows.append(row)
            if verbose:
                print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return rows


# ----------------------------------------------------------------------------------------
# Study 2: real building loads on the simulated feeder (daily-mean operating points)
# ----------------------------------------------------------------------------------------
REAL_METRICS = ["pf_residual", "violation_gap", "w1", "corr_err", "tstr_ratio", "memorisation",
                "emis_err_flat", "nodal_balance_max_pu", "voltage_limit_violation_share"]                     # last one is exploratory
REAL_TARGET_VENUE = "Eagle_food_Kay"                # TSTR target, fixed in the protocol
REAL_ARM_KWARGS = dict(t=1, hidden=32, latent=3, k=8)


def load_real_days(path):
    """Read the BDG2 hospitality daily file.

    Returns (venue_ids sorted alphabetically, dates, kWh array (days, venues)) restricted to the
    days on which every venue has a reading.
    """
    table = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table.setdefault(row["utc_date"], {})[row["venue_id"]] = float(row["electricity_kwh"])
    venues = sorted({v for day in table.values() for v in day})
    dates = sorted(d for d, day in table.items() if len(day) == len(venues))
    kwh = np.array([[table[d][v] for v in venues] for d in dates])
    return venues, dates, kwh


def calibrate_scale(kwh_train, target=0.10):
    """Scale s (similar premises per bus) so that `target` of train days violate 0.95 pu.

    Mean kW = kWh / 24; per unit = kW / 1000 * s. Bracketing then bisection on train data only.
    A scale at which the load flow does not converge counts as a violation, so the search never
    relies on unreliable voltages.
    """
    base = kwh_train / 24.0 / 1000.0

    def share(scale):
        v, _, converged = _sweep(base * scale, TAN_PHI, 40)
        if not converged:
            return 1.0
        return float(np.mean(np.abs(v).min(axis=1) < V_MIN))

    lo, hi = 0.0, 0.25
    while share(hi) < target:
        lo, hi = hi, hi * 1.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if share(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def real_split(path):
    """Chronological split per the protocol: 282 train + 50 validation days of 2016, test = 2017."""
    venues, dates, kwh = load_real_days(path)
    is16 = np.array([d.startswith("2016") for d in dates])
    k16, k17 = kwh[is16], kwh[~is16]
    train, val = k16[:282], k16[282:332]
    scale = calibrate_scale(train)
    to_pu = lambda a: a / 24.0 / 1000.0 * scale
    return venues, to_pu(train), to_pu(val), to_pu(k17), scale


def tstr_real_mae(P_train, P_test, target, seed=0):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover
        raise ImportError("TSTR needs scikit-learn: pip install -e .[synthetic]") from exc
    others = [j for j in range(P_train.shape[1]) if j != target]
    m = HistGradientBoostingRegressor(max_iter=200, random_state=seed).fit(P_train[:, others], P_train[:, target])
    return float(np.mean(np.abs(m.predict(P_test[:, others]) - P_test[:, target])))


def run_real_experiment(path, seeds=(0, 1, 2, 3, 4), epochs=300, verbose=True):
    venues, tr, va, te, scale = real_split(path)
    target = venues.index(REAL_TARGET_VENUE)
    trtr = tstr_real_mae(tr, te, target)
    # Reference: the real 2016 training days scored against 2017. It shows how much of each metric
    # reflects observed train-to-test shift; it is not a lower bound.
    w1_ref, corr_ref = fidelity(tr, te)
    reference = dict(violation_gap=violation_gap(tr, te, 1), w1=w1_ref, corr_err=corr_ref,
                     emis_err_flat=emissions_error(tr, te, t=1),
                     violation_share_train=violation_days(day_voltages(tr, 1)),
                     violation_share_test=violation_days(day_voltages(te, 1)))
    rows = []
    for seed in seeds:
        for arm in ARMS:
            P, V = run_arm(arm, tr, va, n_syn=len(tr), seed=seed, epochs=epochs, **REAL_ARM_KWARGS)
            w1, corr = fidelity(P, te)
            err = emissions_error(P, te, t=1)
            row = dict(seed=seed, arm=arm, pf_residual=pf_residual(P, V, 1),
                       violation_gap=violation_gap(P, te, 1), w1=w1, corr_err=corr,
                       tstr_ratio=tstr_real_mae(P, te, target, seed) / trtr,
                       memorisation=memorisation(P, tr), emis_err_flat=float(err))
            row.update(physics_diagnostics(P, 1 if P.shape[1] == N_LOAD else T))
            rows.append(row)
            if verbose:
                print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return rows, scale, reference


# ----------------------------------------------------------------------------------------
# Study 3: rolling-origin evaluation on the real building loads
# ----------------------------------------------------------------------------------------
ROLL_ARMS = ARMS + ["GMM"]
ROLL_ORIGINS = (0, 60, 120, 180, 240)
ROLL_CONTRASTS = CONTRASTS + [("GMM", "GAUSS")]
ROLL_METRICS = REAL_METRICS + ["w1_excess", "corr_err_excess"]   # excess = value minus distribution-shift reference


def rolling_window(kwh, origin):
    """Train = days[o:o+240], validation = days[o+240:o+280], test = days[o+280:o+400]; scale from train."""
    train, val, test = kwh[origin:origin + 240], kwh[origin + 240:origin + 280], kwh[origin + 280:origin + 400]
    scale = calibrate_scale(train)
    to_pu = lambda a: a / 24.0 / 1000.0 * scale
    return to_pu(train), to_pu(val), to_pu(test), scale


def run_rolling_experiment(path, epochs=300, verbose=True):
    venues, _, kwh = load_real_days(path)
    target = venues.index(REAL_TARGET_VENUE)
    rows, floors = [], []
    for idx, origin in enumerate(ROLL_ORIGINS):
        tr, va, te, scale = rolling_window(kwh, origin)
        w1_floor, corr_floor = fidelity(tr, te, ref=tr)
        floors.append(dict(origin=origin, scale=scale, w1_floor=w1_floor, corr_err_floor=corr_floor,
                           violation_gap_floor=violation_gap(tr, te, 1),
                           violation_share_train=violation_days(day_voltages(tr, 1)),
                           violation_share_test=violation_days(day_voltages(te, 1))))
        trtr = tstr_real_mae(tr, te, target)
        for arm in ROLL_ARMS:
            P, V = run_arm(arm, tr, va, n_syn=len(tr), seed=idx, epochs=epochs, **REAL_ARM_KWARGS)
            w1, corr = fidelity(P, te, ref=tr)
            err = emissions_error(P, te, t=1)
            row = dict(seed=idx, arm=arm, pf_residual=pf_residual(P, V, 1),
                       violation_gap=violation_gap(P, te, 1), w1=w1, corr_err=corr,
                       tstr_ratio=tstr_real_mae(P, te, target, idx) / trtr,
                       memorisation=memorisation(P, tr), emis_err_flat=float(err),
                       w1_excess=w1 - w1_floor, corr_err_excess=corr - corr_floor)
            row.update(physics_diagnostics(P, 1 if P.shape[1] == N_LOAD else T))
            rows.append(row)
            if verbose:
                print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return rows, floors


def summarise(rows, metrics=None, arms=None, contrasts=None, intervals=True) -> str:
    """Mean and sd per arm, plus paired 95% t-intervals for the main contrasts.

    Standard deviations and intervals need at least two seeds; with one seed they are omitted.
    """
    from scipy import stats
    metrics = metrics or METRICS
    arms = arms or ARMS
    contrasts = contrasts or CONTRASTS
    seeds = sorted({r["seed"] for r in rows})
    multi = len(seeds) > 1
    head = ["arm"]
    for m in metrics:
        head += [f"{m}_mean"] + ([f"{m}_sd"] if multi else [])
    lines = [",".join(head)]
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm]
        cells = [arm]
        for m in metrics:
            v = np.array([r[m] for r in sel])
            cells.append(f"{v.mean():.5f}")
            if multi:
                cells.append(f"{v.std(ddof=1):.5f}")
        lines.append(",".join(cells))
    lines.append("")
    if "w1_excess" in metrics:
        intervals = False
    if not multi or not intervals:
        lines.append("Only descriptive summaries: overlapping rolling windows are dependent; no inferential intervals." if not intervals else "Paired intervals need at least two seeds.")
        return "\n".join(lines)
    lines.append("Paired differences (A minus B), 95% t-interval, number of seeds with A<B")
    by = {(r["arm"], r["seed"]): r for r in rows}
    for a, b in contrasts:
        for m in ["pf_residual", "w1", "corr_err", "violation_gap", "tstr_ratio"]:
            x = np.array([by[(a, s)][m] - by[(b, s)][m] for s in seeds])
            mu = x.mean()
            h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            lines.append(f"{a} - {b} {m}: {mu:+.4f} [{mu - h:+.4f},{mu + h:+.4f}] A<B in {int((x < 0).sum())}/{len(x)}")
    return "\n".join(lines)


def _write(rows, metrics, output: Path, reference=None, arms=None, contrasts=None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "raw.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=["seed", "arm"] + metrics)
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows, metrics, arms, contrasts, intervals=(arms != ROLL_ARMS))
    if reference:
        summary += "\n\nReference (real 2016 train days scored against 2017; distribution-shift reference): " + ", ".join(
            f"{k}={v:.5f}" for k, v in reference.items())
    (output / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic feeder energy data study.")
    parser.add_argument("--study", choices=["simulated", "real", "rolling"], default="simulated",
                        help="simulated: Study 1; real: Study 2 (real BDG2 loads, one split); "
                             "rolling: Study 3 (real loads, five rolling origins, adds a GMM arm)")
    parser.add_argument("--real-data", type=Path, default=Path("data/real/bdg2_hospitality_daily.csv"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.study == "rolling":
        rows, floors = run_rolling_experiment(args.real_data, args.epochs)
        out = args.output or Path("output/synthetic_grid_rolling")
        _write(rows, ROLL_METRICS, out, arms=ROLL_ARMS, contrasts=ROLL_CONTRASTS)
        with (out / "floors.csv").open("w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=list(floors[0]))
            w.writeheader()
            w.writerows(floors)
    elif args.study == "real":
        rows, scale, reference = run_real_experiment(args.real_data, args.seeds, args.epochs)
        print(f"calibrated scale (similar premises per bus): {scale:.3f}")
        _write(rows, REAL_METRICS, args.output or Path("output/synthetic_grid_real"), reference)
    else:
        rows = run_experiment(args.seeds, args.epochs)
        _write(rows, METRICS, args.output or Path("output/synthetic_grid"))


if __name__ == "__main__":
    main()
