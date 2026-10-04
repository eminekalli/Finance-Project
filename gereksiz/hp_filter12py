"""Lokal HP düzeltmesi; result.attrs CSV'ye yazılmaz."""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from src.data import TICKER
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.filters.hp_filter import hpfilter

LAMBDA = 144000
ALPHA = 0.05
WINDOW = 3
EST_WINDOW = 30
# Olay açıklamalarını kullanıcı doldurmalı (TODO).
JUMP_DATES = {
    "2023-01-05": {"yon": "negatif", "olay": "TODO"},
    "2023-01-11": {"yon": "negatif", "olay": "TODO"},
    "2023-02-01": {"yon": "negatif", "olay": "TODO"},
    "2023-02-07": {"yon": "negatif", "olay": "TODO"},
    "2023-05-15": {"yon": "negatif", "olay": "TODO"},
    "2023-10-25": {"yon": "negatif", "olay": "TODO"},
    "2024-08-05": {"yon": "negatif", "olay": "TODO"},
    "2025-03-19": {"yon": "negatif", "olay": "TODO"},
    "2025-03-21": {"yon": "negatif", "olay": "TODO"},
    "2026-05-21": {"yon": "negatif", "olay": "TODO"},
    "2026-09-16": {"yon": "negatif", "olay": "TODO"},
    "2023-01-12": {"yon": "pozitif", "olay": "TODO"},
    "2023-02-03": {"yon": "pozitif", "olay": "TODO"},
    "2023-02-15": {"yon": "pozitif", "olay": "TODO"},
    "2023-05-11": {"yon": "pozitif", "olay": "TODO"},
    "2023-06-05": {"yon": "pozitif", "olay": "TODO"},
    "2024-04-05": {"yon": "pozitif", "olay": "TODO"},
    "2025-06-30": {"yon": "pozitif", "olay": "TODO"},
    "2025-09-15": {"yon": "pozitif", "olay": "TODO"},
    "2026-04-08": {"yon": "pozitif", "olay": "TODO"},
}


def _date_metadata(jump_dates):
    if isinstance(jump_dates, dict):
        return {pd.Timestamp(k).normalize(): v for k, v in jump_dates.items()}
    return {pd.Timestamp(k).normalize(): {} for k in jump_dates}


def normality_tests(r: pd.Series, alpha: float = ALPHA) -> dict:
    """Özet Jarque-Bera normallik istatistiklerini döndürür."""
    r = pd.Series(r).dropna()
    if len(r) < 3:
        return {"jb_stat": np.nan, "jb_p": np.nan, "skew": np.nan,
                "ex_kurt": np.nan, "all_pass": False}
    jb_stat, jb_p = stats.jarque_bera(r)
    return {"jb_stat": float(jb_stat), "jb_p": float(jb_p),
            "skew": float(stats.skew(r, bias=False)),
            "ex_kurt": float(stats.kurtosis(r, fisher=True, bias=False)),
            "all_pass": bool(jb_p > alpha)}


def _prepare(raw_price):
    p = pd.Series(raw_price, copy=True).astype(float)
    p.index = pd.to_datetime(p.index).normalize()
    p = p[~p.index.duplicated(keep="first")].sort_index().dropna()
    if (p <= 0).any():
        raise ValueError("Log dönüşümü için fiyatlar pozitif olmalıdır.")
    return p


def _blocks(positions, window, merge_close_dates, n):
    positions = sorted(set(positions))
    if not positions:
        return []
    groups = [[positions[0]]]
    for pos in positions[1:]:
        if merge_close_dates and pos - groups[-1][-1] <= 2 * window:
            groups[-1].append(pos)
        else:
            groups.append([pos])
    return [(max(0, g[0] - window), min(n - 1, g[-1] + window)) for g in groups]


def _correction(log_price, dates, lamb, window, est_window, merge_close_dates=True,
                snap_to_next_trading_day=False):
    idx = log_price.index
    positions, mapped = [], {}
    for date, meta in _date_metadata(dates).items():
        pos = idx.get_indexer([date])[0]
        if pos < 0 and snap_to_next_trading_day:
            pos = int(idx.searchsorted(date))
            if pos >= len(idx):
                pos = -1
        if pos < 0:
            warnings.warn(f"Sıçrama tarihi indekste bulunamadı: {date.date()}", UserWarning)
            continue
        positions.append(pos)
        mapped[pos] = (idx[pos], meta)
    blocks = _blocks(positions, window, merge_close_dates, len(idx))
    n = len(idx)
    out = log_price.copy()
    trend_out = pd.Series(np.nan, index=idx)
    cycle_out = pd.Series(np.nan, index=idx)
    weight_all = np.zeros(n)
    proposed = log_price.to_numpy(copy=True)
    # HP daima ham log seriye uygulanır; blokların tamı eğitim penceresine dahil edilir.
    for lo, hi in blocks:
        a, b = max(0, lo - est_window), min(n - 1, hi + est_window)
        sub = log_price.iloc[a:b + 1]
        if len(sub) < 3:
            continue
        cycle, trend = hpfilter(sub, lamb=lamb)
        trend_out.iloc[lo:hi + 1] = trend.iloc[lo - a:hi - a + 1]
        cycle_out.iloc[lo:hi + 1] = cycle.iloc[lo - a:hi - a + 1]
        size = hi - lo + 1
        # Blok uçlarında sıfır ağırlık, ortada en çok bireysel düzeltme.
        w = np.ones(size) if size == 1 else np.sin(np.linspace(0, np.pi, size)) ** 2
        w[np.isclose(w, 0.0, atol=1e-12)] = 0.0
        existing = weight_all[lo:hi + 1]
        take = w > existing
        local = proposed[lo:hi + 1]
        local[take] = (log_price.iloc[lo:hi + 1].to_numpy()[take] * (1 - w[take])
                       + trend.iloc[lo - a:hi - a + 1].to_numpy()[take] * w[take])
        proposed[lo:hi + 1] = local
        weight_all[lo:hi + 1] = np.maximum(existing, w)
    out[:] = proposed
    changed = np.flatnonzero(weight_all > 0)
    return out, changed, trend_out, cycle_out, mapped, blocks


def find_minimal_hp_correction(raw_price: pd.Series, lamb: float = LAMBDA,
        alpha: float = ALPHA, jump_dates=JUMP_DATES, window: int = WINDOW,
        est_window: int = EST_WINDOW, merge_close_dates: bool = True,
        mode: str = "all", snap_to_next_trading_day: bool = False):
    """Apply deterministic local HP corrections and return series plus diagnostics."""
    if window < 0 or est_window < 1:
        raise ValueError("window >= 0 ve est_window >= 1 olmalıdır.")
    if mode not in {"all", "sequential", "greedy"}:
        raise ValueError("mode all, sequential veya greedy olmalıdır.")
    price = _prepare(raw_price)
    logp = np.log(price)
    dates = _date_metadata(jump_dates)
    active_dates = dict(dates)
    effective, rejected = [], []

    def apply(ds):
        return _correction(logp, ds, lamb, window, est_window, merge_close_dates,
                           snap_to_next_trading_day)

    corrected, replaced, tr, cy, mapped, blocks = apply(active_dates)
    history = [{"k": 0, "date": pd.NaT, "status": "baseline",
                **normality_tests(logp.diff().dropna(), alpha)}]
    if mode == "all":
        candidates = sorted(mapped.items())
        processed = {}
        for pos, (date, meta) in candidates:
            processed[date] = meta
            step = apply(processed)[0]
            history.append({"k": len(history), "date": date, "status": "applied",
                            **normality_tests(step.diff().dropna(), alpha)})
            effective.append(date)
    elif mode == "sequential":
        corrected = logp.copy()
        active_dates = {}
        for pos, (date, meta) in sorted(mapped.items()):
            active_dates[date] = meta
            corrected, replaced, tr, cy, _, _ = apply(active_dates)
            test = normality_tests(corrected.diff().dropna(), alpha)
            history.append({"k": len(history), "date": date, "status": "applied", **test})
            effective.append(date)
            if test["all_pass"]:
                break
    else:
        remaining = dict(mapped)
        corrected, replaced = logp.copy(), np.array([], dtype=int)
        current = normality_tests(corrected.diff().dropna(), alpha)
        while remaining and not current["all_pass"]:
            choices = []
            for pos, (candidate_date, meta) in remaining.items():
                trial_dates = {d: dates[d] for d in effective + [candidate_date]}
                trial = apply(trial_dates)
                test = normality_tests(trial[0].diff().dropna(), alpha)
                choices.append((test["jb_stat"], candidate_date, pos, trial, test))
            choices.sort(key=lambda z: (np.inf if pd.isna(z[0]) else z[0], z[1]))
            _, chosen, chosen_pos, trial, test = choices[0]
            if not np.isfinite(test["jb_stat"]) or test["jb_stat"] >= current["jb_stat"]:
                break
            corrected, replaced, tr, cy, _, _ = trial
            effective.append(chosen)
            remaining.pop(chosen_pos)
            history.append({"k": len(history), "date": chosen, "status": "effective", **test})
            current = test
        rejected = [d for d in dates if d not in effective]
        history.extend({"k": len(history), "date": d, "status": "unnecessary",
                        **current} for d in rejected)
    if mode != "greedy":
        rejected = [d for d in dates if d not in effective]
    # Produce metadata matching selected final state for greedy/sequential.
    if mode != "all":
        corrected, replaced, tr, cy, mapped, blocks = apply({d: dates[d] for d in effective})
    replaced_idx = price.index[replaced]
    hist = pd.DataFrame(history)
    return corrected, list(replaced_idx), hist, tr, cy, bool(normality_tests(corrected.diff().dropna(), alpha)["all_pass"])


def run_hp_filter(data, lamb=LAMBDA, alpha=ALPHA, jump_dates=JUMP_DATES,
        window=WINDOW, est_window=EST_WINDOW, mode="all", output_dir=None,
        merge_close_dates=True, snap_to_next_trading_day=False):
    """Run HP correction and save outputs; attrs such as history are not in CSV."""
    if data is None or "Close" not in data:
        raise ValueError("Market data veya 'Close' sütunu bulunamadı.")
    df = data.copy()
    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index().loc[lambda x: ~x.index.duplicated(keep="first")]
    price = pd.to_numeric(df["Close"], errors="coerce").dropna()
    corrected, replaced, history, trend, cycle, converged = find_minimal_hp_correction(
        price, lamb, alpha, jump_dates, window, est_window, merge_close_dates,
        mode, snap_to_next_trading_day)
    corrected_price = np.exp(corrected)
    result = df.loc[price.index].copy()
    result["Close_Original"] = price
    result["Close"] = corrected_price
    result["Is_Corrected"] = result.index.isin(replaced)
    replaced_idx = result.index[result["Is_Corrected"]]
    result["Correction_Order"] = np.nan
    result.loc[replaced_idx, "Correction_Order"] = pd.Series(range(1, len(replaced_idx)+1), index=replaced_idx)
    result["Log_Return_Original"] = np.log(price / price.shift())
    result["Log_Return"] = corrected.diff()
    result["HP_Trend"] = trend
    result["HP_Cycle"] = cycle
    result["HP_Trend_Price"] = np.exp(trend)
    result.attrs.update(history=history, converged=converged, mode=mode,
                         jump_dates=_date_metadata(jump_dates))
    outdir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent.parent
    outdir.mkdir(parents=True, exist_ok=True)
    result.to_csv(outdir / "xu030_is_hp_corrected.csv", index_label="Date")
    history.to_csv(outdir / "xu030_hp_correction_history.csv", index=False)
    print(f"\n{TICKER} — LOKAL HP DÜZELTMESİ | Lambda: {lamb} | Tahmin: {est_window} | Değişim: ±{window}")
    date_statuses = history.get("status", pd.Series(dtype=str))
    applied_dates = int(date_statuses.isin(["applied", "effective"]).sum())
    print(f"Uygulanan sıçrama tarihi: {applied_dates} | Düzeltilen gün: {len(replaced)} | Normallik: {'EVET' if converged else 'HAYIR'}")
    return result


def sensitivity_analysis(raw_price, windows=(1, 2, 3, 5), est_windows=(15, 30, 60),
                         lambdas=(1600, 14400, 144000), output_dir=None):
    """Summarize JB and intervention sensitivity; writes a CSV summary."""
    price = _prepare(raw_price)
    rows = []
    for w in windows:
        for ew in est_windows:
            for lam in lambdas:
                corr, replaced, *_ = find_minimal_hp_correction(
                    price, lamb=lam, window=w, est_window=ew, mode="all")
                ret = corr.diff().dropna()
                nt = normality_tests(ret)
                rawvar = float(np.var(np.log(price).diff().dropna(), ddof=1))
                newvar = float(np.var(ret, ddof=1))
                rows.append({"window": w, "est_window": ew, "lambda": lam,
                    "JB stat": nt["jb_stat"], "JB p": nt["jb_p"], "skew": nt["skew"],
                    "excess kurtosis": nt["ex_kurt"], "changed_day_ratio": len(replaced)/len(price),
                    "removed_variance_ratio": 1-newvar/rawvar if rawvar else np.nan})
    table = pd.DataFrame(rows)
    outdir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent.parent
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "xu030_hp_sensitivity.csv", index=False)
    print(table.to_string(index=False))
    return table


def correction_cost_report(result):
    """Print and return intervention costs; assert aggregate log return is preserved."""
    raw = result["Log_Return_Original"].dropna()
    corrected = result["Log_Return"].dropna()
    assert np.isclose(raw.sum(), corrected.sum(), rtol=1e-9, atol=1e-9), "Toplam log getiri korunmadı."
    changed = result["Is_Corrected"].fillna(False)
    rows = []
    for date, meta in result.attrs.get("jump_dates", _date_metadata(JUMP_DATES)).items():
        if date not in result.index:
            continue
        i = result.index.get_loc(date)
        before = result["Log_Return_Original"].iloc[i]
        after = result["Log_Return"].iloc[i]
        rows.append({"date": date, "label": meta.get("yon", ""),
                     "return_before": before, "return_after": after, "change": after-before})
    a, b = np.log(result["Close_Original"]).diff().dropna(), np.log(result["Close"]).diff().dropna()
    report = {"changed_days": int(changed.sum()), "changed_ratio": float(changed.mean()),
              "removed_variance_ratio": float(1-b.var()/a.var()) if a.var() else np.nan,
              "per_date": pd.DataFrame(rows)}
    print(f"Değiştirilen gün: {report['changed_days']} ({report['changed_ratio']:.2%})")
    print(f"Kaldırılan varyans oranı: {report['removed_variance_ratio']:.6f}")
    print(report["per_date"].to_string(index=False))
    return report


def detect_jumps_by_threshold(log_returns, k=3):
    """Return dates whose absolute log return exceeds k times sample sigma."""
    r = pd.Series(log_returns).dropna()
    sigma = r.std()
    return r.index[(r.abs() > k * sigma)]


def compare_manual_vs_objective_dates(raw_price, k=3, **kwargs):
    """Apply identical HP settings to manual and threshold-selected dates."""
    price = _prepare(raw_price)
    auto_dates = detect_jumps_by_threshold(np.log(price).diff(), k)
    manual = find_minimal_hp_correction(price, **kwargs)[0]
    objective = find_minimal_hp_correction(price, jump_dates=list(auto_dates), **kwargs)[0]
    return pd.DataFrame({"manual": manual, "objective": objective}, index=price.index)


def plot_history(history):
    """Plot JB p-value against correction order k."""
    import matplotlib.pyplot as plt
    ax = history.plot(x="k", y="jb_p", marker="o", legend=False)
    ax.axhline(ALPHA, color="red", linestyle="--", label=f"alpha={ALPHA}")
    ax.set(xlabel="k", ylabel="JB p-değeri", title="Düzeltme geçmişi")
    ax.legend()
    return ax


def run_hp_statistical_analysis(data: pd.DataFrame = None, alpha: float = ALPHA,
        save_plots: bool = True, show_plots: bool = True, verbose=False) -> dict:
    """Run src.stat diagnostics on corrected closes."""
    base = Path(__file__).resolve().parent.parent
    if data is None:
        path = base / "xu030_is_hp_corrected.csv"
        if not path.exists():
            raise FileNotFoundError(f"Düzeltilmiş veri dosyası bulunamadı: {path}")
        data = pd.read_csv(path, index_col="Date", parse_dates=["Date"]).sort_index()
    from src.stat import run_statistical_analysis
    return run_statistical_analysis(data=data, alpha=alpha, save_plots=save_plots,
                                    show_plots=show_plots)


def compare_raw_vs_corrected(result: pd.DataFrame, alpha: float = ALPHA,
                             lags=(1, 5, 10, 20)) -> pd.DataFrame:
    """Compare JB, ARCH-LM and Ljung-Box diagnostics."""
    rows = {}
    for name, r in {"Ham": result["Log_Return_Original"].dropna(),
                    "Düzeltilmiş": result["Log_Return"].dropna()}.items():
        n = normality_tests(r, alpha)
        lb = acorr_ljungbox(r, lags=list(lags), return_df=True)
        lb2 = acorr_ljungbox(r ** 2, lags=list(lags), return_df=True)
        row = {"N": len(r), "Skew": n["skew"], "Excess Kurt": n["ex_kurt"],
               "JB p": n["jb_p"], "ARCH-LM p": het_arch(r, nlags=10)[1]}
        for lag in lags:
            row[f"LB p (lag {lag})"] = lb.loc[lag, "lb_pvalue"]
            row[f"LB² p (lag {lag})"] = lb2.loc[lag, "lb_pvalue"]
        rows[name] = row
    table = pd.DataFrame(rows)
    print("\nHAM vs DÜZELTİLMİŞ GETİRİLER — NORMALLİK & BAĞIMLILIK\n", table.to_string())
    return table
