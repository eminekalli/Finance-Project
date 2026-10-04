"""HP düzeltme tanı grafiklerini dışa aktarır."""
from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf


def create_hp_report_plots(result: pd.DataFrame, sensitivity: Optional[pd.DataFrame] = None,
                           output_dir: Union[str, Path] = "figures/hp_report",
                           jump_dates=None) -> list[Path]:
    """Save six core report figures and an optional sensitivity heatmap."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    idx = result.index
    raw_log = np.log(pd.to_numeric(result["Close_Original"], errors="coerce").dropna())
    corrected_log = np.log(pd.to_numeric(result["Close"], errors="coerce").reindex(raw_log.index))
    raw_r = result["Log_Return_Original"].reindex(raw_log.index).dropna()
    corrected_r = result["Log_Return"].reindex(raw_r.index).dropna()
    history = result.attrs.get("history")
    if history is None:
        raise ValueError("result.attrs['history'] bulunamadı; run_hp_filter çıktısını kullanın.")
    history = pd.DataFrame(history)

    # 1. Ham ve düzeltilmiş log fiyat serileri.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(raw_log.index, raw_log, label="Ham", alpha=.75, linewidth=1)
    ax.plot(corrected_log.index, corrected_log, label="Düzeltilmiş", linewidth=1.2)
    ax.set(title="Ham ve düzeltilmiş log kapanış fiyatı", xlabel="Tarih", ylabel="Log fiyat")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    saved.append(_save(fig, out / "01_raw_vs_corrected_log_price.png"))

    # 2. Her manuel sıçrama için ±10 işlem günlük yakın plan.
    dates = jump_dates if jump_dates is not None else result.attrs.get("jump_dates", {})
    if isinstance(dates, dict):
        dates = list(dates)
    dates = sorted(pd.Timestamp(d).normalize() for d in dates if pd.Timestamp(d).normalize() in idx)
    if dates:
        ncols = 4
        nrows = int(np.ceil(len(dates) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.1*nrows), squeeze=False)
        status = history.dropna(subset=["date"]).set_index(pd.to_datetime(history.dropna(subset=["date"])["date"]).dt.normalize())["status"].to_dict()
        for ax, date in zip(axes.flat, dates):
            pos = idx.get_loc(date)
            left, right = max(0, pos-10), min(len(idx), pos+11)
            section = idx[left:right]
            ax.plot(section, np.log(result.loc[section, "Close_Original"]), label="Ham", lw=1)
            ax.plot(section, np.log(result.loc[section, "Close"]), label="Düzeltilmiş", lw=1)
            ax.axvline(date, color="black", linestyle="--", lw=.8)
            label = status.get(date, "seçilmedi")
            ax.set_title(f"{date:%Y-%m-%d} ({label})", fontsize=9)
            ax.tick_params(axis="x", labelrotation=35, labelsize=7)
            ax.grid(alpha=.2)
        for ax in axes.flat[len(dates):]:
            ax.remove()
        axes.flat[0].legend(fontsize=8)
        fig.suptitle("Sıçrama tarihleri çevresinde log fiyat yakın planları", y=1.002)
        fig.tight_layout()
        saved.append(_save(fig, out / "02_jump_date_zoom_panels.png"))

    # 7. Ham/düzeltilmiş Q-Q grafikleri.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, series, title in zip(axes, (raw_r, corrected_r), ("Ham getiriler", "Düzeltilmiş getiriler")):
        stats.probplot(series, dist="norm", plot=ax)
        ax.set_title(f"Normal Q-Q: {title}"); ax.set_xlabel("Teorik normal kantiller")
        ax.set_ylabel("Getiri kantilleri"); ax.grid(alpha=.25)
    fig.tight_layout(); saved.append(_save(fig, out / "07_qq_raw_vs_corrected.png"))

    # 6. Getiri histogramları ve her seriye uyan normal yoğunluk.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, series, title, color in zip(axes, (raw_r, corrected_r), ("Ham", "Düzeltilmiş"), ("#5279a5", "#d17a45")):
        vals = series.to_numpy()
        ax.hist(vals, bins=45, density=True, alpha=.6, color=color, edgecolor="white", label="Getiri histogramı")
        x = np.linspace(vals.min(), vals.max(), 400)
        ax.plot(x, stats.norm.pdf(x, vals.mean(), vals.std(ddof=1)), color="black", lw=1.5, label="Uydurulmuş normal yoğunluk")
        ax.set(title=f"{title} getiriler", xlabel="Log getiri", ylabel="Yoğunluk")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); saved.append(_save(fig, out / "06_return_histogram_vs_normal.png"))

    # 10. Düzeltme turu k'ya göre JB p-değeri.
    hist = history.dropna(subset=["jb_p"]).copy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if not hist.empty:
        ax.plot(hist["k"], hist["jb_p"].clip(lower=1e-300), marker="o", lw=1.3)
        ax.set_yscale("log")
    ax.axhline(.05, color="red", linestyle="--", label="α = 0,05")
    ax.set(title="Düzeltme turuna göre Jarque–Bera p-değeri", xlabel="Düzeltme turu (k)", ylabel="JB p-değeri (log ölçek)")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    saved.append(_save(fig, out / "10_jb_p_by_k.png"))

    # 12. Kare getirilerin ACF karşılaştırması.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    max_lag = min(40, len(raw_r)-1, len(corrected_r)-1)
    lags = np.arange(max_lag+1)
    for series, label, color, offset in ((raw_r, "Ham", "#5279a5", -.12), (corrected_r, "Düzeltilmiş", "#d17a45", .12)):
        values = acf(series.to_numpy()**2, nlags=max_lag, fft=True)
        ax.vlines(lags+offset, 0, values, color=color, linewidth=1.2, label=label)
        ax.scatter(lags+offset, values, color=color, s=12)
    ax.axhline(1.96/np.sqrt(len(raw_r)), color="gray", linestyle="--", lw=.8)
    ax.axhline(-1.96/np.sqrt(len(raw_r)), color="gray", linestyle="--", lw=.8)
    ax.axhline(0, color="black", lw=.6)
    ax.set(title="Kare log getirilerin otokorelasyonu", xlabel="Gecikme", ylabel="ACF")
    ax.legend(); ax.grid(axis="y", alpha=.2); fig.tight_layout()
    saved.append(_save(fig, out / "12_squared_return_acf.png"))

    # 15. Duyarlılık ısı haritası: her lambda için JB p-değeri paneli.
    if sensitivity is not None and not sensitivity.empty:
        required = {"window", "est_window", "lambda", "JB p"}
        if required.issubset(sensitivity.columns):
            lambdas = sorted(sensitivity["lambda"].unique())
            fig, axes = plt.subplots(1, len(lambdas), figsize=(5*len(lambdas), 4.5), squeeze=False)
            for ax, lam in zip(axes.flat, lambdas):
                pivot = sensitivity[sensitivity["lambda"] == lam].pivot(index="est_window", columns="window", values="JB p").sort_index()
                pvals = pivot.to_numpy(dtype=float)
                image = ax.imshow(-np.log10(np.clip(pvals, 1e-300, 1)), aspect="auto", cmap="viridis")
                ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
                ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
                ax.set(xlabel="Değiştirme penceresi ± işlem günü", ylabel="Tahmin penceresi (işlem günü)", title=f"λ = {lam}")
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        ax.text(j, i, f"{pvals[i,j]:.3g}", ha="center", va="center", color="white" if -np.log10(max(pvals[i,j],1e-300)) > 1.5 else "black", fontsize=8)
            fig.subplots_adjust(left=.07, right=.89, bottom=.17, top=.82, wspace=.35)
            colorbar_ax = fig.add_axes([.915, .2, .018, .58])
            fig.colorbar(image, cax=colorbar_ax, label="−log₁₀(JB p-değeri)")
            fig.suptitle("HP parametre duyarlılığı: JB p-değeri", y=.97)
            saved.append(_save(fig, out / "15_sensitivity_heatmap_jb_p.png"))
    return saved


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafik kaydedildi: {path}")
    return path
