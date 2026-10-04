from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data import TICKER
from src.hp_filter import LAMBDA
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch,
    lilliefors
)
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="statsmodels"
)


def run_hp_statistical_analysis(
    data: pd.DataFrame = None,
    alpha: float = 0.05,
    save_plots: bool = True,
    show_plots: bool = True
) -> dict:
    """
    HP Filtresi uygulandıktan sonra elde edilen Döngü (HP_Cycle) bileşeninin
    Brownian Motion / İstatistiksel varsayımlar açısından kapsamlı analizini yapar.
    """

    # ============================================================
    # 1. PATH SETUP
    # ============================================================
    BASE_DIR = Path(__file__).resolve().parent.parent
    FIGURES_DIR = BASE_DIR / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 2. DÜZELTİLMİŞ BIST30 DOSYASINI YÜKLE VE ANALİZ ET
    # ============================================================
    corrected_data_path = BASE_DIR / "xu030_is_hp_corrected.csv"
    if not corrected_data_path.exists():
        raise FileNotFoundError(
            f"Düzeltilmiş veri dosyası bulunamadı: {corrected_data_path}. "
            "Önce hp_filter.run_hp_filter(...) çalıştırıp dosyayı oluşturun."
        )

    hp_result = pd.read_csv(
        corrected_data_path,
        index_col="Date",
        parse_dates=["Date"],
    ).sort_index()
    if "Close" not in hp_result.columns:
        raise ValueError("Düzeltilmiş CSV içinde 'Close' sütunu bulunamadı.")

    # CSV'deki Close sütunu, seçili 20 tarihi HP Trend ile değiştirilmiş
    # nihai tam dönem serisidir; diğer günler orijinal kapanış olarak korunur.
    corrected_price = pd.to_numeric(hp_result["Close"], errors="coerce").dropna()
    if (corrected_price <= 0).any():
        raise ValueError("Log HP filtresi için düzeltilmiş fiyatlar pozitif olmalıdır.")

    # Varsayım testleri CSV'deki tam dönem verisinin HP_Cycle bileşeni üzerindedir.
    cycle_log, _ = hpfilter(np.log(corrected_price), lamb=LAMBDA)
    cycle = cycle_log.dropna()
    hp_result["Analysis_HP_Cycle"] = cycle.reindex(hp_result.index)
    n_obs = len(cycle)

    if n_obs < 50:
        raise ValueError("İstatistiksel analiz için en az 50 gözlem gereklidir.")

    # ============================================================
    # 3. BASIC INFORMATION
    # ============================================================
    print("\n" + "=" * 90)
    print("HP CYCLE — STATISTICAL & BROWNIAN MOTION DIAGNOSTICS")
    print("=" * 90)
    print("\nDATA INFORMATION")
    print("-" * 90)
    print(f"Input file                : {corrected_data_path.name}")
    print(f"Observations              : {n_obs}")
    print(f"Start Date                : {cycle.index.min().date()}")
    print(f"End Date                  : {cycle.index.max().date()}")

    # ============================================================
    # 4. DESCRIPTIVE STATISTICS
    # ============================================================
    mean_cyc = cycle.mean()
    median_cyc = cycle.median()
    std_cyc = cycle.std(ddof=1)
    variance_cyc = cycle.var(ddof=1)
    skewness_cyc = stats.skew(cycle, bias=False)
    kurtosis_excess = stats.kurtosis(cycle, fisher=True, bias=False)
    kurtosis_pearson = stats.kurtosis(cycle, fisher=False, bias=False)
    min_cyc = cycle.min()
    max_cyc = cycle.max()

    print("\n" + "=" * 90)
    print("1. DESCRIPTIVE STATISTICS — HP CYCLE")
    print("=" * 90)
    print(f"Cycle Mean                : {mean_cyc:+.8f} (Sıfıra yakın olmalı)")
    print(f"Cycle Median              : {median_cyc:+.8f}")
    print(f"Cycle Std. Deviation      : {std_cyc:.8f}")
    print(f"Cycle Variance            : {variance_cyc:.8f}")
    print(f"Minimum Cycle Value       : {min_cyc:+.4f}")
    print(f"Maximum Cycle Value       : {max_cyc:+.4f}")
    print(f"Skewness                  : {skewness_cyc:+.6f}")
    print(f"Pearson Kurtosis          : {kurtosis_pearson:.6f}")
    print(f"Excess Kurtosis           : {kurtosis_excess:+.6f}")

    # ============================================================
    # 5. NORMALITY / GAUSSIANITY
    # ============================================================
    print("\n" + "=" * 90)
    print("2. NORMALITY / GAUSSIANITY DIAGNOSTICS (HP CYCLE)")
    print("=" * 90)

    jb_stat, jb_pvalue = stats.jarque_bera(cycle)
    print("\n[2.1] JARQUE-BERA TEST")
    print("-" * 90)
    print(f"Statistic                 : {jb_stat:.6f}")
    print(f"p-value                   : {jb_pvalue:.6e}")
    jb_decision = "REJECT" if jb_pvalue < alpha else "FAIL TO REJECT"
    jb_status = "NON-NORMAL" if jb_pvalue < alpha else "NORMALITY NOT REJECTED"
    print(f"Decision                  : {jb_decision}")
    print(f"Interpretation            : {jb_status}")

    lil_stat, lil_pvalue = lilliefors(cycle, dist="norm")
    print("\n[2.2] LILLIEFORS TEST")
    print("-" * 90)
    print(f"Statistic                 : {lil_stat:.6f}")
    print(f"p-value                   : {lil_pvalue:.6e}")
    lil_decision = "REJECT" if lil_pvalue < alpha else "FAIL TO REJECT"
    print(f"Decision                  : {lil_decision}")

    ad_result = stats.anderson(cycle, dist="norm")
    ad_5_index = np.where(ad_result.significance_level == 5.0)[0]
    ad_critical_5 = ad_result.critical_values[ad_5_index[0]] if len(ad_5_index) > 0 else np.nan

    print("\n[2.3] ANDERSON-DARLING TEST")
    print("-" * 90)
    print(f"Statistic                 : {ad_result.statistic:.6f}")
    print(f"5% Critical Value         : {ad_critical_5:.6f}")
    ad_decision = "REJECT" if (not np.isnan(ad_critical_5) and ad_result.statistic > ad_critical_5) else "FAIL TO REJECT"
    print(f"Decision                  : {ad_decision}")

    # ============================================================
    # 6. INDEPENDENT INCREMENTS / SERIAL DEPENDENCE
    # ============================================================
    print("\n" + "=" * 90)
    print("3. SERIAL DEPENDENCE (HP CYCLE)")
    print("=" * 90)

    lags = [lag for lag in [1, 5, 10, 20, 40] if lag < n_obs // 2]
    lb_cycle = acorr_ljungbox(cycle, lags=lags, return_df=True)

    print("\n[3.1] LJUNG-BOX TEST — HP CYCLE")
    print("-" * 90)
    print(f"{'Lag':<10}{'LB Statistic':<20}{'p-value':<20}{'Decision':<20}")
    print("-" * 70)

    for lag in lags:
        statistic = lb_cycle.loc[lag, "lb_stat"]
        pvalue = lb_cycle.loc[lag, "lb_pvalue"]
        decision = "REJECT" if pvalue < alpha else "FAIL TO REJECT"
        print(f"{lag:<10}{statistic:<20.6f}{pvalue:<20.6e}{decision:<20}")

    # ============================================================
    # 7. CONSTANT VARIANCE / VOLATILITY CLUSTERING
    # ============================================================
    squared_cycle = cycle ** 2
    lb_squared_cyc = acorr_ljungbox(squared_cycle, lags=lags, return_df=True)

    print("\n" + "=" * 90)
    print("4. CONSTANT VARIANCE / VOLATILITY (HP CYCLE)")
    print("=" * 90)

    print("\n[4.1] LJUNG-BOX — SQUARED HP CYCLE")
    print("-" * 90)
    for lag in lags:
        statistic = lb_squared_cyc.loc[lag, "lb_stat"]
        pvalue = lb_squared_cyc.loc[lag, "lb_pvalue"]
        decision = "VOLATILITY DEPENDENCE" if pvalue < alpha else "NO SIGNIFICANT DEPENDENCE"
        print(f"Lag {lag:<6} | Stat: {statistic:<15.4f} | p-value: {pvalue:<20.6e} | {decision}")

    arch_lags = min(10, max(1, n_obs // 5))
    arch_stat, arch_pvalue, arch_f_stat, arch_f_pvalue = het_arch(cycle, nlags=arch_lags)
    print(f"\n[4.2] ENGLE ARCH-LM TEST on HP Cycle:")
    print(f"LM p-value                : {arch_pvalue:.6e}")
    arch_decision = "REJECT" if arch_pvalue < alpha else "FAIL TO REJECT"
    print(f"Decision                  : {arch_decision}")

    # ============================================================
    # 8. STATIONARITY (CRITICAL FOR HP CYCLE)
    # ============================================================
    print("\n" + "=" * 90)
    print("5. STATIONARITY OF HP CYCLE")
    print("=" * 90)

    adf_result = adfuller(cycle, autolag="AIC")
    adf_stat, adf_pvalue, adf_lag, _, adf_critical, _ = adf_result

    print("\n[5.1] AUGMENTED DICKEY-FULLER")
    print("-" * 90)
    print(f"ADF Statistic             : {adf_stat:.6f}")
    print(f"p-value                   : {adf_pvalue:.6e}")
    print(f"5% Critical Value         : {adf_critical['5%']:.6f}")
    adf_decision = "REJECT UNIT ROOT" if adf_pvalue < alpha else "FAIL TO REJECT UNIT ROOT"
    adf_status = "STATIONARY" if adf_pvalue < alpha else "NON-STATIONARY"
    print(f"Decision                  : {adf_decision}")
    print(f"Interpretation            : {adf_status} (HP Cycle teorik olarak durağan olmalıdır)")

    kpss_result = kpss(cycle, regression="c", nlags="auto")
    kpss_stat, kpss_pvalue, kpss_lags, kpss_critical = kpss_result

    print("\n[5.2] KPSS")
    print("-" * 90)
    print(f"KPSS Statistic            : {kpss_stat:.6f}")
    print(f"p-value                   : {kpss_pvalue:.6e}")
    print(f"5% Critical Value         : {kpss_critical['5%']:.6f}")
    kpss_decision = "REJECT STATIONARITY" if kpss_pvalue < alpha else "FAIL TO REJECT STATIONARITY"
    print(f"Decision                  : {kpss_decision}")

    # ============================================================
    # 9. PLOTS
    # ============================================================
    asset_name = TICKER.replace("=", "").replace(".", "_").replace("-", "_").lower()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    fig.suptitle(f"{TICKER} — HP Cycle Statistical Diagnostics", fontsize=12, fontweight="bold")

    # Histogram & Norm
    axes[0, 0].hist(cycle, bins=50, density=True, alpha=0.65, edgecolor="black", label="HP Cycle")
    x = np.linspace(cycle.min(), cycle.max(), 300)
    axes[0, 0].plot(x, stats.norm.pdf(x, loc=mean_cyc, scale=std_cyc), linewidth=2, label="Normal Dist")
    axes[0, 0].set_title("HP Cycle Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Q-Q Plot
    stats.probplot(cycle, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("HP Cycle Q-Q Plot")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # ACF of Cycle
    plot_acf(cycle, lags=min(40, n_obs // 2 - 1), alpha=0.05, ax=axes[1, 0], title="HP Cycle Autocorrelation")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Cycle Time Series
    axes[1, 1].plot(cycle.index, cycle, linewidth=0.8, color="purple")
    axes[1, 1].axhline(0, linestyle="--", color="black")
    axes[1, 1].set_title("HP Cycle Time Series (Mean-Reverting)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    if save_plots:
        fig.savefig(FIGURES_DIR / f"{asset_name}_hp_cycle_diagnostics.png", dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    return {
        "hp_result": hp_result,
        "corrected_price": corrected_price,
        "cycle": cycle,
        "metrics": {
            "mean": mean_cyc,
            "std": std_cyc,
            "skewness": skewness_cyc,
            "excess_kurtosis": kurtosis_excess
        },
        "stationarity": {
            "adf_pvalue": adf_pvalue,
            "adf_status": adf_status
        }
    }