from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data import TICKER
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch,
)
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="statsmodels"
)


def run_statistical_analysis(
    data: pd.DataFrame = None,
    alpha: float = 0.05,
    save_plots: bool = True,
    show_plots: bool = True,
    verbose: bool = True  
) -> dict:
    """
    BIST 30 log-getiri serisinin Brownian Motion / GBM
    varsayımları açısından kapsamlı istatistiksel analizini yapar.
    """

    # ============================================================
    # 1. PATH SETUP
    # ============================================================
    BASE_DIR = Path(__file__).resolve().parent.parent
    FIGURES_DIR = BASE_DIR / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 2. DATA LOADING
    # ============================================================
    if data is None:
        raise ValueError(
            "DataFrame bulunamadı. "
            "main.py içinde data.py'den gelen veriyi "
            "run_statistical_analysis(data=...) şeklinde gönderin."
        )

    # ============================================================
    # 3. DATA PREPARATION
    # ============================================================
    if "Close" not in data.columns:
        raise ValueError("'Close' sütunu veri içerisinde bulunamadı.")

    df = data[["Close"]].copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()

    if "Log_Return" not in df.columns:
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["Log_Return"] = pd.to_numeric(df["Log_Return"], errors="coerce")
    df = df.dropna(subset=["Log_Return"])

    returns = df["Log_Return"]
    n_obs = len(returns)

    if n_obs < 100:
        raise ValueError("İstatistiksel analiz için gözlem sayısı çok düşük.")

    # ============================================================
    # 4. BASIC INFORMATION
    # ============================================================
    if verbose:
        print("\n" + "=" * 90)
        print("ASSET — BROWNIAN MOTION STATISTICAL DIAGNOSTICS")
        print("=" * 90)
        print("\nDATA INFORMATION")
        print("-" * 90)
        print(f"Observations              : {n_obs}")
        print(f"Start Date                : {df.index.min().date()}")
        print(f"End Date                  : {df.index.max().date()}")

    # ============================================================
    # 5. DESCRIPTIVE STATISTICS
    # ============================================================
    mean_ret = returns.mean()
    median_ret = returns.median()
    std_ret = returns.std(ddof=1)
    variance_ret = returns.var(ddof=1)
    skewness_ret = stats.skew(returns, bias=False)
    kurtosis_excess = stats.kurtosis(returns, fisher=True, bias=False)
    kurtosis_pearson = stats.kurtosis(returns, fisher=False, bias=False)
    min_ret = returns.min()
    max_ret = returns.max()
    annualized_mean = mean_ret * 252
    annualized_volatility = std_ret * np.sqrt(252)

    print("\n" + "=" * 90)
    print("1. DESCRIPTIVE STATISTICS — STYLIZED FACTS")
    print("=" * 90)
    print(f"Daily Mean Return         : {mean_ret:+.8f}")
    print(f"Daily Median Return       : {median_ret:+.8f}")
    print(f"Daily Std. Deviation      : {std_ret:.8f}")
    print(f"Daily Variance            : {variance_ret:.8f}")
    print(f"Minimum Daily Return      : {min_ret:+.4%}")
    print(f"Maximum Daily Return      : {max_ret:+.4%}")
    print(f"Skewness                  : {skewness_ret:+.6f}")
    print(f"Pearson Kurtosis          : {kurtosis_pearson:.6f}")
    print(f"Excess Kurtosis           : {kurtosis_excess:+.6f}")
    print(f"Annualized Mean Return    : {annualized_mean:+.4%}")
    print(f"Annualized Volatility     : {annualized_volatility:.4%}")

    # ============================================================
    # 6. NORMALITY / GAUSSIANITY
    # ============================================================
    print("\n" + "=" * 90)
    print("2. NORMALITY / GAUSSIANITY DIAGNOSTICS")
    print("=" * 90)

    # 2.1 Jarque-Bera
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    print("\n[2.1] JARQUE-BERA TEST")
    print("-" * 90)
    print(f"Statistic                 : {jb_stat:.6f}")
    print(f"p-value                   : {jb_pvalue:.6e}")
    print("H0                        : Returns are normally distributed.")
    jb_decision = "REJECT" if jb_pvalue < alpha else "FAIL TO REJECT"
    jb_status = "NON-NORMAL" if jb_pvalue < alpha else "NORMALITY NOT REJECTED"
    print(f"Decision                  : {jb_decision}")
    print(f"Interpretation            : {jb_status}")

    # ============================================================
    # 7. BROWNIAN VARIANCE SCALING
    # ============================================================
    print("\n" + "=" * 90)
    print("3. BROWNIAN MOTION — VARIANCE SCALING")
    print("=" * 90)

    horizons = [1, 5, 10, 20, 60]
    scaling_results = []
    daily_variance = returns.var(ddof=1)

    for h in horizons:
        horizon_returns = returns.rolling(window=h).sum().dropna()
        variance = horizon_returns.var(ddof=1)
        expected_variance = daily_variance * h
        scaling_ratio = variance / daily_variance
        theoretical_ratio = h
        scaling_error = (scaling_ratio / theoretical_ratio) - 1

        scaling_results.append({
            "Horizon": h,
            "Variance": variance,
            "Expected_BM_Variance": expected_variance,
            "Observed_Ratio": scaling_ratio,
            "Theoretical_Ratio": theoretical_ratio,
            "Scaling_Error": scaling_error
        })

    scaling_df = pd.DataFrame(scaling_results)

    print(
        f"{'Horizon':<12}"
        f"{'Variance':<18}"
        f"{'BM Expected':<18}"
        f"{'Observed Ratio':<18}"
        f"{'BM Ratio':<12}"
        f"{'Error':<12}"
    )
    print("-" * 90)

    for _, row in scaling_df.iterrows():
        print(
            f"{int(row['Horizon']):<12}"
            f"{row['Variance']:<18.8f}"
            f"{row['Expected_BM_Variance']:<18.8f}"
            f"{row['Observed_Ratio']:<18.4f}"
            f"{row['Theoretical_Ratio']:<12.4f}"
            f"{row['Scaling_Error']:<12.2%}"
        )

    # Variance scaling exponent
    log_horizons = np.log(scaling_df["Horizon"].values)
    log_variances = np.log(scaling_df["Variance"].values)
    scaling_slope, scaling_intercept = np.polyfit(log_horizons, log_variances, 1)

    log_variance_fitted = scaling_intercept + scaling_slope * log_horizons
    ss_res = np.sum((log_variances - log_variance_fitted) ** 2)
    ss_tot = np.sum((log_variances - log_variances.mean()) ** 2)
    scaling_r_squared = 1 - (ss_res / ss_tot)

    print("\nVARIANCE SCALING EXPONENT")
    print("-" * 90)
    print(f"Scaling Exponent (β)     : {scaling_slope:.6f}")
    print(f"Brownian Benchmark (β)   : 1.000000")
    print(f"R-squared                 : {scaling_r_squared:.6f}")

    scaling_deviation = abs(scaling_slope - 1)
    scaling_status = "COMPATIBLE WITH BROWNIAN SCALING" if scaling_deviation <= 0.10 else "INCONSISTENT WITH BROWNIAN SCALING"
    print(f"Interpretation            : {scaling_status}")

    # ============================================================
    # 8. INDEPENDENT INCREMENTS
    # ============================================================
    print("\n" + "=" * 90)
    print("4. INDEPENDENT INCREMENTS / SERIAL DEPENDENCE")
    print("=" * 90)

    lags = [1, 5, 10, 20, 40]
    lb_returns = acorr_ljungbox(returns, lags=lags, return_df=True)

    print("\n[4.1] LJUNG-BOX TEST — RETURNS")
    print("-" * 90)
    print(f"{'Lag':<10}{'LB Statistic':<20}{'p-value':<20}{'Decision':<20}")
    print("-" * 70)

    for lag in lags:
        statistic = lb_returns.loc[lag, "lb_stat"]
        pvalue = lb_returns.loc[lag, "lb_pvalue"]
        decision = "REJECT" if pvalue < alpha else "FAIL TO REJECT"
        print(f"{lag:<10}{statistic:<20.6f}{pvalue:<20.6e}{decision:<20}")

    # ============================================================
    # 9. CONSTANT VARIANCE / VOLATILITY CLUSTERING
    # ============================================================
    squared_returns = returns ** 2
    lb_squared = acorr_ljungbox(squared_returns, lags=lags, return_df=True)

    print("\n" + "=" * 90)
    print("5. CONSTANT VARIANCE / VOLATILITY CLUSTERING")
    print("=" * 90)

    print("\n[5.1] LJUNG-BOX — SQUARED RETURNS")
    print("-" * 90)
    print(f"{'Lag':<10}{'LB Statistic':<20}{'p-value':<20}{'Decision':<25}")
    print("-" * 75)

    for lag in lags:
        statistic = lb_squared.loc[lag, "lb_stat"]
        pvalue = lb_squared.loc[lag, "lb_pvalue"]
        decision = "VOLATILITY DEPENDENCE" if pvalue < alpha else "NO SIGNIFICANT DEPENDENCE"
        print(f"{lag:<10}{statistic:<20.6f}{pvalue:<20.6e}{decision:<25}")

    # ARCH-LM
    arch_lags = 10
    arch_stat, arch_pvalue, arch_f_stat, arch_f_pvalue = het_arch(returns, nlags=arch_lags)

    print("\n[5.2] ENGLE ARCH-LM TEST")
    print("-" * 90)
    print(f"LM Statistic              : {arch_stat:.6f}")
    print(f"LM p-value                : {arch_pvalue:.6e}")
    print(f"F Statistic               : {arch_f_stat:.6f}")
    print(f"F p-value                 : {arch_f_pvalue:.6e}")
    print("H0                        : No ARCH / conditional heteroskedasticity.")

    if arch_pvalue < alpha:
        arch_decision = "REJECT"
        arch_status = "ARCH EFFECT DETECTED — TIME-VARYING VOLATILITY"
    else:
        arch_decision = "FAIL TO REJECT"
        arch_status = "NO SIGNIFICANT ARCH EFFECT DETECTED"

    print(f"Decision                  : {arch_decision}")
    print(f"Interpretation            : {arch_status}")

    # ============================================================
    # 10. STATIONARITY
    # ============================================================
    print("\n" + "=" * 90)
    print("6. STATIONARITY OF LOG RETURNS")
    print("=" * 90)

    # ADF
    adf_result = adfuller(returns, autolag="AIC")
    adf_stat, adf_pvalue, adf_lag, _, adf_critical, _ = adf_result

    print("\n[6.1] AUGMENTED DICKEY-FULLER")
    print("-" * 90)
    print(f"ADF Statistic             : {adf_stat:.6f}")
    print(f"p-value                   : {adf_pvalue:.6e}")
    print(f"Used Lag                 : {adf_lag}")
    print(f"5% Critical Value        : {adf_critical['5%']:.6f}")
    adf_decision = "REJECT UNIT ROOT" if adf_pvalue < alpha else "FAIL TO REJECT UNIT ROOT"
    adf_status = "STATIONARY" if adf_pvalue < alpha else "NON-STATIONARY"
    print(f"Decision                  : {adf_decision}")
    print(f"Interpretation            : {adf_status}")

    # KPSS
    kpss_result = kpss(returns, regression="c", nlags="auto")
    kpss_stat, kpss_pvalue, kpss_lags, kpss_critical = kpss_result

    print("\n[6.2] KPSS")
    print("-" * 90)
    print(f"KPSS Statistic            : {kpss_stat:.6f}")
    print(f"p-value                   : {kpss_pvalue:.6e}")
    print(f"Used Lags                 : {kpss_lags}")
    print(f"5% Critical Value        : {kpss_critical['5%']:.6f}")
    kpss_decision = "REJECT STATIONARITY" if kpss_pvalue < alpha else "FAIL TO REJECT STATIONARITY"
    kpss_status = "NON-STATIONARY" if kpss_pvalue < alpha else "STATIONARY"
    print(f"Decision                  : {kpss_decision}")
    print(f"Interpretation            : {kpss_status}")

    # ============================================================
    # 11. TAIL ANALYSIS
    # ============================================================
    print("\n" + "=" * 90)
    print("7. TAIL / EXTREME EVENT ANALYSIS")
    print("=" * 90)

    empirical_quantiles = {
        "0.1%": returns.quantile(0.001),
        "1%": returns.quantile(0.01),
        "5%": returns.quantile(0.05),
        "95%": returns.quantile(0.95),
        "99%": returns.quantile(0.99),
        "99.9%": returns.quantile(0.999)
    }

    print("\nEMPIRICAL RETURN QUANTILES")
    print("-" * 90)
    for q, value in empirical_quantiles.items():
        print(f"{q:<10} : {value:+.6%}")

    thresholds = [0.01, 0.02, 0.03, 0.05]
    tail_probabilities = {}

    print("\nEMPIRICAL TAIL PROBABILITIES")
    print("-" * 90)
    for threshold in thresholds:
        left_probability = np.mean(returns < -threshold)
        right_probability = np.mean(returns > threshold)
        tail_probabilities[threshold] = {
            "left": left_probability,
            "right": right_probability
        }
        print(
            f"|r| > {threshold:.1%}"
            f"    Left: {left_probability:.4%}"
            f"    Right: {right_probability:.4%}"
        )

    # ============================================================
    # 12. BROWNIAN COMPATIBILITY SUMMARY
    # ============================================================
    normality_ok = jb_pvalue >= alpha

    independence_ok = not any(
        lb_returns.loc[lag, "lb_pvalue"] < alpha
        for lag in lags
    )

    constant_variance_ok = (
        arch_pvalue >= alpha
        and not any(
            lb_squared.loc[lag, "lb_pvalue"] < alpha
            for lag in lags
        )
    )

    stationarity_ok = (
        adf_pvalue < alpha
        and kpss_pvalue >= alpha
    )

    compatibility_summary = pd.DataFrame(
        {
            "Property": [
                "Gaussian / Normal Increments",
                "Independent Increments",
                "Constant Variance",
                "Stationary Returns"
            ],
            "Status": [
                "PASS" if normality_ok else "FAIL",
                "PASS" if independence_ok else "FAIL",
                "PASS" if constant_variance_ok else "FAIL",
                "PASS" if stationarity_ok else "FAIL"
            ],
            "Interpretation": [
                "Returns compatible with Gaussianity" if normality_ok else "Non-Gaussian / fat-tail or asymmetric behaviour",
                "No significant serial dependence detected" if independence_ok else "Serial dependence detected",
                "No significant volatility clustering detected" if constant_variance_ok else "Time-varying volatility / ARCH detected",
                "Returns appear stationary" if stationarity_ok else "Stationarity is not supported"
            ]
        }
    )

    print("\n" + "=" * 90)
    print("8. BROWNIAN MOTION COMPATIBILITY SUMMARY")
    print("=" * 90)
    print(compatibility_summary.to_string(index=False))

    # ============================================================
    # 13. PLOTS 
    # ============================================================

    asset_name = (
        TICKER
        .replace("=", "")
        .replace(".", "_")
        .replace("-", "_")
        .lower()
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(15, 12),
        layout="constrained"
    )

    fig.suptitle(
        f"{TICKER} — Brownian Motion Statistical Diagnostics",
        fontsize=10,
        fontweight="bold",
        y=0.99
    )

    # ------------------------------------------------------------
    # Plot 1 — Return Distribution
    # ------------------------------------------------------------
    ax1 = axes[0, 0]

    ax1.hist(
        returns,
        bins=100,
        density=True,
        alpha=0.65,
        edgecolor="black",
        linewidth=0.4,
        label="Empirical Returns"
    )

    x = np.linspace(
        returns.min(),
        returns.max(),
        500
    )

    ax1.plot(
        x,
        stats.norm.pdf(
            x,
            loc=mean_ret,
            scale=std_ret
        ),
        linewidth=2,
        label="Normal Distribution"
    )

    ax1.set_title(
        "Log Returns — Empirical vs Normal Distribution",
        fontsize=8,
        fontweight="bold",
        pad=10
    )

    ax1.set_xlabel("Log Return", labelpad=6)
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)


    # ------------------------------------------------------------
    # Plot 2 — Q-Q Plot
    # ------------------------------------------------------------
    ax2 = axes[0, 1]

    stats.probplot(
        returns,
        dist="norm",
        plot=ax2
    )

    ax2.set_title(
        "Log Returns — Normal Q-Q Plot",
        fontsize=8,
        fontweight="bold",
        pad=10
    )

    ax2.set_xlabel(
        "Theoretical Quantiles",
        labelpad=6
    )

    ax2.set_ylabel("Ordered Values")
    ax2.grid(True, linestyle="--", alpha=0.5)


    # ------------------------------------------------------------
    # Plot 3 — Return ACF
    # ------------------------------------------------------------
    ax3 = axes[1, 0]

    plot_acf(
        returns,
        lags=40,
        alpha=0.05,
        ax=ax3,
        title=None
    )

    ax3.set_title(
        "Log Returns — Autocorrelation",
        fontsize=8,
        fontweight="bold",
        pad=10
    )

    ax3.set_xlabel("Lag", labelpad=6)
    ax3.set_ylabel("Autocorrelation")
    ax3.grid(True, linestyle="--", alpha=0.5)


    # ------------------------------------------------------------
    # Plot 4 — Squared Return ACF
    # ------------------------------------------------------------
    ax4 = axes[1, 1]

    plot_acf(
        squared_returns,
        lags=40,
        alpha=0.05,
        ax=ax4,
        title=None
    )

    ax4.set_title(
        "Squared Log Returns — Volatility Clustering",
        fontsize=8,
        fontweight="bold",
        pad=10
    )

    ax4.set_xlabel("Lag", labelpad=6)

    ax4.set_ylabel(
        "Autocorrelation of Squared Returns"
    )

    ax4.grid(True, linestyle="--", alpha=0.5)


    # ------------------------------------------------------------
    # Plot 5 — Return Time Series
    # ------------------------------------------------------------
    ax5 = axes[2, 0]

    ax5.plot(
        df.index,
        returns,
        linewidth=0.7
    )

    ax5.axhline(
        0,
        linestyle="--",
        linewidth=1,
        color="black"
    )

    ax5.set_title(
        "Daily Log Returns",
        fontsize=8,
        fontweight="bold",
        pad=10
    )

    ax5.set_xlabel("Date", labelpad=6)
    ax5.set_ylabel("Log Return")
    ax5.grid(True, linestyle="--", alpha=0.5)


    # ------------------------------------------------------------
    # Plot 6 — Brownian Motion Summary
    # ------------------------------------------------------------
    ax6 = axes[2, 1]

    ax6.axis("off")

    summary_text = (
        f"{TICKER} Brownian Motion Summary\n\n"
        f"Gaussian Increments : "
        f"{'PASS' if normality_ok else 'FAIL'}\n"
        f"Independent Increments : "
        f"{'PASS' if independence_ok else 'FAIL'}\n"
        f"Constant Variance : "
        f"{'PASS' if constant_variance_ok else 'FAIL'}\n"
        f"Stationary Returns : "
        f"{'PASS' if stationarity_ok else 'FAIL'}\n\n"
        f"Variance Scaling β : "
        f"{scaling_slope:.4f}\n"
        f"R² : "
        f"{scaling_r_squared:.4f}"
    )

    ax6.text(
        0.5,
        0.5,
        summary_text,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=10,
        transform=ax6.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="#f8f9fa",
            edgecolor="#ced4da"
        )
    )


    # ------------------------------------------------------------
    # Save / Show
    # ------------------------------------------------------------
    if save_plots:
        fig.savefig(
            FIGURES_DIR / f"{asset_name}_diagnostics_panel.png",
            dpi=300,
            bbox_inches="tight"
        )

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # ============================================================
    # 14. FINAL RETURN OBJECT
    # ============================================================
    return {
        "returns": returns,
        "n_obs": n_obs,
        "metrics": {
            "mean": mean_ret,
            "median": median_ret,
            "std": std_ret,
            "variance": variance_ret,
            "skewness": skewness_ret,
            "kurtosis_excess": kurtosis_excess,
            "kurtosis_pearson": kurtosis_pearson,
            "min": min_ret,
            "max": max_ret,
            "annualized_mean": annualized_mean,
            "annualized_volatility": annualized_volatility
        },
        "normality": {
            "jb_stat": jb_stat,
            "jb_pvalue": jb_pvalue,
            "jb_decision": jb_decision
        },
        "independence": {
            "ljung_box_returns": lb_returns
        },
        "volatility": {
            "ljung_box_squared": lb_squared,
            "arch_stat": arch_stat,
            "arch_pvalue": arch_pvalue,
            "arch_f_stat": arch_f_stat,
            "arch_f_pvalue": arch_f_pvalue,
            "arch_decision": arch_decision
        },
        "stationarity": {
            "adf_stat": adf_stat,
            "adf_pvalue": adf_pvalue,
            "adf_lag": adf_lag,
            "adf_decision": adf_decision,
            "kpss_stat": kpss_stat,
            "kpss_pvalue": kpss_pvalue,
            "kpss_lags": kpss_lags,
            "kpss_decision": kpss_decision
        },
        "tails": {
            "empirical_quantiles": empirical_quantiles,
            "tail_probabilities": tail_probabilities
        },
        "variance_scaling": {
            "results": scaling_df,
            "scaling_exponent": scaling_slope,
            "scaling_intercept": scaling_intercept,
            "r_squared": scaling_r_squared,
            "status": scaling_status
        },
        "brownian_compatibility": compatibility_summary
    }


# =================================================================
# MAIN
# =================================================================
if __name__ == "__main__":
    results = run_statistical_analysis(
        save_plots=True,
        show_plots=True
    )
