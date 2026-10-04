from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from src.data import TICKER
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch,
)
from statsmodels.tsa.filters.hp_filter import hpfilter

# ============================================================
# PARAMETRELER
# ============================================================
LAMBDA = 144000
ALPHA = 0.05
WINDOW = 3  # sıçrama tarihinin iki yanındaki işlem günü sayısı
JUMP_DATES = [
    # Negatif Şoklar
    "2023-01-05", "2023-01-11", "2023-02-01", "2023-02-07", 
    "2023-05-15", "2023-10-25", "2024-08-05", 
    "2025-03-19", "2025-03-21", "2026-05-21", "2026-09-16",
    # Pozitif Sıçramalar
    "2023-01-12", "2023-02-03", "2023-02-15", "2023-05-11", 
    "2023-06-05", "2024-04-05", "2025-06-30", "2025-09-15", 
    "2026-04-08"
]


#============================================================
#YARDIMCI FONKSİYONLAR
# ============================================================
def normality_tests(r: pd.Series, alpha: float = ALPHA) -> dict:
    """Jarque-Bera normallik testini çalıştırır."""
    jb_stat, jb_p = stats.jarque_bera(r)
    return {
        "jb_stat": jb_stat,
        "jb_p": jb_p,
        "skew": stats.skew(r, bias=False),
        "ex_kurt": stats.kurtosis(r, fisher=True, bias=False),
        "all_pass": bool(jb_p > alpha),
    }


def find_minimal_hp_correction(
    raw_price: pd.Series,
    lamb: float = LAMBDA,
    alpha: float = ALPHA,
    jump_dates=JUMP_DATES,
    window: int = WINDOW,
):
    """Sıçrama tarihlerinin çevresine lokal HP düzeltmesi uygular.
    
    Uçlardaki keskin kopmaları önlemek için 'Cosine Blending' (yumuşak geçiş) kullanır.
    """
    if window < 0:
        raise ValueError("window sıfır veya pozitif olmalıdır.")
    
    log_p = np.log(raw_price.astype(float))
    corrected = log_p.copy()
    replaced = []
    history = []
    local_trend = pd.Series(np.nan, index=corrected.index, dtype=float)
    local_cycle = pd.Series(np.nan, index=corrected.index, dtype=float)

    base = normality_tests(corrected.diff().dropna(), alpha)
    history.append({"k": 0, "date": pd.NaT, **base})

    dates = pd.to_datetime(list(jump_dates)).normalize()
    applicable = [d for d in dates if d in corrected.index]

    for jump_date in applicable:
        idx = corrected.index.get_loc(jump_date)
        start_idx = max(0, idx - window)
        end_idx = min(len(corrected) - 1, idx + window)
        
        sub = corrected.iloc[start_idx:end_idx + 1]
        if len(sub) <= 2:
            continue
        
        window_cycle, window_trend = hpfilter(sub, lamb=lamb)
        
        # --- KOSİNÜS AĞIRLIKLI YUMUŞATMA (BOUNDARY BLENDING) ---
        original_sub = sub.values
        trend_sub = window_trend.values
        n_sub = len(original_sub)
        
        weights = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, n_sub)))
        smoothed_values = original_sub * (1 - weights) + trend_sub * weights
        
        smoothed = pd.Series(smoothed_values, index=sub.index)
        
        corrected.iloc[start_idx:end_idx + 1] = smoothed
        local_trend.iloc[start_idx:end_idx + 1] = window_trend.values
        local_cycle.iloc[start_idx:end_idx + 1] = window_cycle.values
        replaced.extend(corrected.index[start_idx:end_idx + 1])
        
        res = normality_tests(corrected.diff().dropna(), alpha)
        history.append({"k": len(history), "date": jump_date, **res})

    replaced = list(dict.fromkeys(replaced))
    hist = pd.DataFrame(history)
    converged = bool(history[-1]["all_pass"])
    
    return corrected, replaced, hist, local_trend, local_cycle, converged


# ============================================================
# ANA FONKSİYON (DÜZELTME)
# ============================================================
def run_hp_filter(
    data,
    lamb: float = LAMBDA,
    alpha: float = ALPHA,
    jump_dates=JUMP_DATES,
    window: int = WINDOW,
):
    if data is None:
        raise ValueError("Market data bulunamadı.")
    if "Close" not in data.columns:
        raise ValueError("'Close' sütunu bulunamadı.")

    df = data.copy()
    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    raw_price = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if (raw_price <= 0).any():
        raise ValueError("Log dönüşümü için Close değerleri pozitif olmalıdır.")

    corrected_log, replaced, history, trend_log, cycle_log, converged = (
        find_minimal_hp_correction(raw_price, lamb, alpha, jump_dates, window)
    )
    corrected_price = np.exp(corrected_log)

    result = df.copy()
    result["Close_Original"] = raw_price
    result["Close"] = corrected_price
    result["Is_Corrected"] = result.index.isin(replaced)
    result["Correction_Order"] = np.nan
    for order, d in enumerate(replaced, start=1):
        result.loc[d, "Correction_Order"] = order

    result["Log_Return_Original"] = np.log(raw_price / raw_price.shift(1))
    result["Log_Return"] = np.log(corrected_price / corrected_price.shift(1))

    result["HP_Trend"] = np.exp(trend_log)
    result["HP_Cycle"] = cycle_log

    base_dir = Path(__file__).resolve().parent.parent

    output_path = base_dir / "xu030_is_hp_corrected.csv"
    result.to_csv(output_path, index_label="Date")
    history_path = base_dir / "xu030_hp_correction_history.csv"
    history.to_csv(history_path, index=False)
    result.attrs["history"] = history
    result.attrs["converged"] = converged

    print("\n" + "=" * 70)
    print(f"{TICKER} — LOKAL HP DÜZELTMESİ (LOG FİYAT)")
    print("=" * 70)
    print(f"Gözlem sayısı        : {len(result)}")
    print(f"Lambda               : {lamb}")
    print(f"Pencere              : ±{window} işlem günü")
    print(f"Eşleşen sıçrama tarihi: {len(history) - 1}")
    print(f"Anlamlılık düzeyi    : {alpha}")
    print(f"Düzeltilen gün sayısı: {len(replaced)}")
    print(f"Normallik sağlandı mı: {'EVET' if converged else 'HAYIR'}")

    return result


# ============================================================
# HP SONRASI GETİRİLERİN BROWNIAN/GBM TANILARI
# ============================================================
def run_hp_statistical_analysis(
    data: pd.DataFrame = None,
    alpha: float = ALPHA,
    save_plots: bool = True,
    show_plots: bool = True,
    verbose=False
) -> dict:
    """HP düzeltmesi sonrası log getirilerin Brownian/GBM tanılarını yapar.

    Tanılar HP döngüsü düzeyine değil, düzeltilmiş kapanış fiyatlarından
    hesaplanan log getirilere uygulanır.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent

    if data is None:
        corrected_data_path = BASE_DIR / "xu030_is_hp_corrected.csv"
        if not corrected_data_path.exists():
            raise FileNotFoundError(
                f"Düzeltilmiş veri dosyası bulunamadı: {corrected_data_path}"
            )
        analysis_data = pd.read_csv(
            corrected_data_path,
            index_col="Date",
            parse_dates=["Date"],
        ).sort_index()
    else:
        analysis_data = data.copy()

    if "Close" not in analysis_data.columns:
        raise ValueError("Düzeltilmiş veride 'Close' sütunu bulunamadı.")

    corrected_price = pd.to_numeric(
        analysis_data["Close"], errors="coerce"
    ).dropna()
    if (corrected_price <= 0).any():
        raise ValueError("Log getiri hesabı için düzeltilmiş Close değerleri pozitif olmalıdır.")

    # Aynı kapsamlı tanı bataryasını ham seridekiyle kullan; analyzer log
    # getirileri düzeltilmiş Close sütunundan yeniden üretir.
    from src.stat import run_statistical_analysis

    return run_statistical_analysis(
        data=analysis_data.loc[corrected_price.index],
        alpha=alpha,
        save_plots=save_plots,
        show_plots=show_plots,
    )


# ============================================================
# HAM vs DÜZELTİLMİŞ KARŞILAŞTIRMA
# ============================================================
def compare_raw_vs_corrected(result: pd.DataFrame, alpha: float = ALPHA,
                             lags=(1, 5, 10, 20)) -> pd.DataFrame:
    series = {
        "Ham": result["Log_Return_Original"].dropna(),
        "Düzeltilmiş": result["Log_Return"].dropna(),
    }
    rows = {}
    for name, r in series.items():
        n = normality_tests(r, alpha)
        lb = acorr_ljungbox(r, lags=list(lags), return_df=True)
        lb2 = acorr_ljungbox(r ** 2, lags=list(lags), return_df=True)
        arch_p = het_arch(r, nlags=10)[1]
        row = {
            "N": len(r),
            "Skew": n["skew"],
            "Excess Kurt": n["ex_kurt"],
            "JB p": n["jb_p"],
            "ARCH-LM p": arch_p,
        }
        for lag in lags:
            row[f"LB p (lag {lag})"] = lb.loc[lag, "lb_pvalue"]
            row[f"LB² p (lag {lag})"] = lb2.loc[lag, "lb_pvalue"]
        rows[name] = row

    table = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("HAM vs DÜZELTİLMİŞ GETİRİLER — NORMALLİK & BAĞIMLILIK")
    print("=" * 70)
    print(table.to_string())
    return table
