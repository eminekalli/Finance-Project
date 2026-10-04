from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# MARKET CONFIGURATION
# ============================================================

TICKER = "XU030.IS"  # BIST 30 Endeksi

START_DATE = "2023-01-01"
END_DATE = "2026-10-02"


def get_market_data(
    ticker: str = TICKER,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    save_plot: bool = True,
    show_plot: bool = True,
) -> pd.DataFrame:

    # ============================================================
    # 1. PATH SETUP
    # ============================================================

    base_dir = Path(__file__).resolve().parent.parent

    # Ticker'dan dosya için güvenli isim üret
    asset_name = (
        ticker
        .replace("=", "")
        .replace(".", "_")
        .replace("-", "_")
        .lower()
    )

    csv_filename = f"{asset_name}_raw.csv"

    raw_path = base_dir / csv_filename

    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 2. DATA DOWNLOAD / LOAD
    # ============================================================

    print("=" * 70)
    print(f"{ticker} — DATA COLLECTION")
    print("=" * 70)

    print(f"Ticker     : {ticker}")
    print(f"Start Date : {start_date}")
    print(f"End Date   : {end_date}")

    if raw_path.exists():

        print("\nLocal CSV found.")
        print(f"Loading data from: {raw_path}")

        df = pd.read_csv(
            raw_path,
            index_col=0,
            parse_dates=True
        )

    else:

        print("\nLocal CSV not found.")
        print(f"Downloading {ticker} from Yahoo Finance...")

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if df.empty:

            raise ValueError(
                f"'{ticker}' için Yahoo Finance'den veri çekilemedi."
            )

        # yfinance MultiIndex düzeltmesi
        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)

        df.to_csv(raw_path)

        print(f"Raw data saved to: {raw_path}")

    # ============================================================
    # 3. DATA CLEANING
    # ============================================================

    if "Close" not in df.columns:

        raise ValueError(
            "Veri içerisinde 'Close' sütunu bulunamadı."
        )

    df = df[["Close"]].copy()

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Close"]
    )

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(
            keep="first"
        )
    ]

    # ============================================================
    # 4. LOG RETURN
    # ============================================================

    df["Log_Return"] = np.log(
        df["Close"] /
        df["Close"].shift(1)
    )

    # ============================================================
    # 5. DATA SUMMARY
    # ============================================================

    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    print(f"Asset        : {ticker}")
    print(f"Observations : {len(df)}")
    print(f"Start        : {df.index.min().date()}")
    print(f"End          : {df.index.max().date()}")

    print("\nMissing Values:")
    print(df.isna().sum())

    print(
        f"\nDuplicate Dates : "
        f"{df.index.duplicated().sum()}"
    )

    print("\nFirst 5 observations:")
    print(df.head())

    # ============================================================
    # 6. PRICE PLOT
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        df.index,
        df["Close"],
        linewidth=1.3,
        label=f"{ticker} Close"
    )

    ax.set_title(
        f"{ticker} — Daily Closing Price",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    ax.legend(
        loc="upper left"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    # ============================================================
    # 7. SAVE / SHOW
    # ============================================================

    if save_plot:

        plot_path = (
            figures_dir /
            f"{asset_name}_close_price.png"
        )

        fig.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"\nPrice plot saved to: "
            f"{plot_path}"
        )

    if show_plot:

        plt.show()

    else:

        plt.close(fig)

    # ============================================================
    # 8. RETURN DATA
    # ============================================================

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    data = get_market_data(
        save_plot=True,
        show_plot=True
    )