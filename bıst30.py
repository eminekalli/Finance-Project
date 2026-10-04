import numpy as np
import yfinance as yf


ticker = "XU030.IS"
start_date = "2020-01-01"
end_date = None  


data = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    auto_adjust=False,
    progress=False
)

if data.empty:
    raise ValueError("Veri indirilemedi. Ticker kodunu veya internet bağlantısını kontrol edin.")


close = data["Close"]


if getattr(close, "ndim", 1) > 1:
    close = close.iloc[:, 0]

# Günlük log getiri: ln(P_t / P_(t-1))
data["Log_Return"] = np.log(close / close.shift(1))


output_file = "bist30_log_returns.xlsx"
data.to_excel(output_file, sheet_name="BIST30")

print(f"Dosya kaydedildi: {output_file}")