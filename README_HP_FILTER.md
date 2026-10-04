# Lokal HP düzeltmesi: kullanım ve sınırlılıklar

Yöntem, yalnızca önceden seçilmiş sıçrama tarihleri çevresinde log kapanışlara lokal HP trendi uygular. Geniş tahmin penceresi (`est_window`, varsayılan 30) trendi kestirir; dar değiştirme penceresi (`window`, varsayılan ±3 işlem günü) sonucu sınırlar. Yakın tarihler varsayılan olarak tek blokta birleştirilir. `mode="all"` geriye dönük varsayılandır; `sequential` JB normalliği sağlanınca durur, `greedy` JB istatistiğini en çok azaltan tarihi seçer.

```python
from src.hp_filter import run_hp_filter, sensitivity_analysis, correction_cost_report

result = run_hp_filter(market_data, mode="all", output_dir="outputs")
sensitivity = sensitivity_analysis(market_data["Close"], output_dir="outputs")
costs = correction_cost_report(result)
```

CSV'ler `xu030_is_hp_corrected.csv`, `xu030_hp_correction_history.csv` ve `xu030_hp_sensitivity.csv` adlarıyla yazılır. `result.attrs` içindeki geçmiş ve durum bilgileri CSV'ye aktarılmaz.

## Kullanım sınırları

HP filtresi çift yönlü olduğundan gelecek gözlemleri kullanır (look-ahead). Düzeltilmiş seri gerçek işlem fiyatı değildir; tahmin veya backtest girdisi olarak kullanılamaz. Bu işlem, seçilen sıçramaların çevresindeki getirileri değiştirir ve normalliği garanti etmez. Lambda, tahmin penceresi ve değiştirme penceresi sonuçları etkiler; duyarlılık tablosu bu etkiyi görünür kılar.
