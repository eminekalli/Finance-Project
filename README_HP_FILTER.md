# XU030 lokal HP düzeltme analizi

Bu proje, BIST 30 (Yahoo Finance sembolü `XU030.IS`) günlük kapanış fiyatlarında önceden belirlenmiş sıçrama tarihleri çevresinde lokal Hodrick–Prescott (HP) düzeltmesi uygular. Ham fiyatı “gerçek” veya tahmin edilebilir hale getirmez. Seçilen hareketler yumuşatıldığında getiri dağılımı, normallik testleri ve diğer tanıların nasıl değiştiğini inceleyen bir analiz akışıdır.

## Ne yapılıyor?

1. Günlük `Close` fiyatı yerel ham CSV’den okunur; yoksa Yahoo Finance’ten istenen tarih aralığı için indirilir.
2. Ham kapanışlardan log getiriler hesaplanır ve ham seri için istatistiksel tanılar üretilir.
3. `JUMP_DATES` içindeki tarihler, seçilen çalışma moduna göre değerlendirilir. HP filtresi ham log fiyat üzerinden hesaplanır; yalnızca sıçrama tarihlerinin dar çevresi sin ağırlıklarıyla trend yönünde yumuşatılır.
4. Düzeltilmiş fiyat serisi ve her düzeltme adımının JB sonuçları CSV olarak kaydedilir.
5. Ham/düzeltilmiş getiriler karşılaştırılır; değiştirilen gün sayısı, getiri değişimleri ve varyans farkı raporlanır.
6. Farklı pencere ve lambda ayarları için duyarlılık tablosu hesaplanır.
7. Altı ana rapor grafiği ve duyarlılık tablosu varsa yedinci grafik olan ısı haritası PNG olarak kaydedilir.

## Çalıştırma

Terminal’de proje klasörüne geçip ana programı çalıştırın:

```bash
cd "/Users/eminekalli/Desktop/Finance Project"
.venv/bin/python main.py
```

Testleri ayrıca çalıştırmak için:

```bash
.venv/bin/python -m pytest tests/test_hp_filter.py -q
```

Test dosyası uygulama çalışırken kendiliğinden çağrılmaz; kodun temel davranışlarını bağımsız doğrular.

## Ana parametreler

| Ayar | Varsayılan | Anlamı |
|---|---:|---|
| `TICKER` | `XU030.IS` | Veri kaynağındaki BIST 30 sembolü |
| Tarih aralığı | `2023-01-01` – `2026-10-02` | `src/data.py` içindeki indirme aralığı; bitiş günü veri sağlayıcısında dahil olmayabilir |
| `LAMBDA` | `144000` | HP trendinin yumuşaklık parametresi |
| `WINDOW` | `3` | Sıçrama başına dar düzeltme penceresi: merkez tarihin ±3 işlem günü |
| `EST_WINDOW` | `30` | HP trendini tahmin ederken blok çevresine eklenen işlem günü |
| `ALPHA` | `0.05` | Jarque–Bera testi için durma/yorumlama eşiği |
| `merge_close_dates` | `True` | Birbirine yakın sıçrama tarihlerini tek bir düzeltme bloğunda birleştirir |
| `mode` (`main.py`) | `greedy` | Adım adım JB istatistiğini en fazla düşüren tarihi seçer |

`WINDOW` düzeltmenin uygulanacağı aralığı, `EST_WINDOW` ise trendin tahmin edildiği daha geniş bağlamı belirler. Seri başlangıcında veya sonunda pencereler veri sınırına göre kırpılır. Yakın olaylar birleştirilince tek blok için tek HP trendi ve cosine ağırlığı kullanılır. Bir tarih indekste yoksa uyarı üretilir; `snap_to_next_trading_day=True` verilirse sonraki işlem gününe kaydırılabilir.

`JUMP_DATES`, her tarih için `yon` ve henüz doldurulmamış `olay` açıklamasını tutar. Olay alanları kod tarafından otomatik bulunmaz; kaynak/olay gerekçesi kullanıcı tarafından belgelenmelidir.

## Düzeltme nasıl hesaplanıyor?

Kapanış fiyatı `P_t` önce `log(P_t)` biçimine çevrilir. Sıçrama tarihleri indeks üzerinde işlem günü konumlarına eşlenir. HP filtresi her blok için ham log fiyatın daha geniş alt serisine uygulanır ve trend bileşeni elde edilir. Trend, yalnızca dar düzeltme bloğunda orijinal log fiyatla karıştırılır:

`yeni log fiyat = ham log fiyat × (1 − ağırlık) + HP trendi × ağırlık`

Cosine ağırlığı pencere uçlarında sıfır, ortasına doğru daha büyüktür; böylece geçiş yumuşatılır. Örtüşen ve birleştirilmemiş alanlarda en büyük ağırlık korunur. Sonra düzeltilmiş log fiyat üstel dönüşümle tekrar fiyat düzeyine (`Close`) çevrilir.

### Çalışma modları

- **`all`**: Eşleşen tüm sıçrama tarihlerini uygular. `find_minimal_hp_correction` fonksiyonunun varsayılanıdır.
- **`sequential`**: Tarihleri sıralı biçimde ekler ve JB p-değeri `alpha` değerini aşınca durur.
- **`greedy`**: Her turda kalan tarihler arasından JB istatistiğini en çok düşüreni seçer; JB p-değeri `alpha` değerini aşınca veya JB istatistiği artık düşmeyince durur. `main.py` şu anda bu modu kullanır.

Greedy seçim, olası tüm tarih kümelerini tarayan kesin optimizasyon değildir; her turda en iyi tek adımı seçen sezgisel yöntemdir. Dolayısıyla matematiksel olarak en az değiştirilen gün sayısını garanti etmez.

## Fonksiyonlar

### `src/hp_filter.py`

- `_date_metadata(jump_dates)`: Tarih listesini veya tarih/metadata sözlüğünü ortak biçime dönüştürür.
- `normality_tests(r, alpha)`: Getirilerden JB istatistiği/p-değeri, çarpıklık ve excess kurtosis hesaplar.
- `_prepare(raw_price)`: Fiyat serisini sayısallaştırır, tarih indeksini temizleyip sıralar; sıfır/negatif fiyatı reddeder.
- `_blocks(positions, window, merge_close_dates, n)`: Yakın tarihleri gruplayıp dar düzeltme bloklarını seri sınırlarına göre belirler.
- `_correction(...)`: Ham log seride geniş pencere HP trendini hesaplar, dar pencerede ağırlıklı düzeltme yapar ve eşleşme/blok bilgilerini döndürür.
- `find_minimal_hp_correction(...)`: `all`, `sequential`, `greedy` modlarını çalıştırır. Düzeltilmiş log fiyatı, değişen tarihleri, history tablosunu, HP trend/döngüsünü ve normallik durumunu döndürür.
- `run_hp_filter(data, ...)`: DataFrame’deki `Close` sütununa düzeltme uygular; analiz sütunlarını ve iki temel CSV’yi yazar.
- `sensitivity_analysis(raw_price, ...)`: `window=(1,2,3,5)`, `est_window=(15,30,60)`, `lambda=(1600,14400,144000)` varsayılanlarının 36 kombinasyonunu hesaplar. Her kombinasyonu `mode="all"` ile değerlendirir; ana akıştaki greedy sonucuyla karıştırılmamalıdır.
- `correction_cost_report(result)`: Değiştirilen gün sayısı/oranı, olay tarihlerindeki getiri farkı ve kaldırılan varyans oranını raporlar. Ham ve düzeltilmiş toplam log getirinin eşitliğini assertion ile kontrol eder.
- `detect_jumps_by_threshold(log_returns, k=3)`: `|getiri| > k × standart sapma` koşulunu sağlayan tarihleri döndürür.
- `compare_manual_vs_objective_dates(raw_price, ...)`: Elle seçilmiş tarihler ile eşik yöntemi tarihlerini aynı HP ayarlarıyla düzeltip iki log seri döndürür.
- `plot_history(history)`: JB p-değerini düzeltme turu `k`'ya karşı çizer.
- `run_hp_statistical_analysis(data, ...)`: Düzeltilmiş fiyatlardan getirileri `src.stat` tanılarına aktarır.
- `compare_raw_vs_corrected(result, ...)`: Ham/düzeltilmiş getiriler için JB, ARCH-LM ve Ljung–Box (getiriler ile kare getiriler) karşılaştırmasını üretir.

### `src/hp_plots.py`

- `create_hp_report_plots(result, sensitivity, output_dir)`: Ana altı grafiği, sensitivity tablosu varsa ayrıca ısı haritasını PNG dosyaları olarak kaydeder. Grafikleri göstermek yerine dosyaya yazmak için `Agg` backend kullanır.

### `src/data.py` ve `src/stat.py`

- `get_market_data(...)`: `xu030_is_raw.csv` varsa onu okur; yoksa Yahoo Finance’ten indirir. `Close` alanını temizler, `Log_Return` ekler ve fiyat grafiğini kaydeder.
- `run_statistical_analysis(...)`: Ham veya düzeltilmiş fiyat verisinden getiri tanılarını ve genel istatistik panelini üretir. Düzeltilmiş analizde `main.py`, `hp_result` DataFrame’ini gönderir.

## `main.py` çalışma sırası

1. `get_market_data(save_plot=True, show_plot=False)` ham veriyi yükler/indirir.
2. `run_statistical_analysis(...)` ham log getirileri inceler.
3. `run_hp_filter(..., mode="greedy", lamb=144000, window=3, est_window=30)` lokal düzeltmeyi uygular ve CSV’leri kaydeder.
4. `run_hp_statistical_analysis(...)` düzeltilmiş seriye tanı uygular.
5. `compare_raw_vs_corrected(...)` iki getiri serisini karşılaştırır.
6. `correction_cost_report(...)` düzeltmenin maliyetini raporlar.
7. `sensitivity_analysis(...)` 36 parametre kombinasyonunu hesaplar; ardından `create_hp_report_plots(...)` rapor grafiklerini yazar.

## Üretilen dosyalar

Dosyalar proje klasörüne göreli yollara yazılır. `main.py` çalıştırıldığında:

| Dosya | İçerik |
|---|---|
| `xu030_is_raw.csv` | Veri sağlayıcıdan indirilen ham OHLCV/Close verisi; sonraki çalıştırmalarda önbellek olarak okunur |
| `xu030_is_hp_corrected.csv` | Ham alanlar, ham kapanış, düzeltilmiş `Close`, değişiklik bayrakları, getiriler ve HP bileşenleri |
| `xu030_hp_correction_history.csv` | Başlangıç ve her düzeltme adımına ait JB/statistik geçmişi; greedy modda seçilen ve gereksiz tarih durumları |
| `xu030_hp_sensitivity.csv` | Pencere/lambda birleşimleri için 36 satırlık duyarlılık tablosu |
| `figures/xu030_is_close_price.png` | Ham günlük kapanış fiyatı grafiği |
| `figures/xu030_is_diagnostics_panel.png` | `src.stat` tarafından üretilen genel istatistik paneli |
| `figures/hp_report/01_raw_vs_corrected_log_price.png` | Ham ve düzeltilmiş log fiyat |
| `figures/hp_report/02_jump_date_zoom_panels.png` | Her manuel sıçrama çevresinde ±10 işlem günlük yakın plan; greedy için seçilen/gereksiz durum başlıklarda |
| `figures/hp_report/06_return_histogram_vs_normal.png` | Ham ve düzeltilmiş getiri histogramları ve normal yoğunlukları |
| `figures/hp_report/07_qq_raw_vs_corrected.png` | Ham ve düzeltilmiş getiri Q-Q grafikleri |
| `figures/hp_report/10_jb_p_by_k.png` | History’deki `k` adımına göre JB p-değeri ve 0,05 çizgisi |
| `figures/hp_report/12_squared_return_acf.png` | Kare ham/düzeltilmiş getirilerin ACF karşılaştırması |
| `figures/hp_report/15_sensitivity_heatmap_jb_p.png` | Her lambda için pencere duyarlılığı ısı haritası; renk `−log10(JB p)`, hücre metni gerçek p-değeridir |

Düzeltilmiş CSV’de önemli sütunlar:

- `Close_Original`: Ham kapanış fiyatı.
- `Close`: Düzeltilmiş kapanış fiyatı.
- `Is_Corrected`: Düzeltme bloğunda ağırlığı sıfır olmayan günü işaretler.
- `Correction_Order`: Değişen gözlemlere tarih sırasıyla verilen sıra numarası; greedy’nin tarih seçme sırası değildir.
- `Log_Return_Original`, `Log_Return`: Ham ve düzeltilmiş log getiriler.
- `HP_Trend`, `HP_Cycle`: İkisi de log düzeyindedir; trend/döngü değerleri ilgili düzeltme bölgelerinde bulunur.
- `HP_Trend_Price`: Log trendin fiyat düzeyine çevrilmiş karşılığı.

History CSV sütunları: `k`, `date`, `status`, `jb_stat`, `jb_p`, `skew`, `ex_kurt`, `all_pass`. Greedy modunda `effective` seçilen, `unnecessary` seçilmeyen tarihtir. `result.attrs` metadata’sı CSV’ye yazılmaz; history ayrıca kaydedilir.

## Testler

`tests/test_hp_filter.py` şu dört özelliği kontrol eder:

1. `JUMP_DATES` sırası karışınca sonuç aynı kalır.
2. Serinin uç log fiyatları sabit ve toplam log getiri korunur.
3. Başlangıçtaki düzeltme penceresi veri sınırına göre güvenle kırpılır.
4. İndekste olmayan sıçrama tarihi uyarı üretir.

Testleri çalıştırma komutu:

```bash
.venv/bin/python -m pytest tests/test_hp_filter.py -q
```

## Yorumlama ve sınırlılıklar

- HP filtresi çift yönlüdür; trend hesabı olay sonrasındaki gözlemleri de kullanır (look-ahead).
- Düzeltilmiş `Close` sentetik analiz serisidir, gerçek işlem fiyatı değildir; tahmin veya backtest girdisi olarak kullanılmamalıdır.
- Bu akışta tarih seçimi ve pencere/Lambda kararları getiri dağılımını, özellikle JB normallik sonucunu etkiler. Normalliğin sağlanması gerçek piyasanın normalleştiğini kanıtlamaz.
- Cosine blending uçlardaki ağırlığı sıfırladığı için düzeltme penceresindeki her gözlem mutlaka değişmeyebilir.
- `sensitivity_analysis` tablo/ısı haritası üretirken `all` modu ile tüm manuel tarihleri uygular; `main.py` ana sonucu ise `greedy` modundadır. Bu iki sonuç ayrı senaryolardır.
- Olay tarihleri ve `olay: "TODO"` açıklamaları kullanıcı tarafından kaynak gösterilerek belgelenmelidir.
