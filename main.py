"""Uçtan uca XU030 veri toplama, HP düzeltme ve karşılaştırma akışı."""
from pathlib import Path

from src.data import get_market_data
from src.hp_plots import create_hp_report_plots
from src.hp_filter import (
    compare_raw_vs_corrected,
    correction_cost_report,
    run_hp_filter,
    run_hp_statistical_analysis,
    sensitivity_analysis,
)
from src.stat import run_statistical_analysis


def main():
    project_dir = Path(__file__).resolve().parent

    print("\n[1/7] BIST 30 verisi yükleniyor...")
    market_data = get_market_data(save_plot=True, show_plot=False)

    print("\n[2/7] Ham getiriler analiz ediliyor...")
    run_statistical_analysis(
        data=market_data,
        save_plots=True,
        show_plots=False,
        verbose=True,
    )

    print("\n[3/7] Lokal HP fiyat düzeltmesi uygulanıyor...")
    hp_result = run_hp_filter(
        data=market_data,
        lamb=144000,
        window=3,
        est_window=30,
        mode="greedy",
        output_dir=project_dir,
    )

    print("\n[4/7] Düzeltilmiş getiriler analiz ediliyor...")
    run_hp_statistical_analysis(
        data=hp_result,
        save_plots=True,
        show_plots=False,
        verbose=False,
    )

    print("\n[5/7] Ham ve düzeltilmiş getiriler karşılaştırılıyor...")
    compare_raw_vs_corrected(hp_result)

    print("\n[6/7] Düzeltme maliyeti hesaplanıyor...")
    correction_cost_report(hp_result)

    print("\n[7/7] Parametre duyarlılığı hesaplanıyor...")
    sensitivity = sensitivity_analysis(market_data["Close"], output_dir=project_dir)
    print("\nRapor grafikleri oluşturuluyor...")
    create_hp_report_plots(
        hp_result,
        sensitivity=sensitivity,
        output_dir=project_dir / "figures" / "hp_report",
    )

    print("\n✔ Tüm işlemler başarıyla tamamlandı.")
    print(f"Düzeltilmiş veri CSV'si kaydedildi: {project_dir / 'xu030_is_hp_corrected.csv'}")
    print(f"Tarih geçmişi CSV'si kaydedildi: {project_dir / 'xu030_hp_correction_history.csv'}")


if __name__ == "__main__":
    main()
