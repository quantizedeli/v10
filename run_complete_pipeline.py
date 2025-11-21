"""
Tüm Modülleri Sırayla Çalıştıran Ana Script
Complete Pipeline Runner - DÜZELTILMIŞ VERSIYON
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Encoding sorununu çöz
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Başlangıç banner'ı"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   NUCLEAR PHYSICS AI-ASSISTED DATA ANALYSIS PROJECT              ║
    ║                                                                   ║
    ║   🚀 OTOMATIK PIPELINE - TÜM MODÜLLER 🚀                          ║
    ║                                                                   ║
    ║   Tüm modülleri sırayla çalıştırır:                              ║
    ║   1. Veri Yükleme & Temizleme                                    ║
    ║   2. Teorik Hesaplamalar (SEMF)                                  ║
    ║   3. Anomali Tespiti                                             ║
    ║   4. Dataset Oluşturma                                           ║
    ║   5. AI Model Eğitimi (RF, GBM, XGBoost, DNN)                   ║
    ║   6. ANFIS Eğitimi (eğer MATLAB varsa)                          ║
    ║   7. Görselleştirme                                              ║
    ║   8. Raporlama                                                   ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def step_1_load_and_clean_data(input_file='aaa2.txt', output_dir='output'):
    """
    ADIM 1: Veri Yükleme ve Temizleme
    """
    try:
        # DÜZELTILMIŞ IMPORT
        from data_processing.data_loader import NuclearDataLoader
        
        logger.info("Veri yükleniyor...")
        loader = NuclearDataLoader(input_file)
        
        # Veriyi yükle
        raw_data = loader.load_data()
        logger.info(f"✓ {len(raw_data)} satır yüklendi")
        
        # Veriyi temizle
        logger.info("Veri temizleniyor...")
        cleaned_data = loader.clean_data()
        logger.info(f"✓ {len(cleaned_data)} satır temizlendi")
        
        # Kaydet
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / 'cleaned_data.csv'
        cleaned_data.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"✓ Temizlenmiş veri kaydedildi: {output_file}")
        
        return cleaned_data
        
    except Exception as e:
        logger.error(f"✗ ADIM 1 BAŞARISIZ: {e}", exc_info=True)
        raise


def step_2_theoretical_calculations(cleaned_data, output_dir='output'):
    """
    ADIM 2: Teorik Hesaplamalar (SEMF)
    """
    try:
        # DÜZELTILMIŞ IMPORT
        from nuclear_physics_modules.semf_calculator import TheoreticalCalculator
        
        logger.info("Teorik hesaplamalar yapılıyor (SEMF)...")
        calculator = TheoreticalCalculator()
        
        enriched_data = calculator.calculate_all_properties(cleaned_data)
        logger.info(f"✓ {len(enriched_data)} nükleus için hesaplamalar tamamlandı")
        
        # Kaydet
        output_file = Path(output_dir) / 'enriched_data.csv'
        enriched_data.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"✓ Zenginleştirilmiş veri kaydedildi: {output_file}")
        
        return enriched_data
        
    except Exception as e:
        logger.error(f"✗ ADIM 2 BAŞARISIZ: {e}", exc_info=True)
        raise


def step_3_anomaly_detection(enriched_data, output_dir='output'):
    """
    ADIM 3: Anomali Tespiti
    """
    try:
        # DÜZELTILMIŞ IMPORT
        from data_processing.anomaly_detector import AnomalyDetector
        
        logger.info("Anomali tespiti yapılıyor...")
        detector = AnomalyDetector()
        
        final_data = detector.detect_all_anomalies(enriched_data)
        logger.info(f"✓ Anomali tespiti tamamlandı")
        
        # İstatistikler
        normal_count = len(final_data[final_data['is_anomaly'] == False])
        anomaly_count = len(final_data[final_data['is_anomaly'] == True])
        logger.info(f"  - Normal: {normal_count}")
        logger.info(f"  - Anomali: {anomaly_count}")
        
        # Kaydet
        output_file = Path(output_dir) / 'final_data.csv'
        final_data.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"✓ Final veri kaydedildi: {output_file}")
        
        return final_data
        
    except Exception as e:
        logger.error(f"✗ ADIM 3 BAŞARISIZ: {e}", exc_info=True)
        raise


def step_4_generate_datasets(final_data, output_dir='output'):
    """
    ADIM 4: Dataset Oluşturma
    """
    try:
        # DÜZELTILMIŞ IMPORT
        from dataset_generation.dataset_generator import DatasetGenerator
        
        logger.info("Dataset'ler oluşturuluyor...")
        generator = DatasetGenerator(final_data, output_dir=f"{output_dir}/datasets")
        
        datasets = generator.generate_all_datasets()
        logger.info(f"✓ {len(datasets)} dataset oluşturuldu")
        
        return datasets
        
    except Exception as e:
        logger.error(f"✗ ADIM 4 BAŞARISIZ: {e}", exc_info=True)
        raise


def step_5_train_ai_models(datasets, output_dir='output'):
    """
    ADIM 5: AI Model Eğitimi
    """
    try:
        # DÜZELTILMIŞ IMPORT
        from ai_training.model_trainer import ModelTrainingPipeline
        
        logger.info("AI modelleri eğitiliyor...")
        pipeline = ModelTrainingPipeline(output_dir=f"{output_dir}/trained_models")
        
        results = pipeline.train_all_datasets(datasets)
        logger.info(f"✓ {len(results)} model eğitildi")
        
        return results
        
    except Exception as e:
        logger.error(f"✗ ADIM 5 BAŞARISIZ: {e}", exc_info=True)
        raise


def step_6_train_anfis(datasets, output_dir='output'):
    """
    ADIM 6: ANFIS Eğitimi (opsiyonel)
    """
    try:
        # DÜZELTILMIŞ IMPORT
        from anfis_training.anfis_operator import ANFISTrainer
        
        logger.info("ANFIS modelleri eğitiliyor...")
        trainer = ANFISTrainer(output_dir=f"{output_dir}/anfis_results")
        
        # MATLAB kontrolü
        if not trainer.matlab_interface.matlab_available:
            logger.warning("⚠ MATLAB bulunamadı, ANFIS eğitimi atlanıyor")
            return None
        
        results = trainer.train_selected_datasets(datasets)
        logger.info(f"✓ {len(results)} ANFIS modeli eğitildi")
        
        return results
        
    except Exception as e:
        logger.warning(f"⚠ ADIM 6 ATLANIYOR: {e}")
        return None


def step_7_visualization(results, output_dir='output'):
    """
    ADIM 7: Görselleştirme
    """
    try:
        # DÜZELTILMIŞ IMPORT
        from visualization.visualization_system import VisualizationManager
        
        logger.info("Görselleştirmeler oluşturuluyor...")
        viz = VisualizationManager(output_dir=f"{output_dir}/visualizations")
        
        viz.create_all_visualizations(results)
        logger.info(f"✓ Görselleştirmeler tamamlandı")
        
    except Exception as e:
        logger.error(f"✗ ADIM 7 BAŞARISIZ: {e}", exc_info=True)
        raise


def step_8_reporting(results, output_dir='output'):
    """
    ADIM 8: Raporlama
    """
    try:
        # DÜZELTILMIŞ IMPORT
        from reporting.reporting_system import ReportingManager
        
        logger.info("Raporlar oluşturuluyor...")
        reporter = ReportingManager(output_dir=f"{output_dir}/reports")
        
        reporter.generate_all_reports(results)
        logger.info(f"✓ Raporlar tamamlandı")
        
    except Exception as e:
        logger.error(f"✗ ADIM 8 BAŞARISIZ: {e}", exc_info=True)
        raise


def main():
    """Ana pipeline fonksiyonu"""
    
    print_banner()
    
    # Çıkış dizini
    output_dir = Path('output_results')
    output_dir.mkdir(exist_ok=True)
    
    start_time = datetime.now()
    logger.info(f"Pipeline başlangıç: {start_time}")
    
    try:
        # =====================================================================
        # ADIM 1: VERİ YÜKLEME VE TEMİZLEME
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("ADIM 1: VERİ YÜKLEME VE TEMİZLEME")
        logger.info("="*80)
        
        cleaned_data = step_1_load_and_clean_data(
            input_file='aaa2.txt',
            output_dir=output_dir
        )
        
        # =====================================================================
        # ADIM 2: TEORİK HESAPLAMALAR
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("ADIM 2: TEORİK HESAPLAMALAR (SEMF)")
        logger.info("="*80)
        
        enriched_data = step_2_theoretical_calculations(
            cleaned_data,
            output_dir=output_dir
        )
        
        # =====================================================================
        # ADIM 3: ANOMALİ TESPİTİ
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("ADIM 3: ANOMALİ TESPİTİ")
        logger.info("="*80)
        
        final_data = step_3_anomaly_detection(
            enriched_data,
            output_dir=output_dir
        )
        
        # =====================================================================
        # ADIM 4: DATASET OLUŞTURMA
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("ADIM 4: DATASET OLUŞTURMA")
        logger.info("="*80)
        
        datasets = step_4_generate_datasets(
            final_data,
            output_dir=output_dir
        )
        
        # =====================================================================
        # ADIM 5: AI MODEL EĞİTİMİ
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("ADIM 5: AI MODEL EĞİTİMİ")
        logger.info("="*80)
        
        ai_results = step_5_train_ai_models(
            datasets,
            output_dir=output_dir
        )
        
        # =====================================================================
        # ADIM 6: ANFIS EĞİTİMİ (OPSİYONEL)
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("ADIM 6: ANFIS EĞİTİMİ")
        logger.info("="*80)
        
        anfis_results = step_6_train_anfis(
            datasets,
            output_dir=output_dir
        )
        
        # =====================================================================
        # ADIM 7: GÖRSELLEŞTİRME
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("ADIM 7: GÖRSELLEŞTİRME")
        logger.info("="*80)
        
        all_results = {
            'ai': ai_results,
            'anfis': anfis_results
        }
        
        step_7_visualization(
            all_results,
            output_dir=output_dir
        )
        
        # =====================================================================
        # ADIM 8: RAPORLAMA
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("ADIM 8: RAPORLAMA")
        logger.info("="*80)
        
        step_8_reporting(
            all_results,
            output_dir=output_dir
        )
        
        # =====================================================================
        # TAMAMLANDI
        # =====================================================================
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*80)
        logger.info("✓ PIPELINE BAŞARIYLA TAMAMLANDI!")
        logger.info("="*80)
        logger.info(f"Başlangıç: {start_time}")
        logger.info(f"Bitiş: {end_time}")
        logger.info(f"Süre: {duration}")
        logger.info(f"Çıkış dizini: {output_dir.absolute()}")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"\n✗ Pipeline başarısız oldu: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()