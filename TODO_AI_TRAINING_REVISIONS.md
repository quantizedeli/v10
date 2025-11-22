# AI Training Features Implementation - TODO List
**Proje:** Nükleer Fizik AI Tahmin Sistemi
**Tarih:** 2025-11-22
**Referans Doküman:** AI_TRAINING_FEATURE_COMBINATIONS.md
**Hedef Çekirdek Sayıları:** 75, 100, 150, 200, ALL

---

## 📋 Genel Bakış

Bu TODO listesi, AI_TRAINING_FEATURE_COMBINATIONS.md dokümanına göre projenin revize edilmesi için gereken tüm adımları içerir.

### Temel Değişiklikler
- ✅ Çekirdek sayıları: **75, 100, 150, 200, ALL** (250, 300, 350 kaldırıldı)
- ✅ 4 Target: MM, QM, MM_QM, Beta_2
- ✅ 43+ Feature (teorik hesaplamalar dahil)
- ✅ Görselleştirme ve raporlama basitleştirildi

---

## 🎯 Faz 1: Constants ve Konfigürasyon Güncellemeleri

### 1.1. core_modules/constants.py Güncelleme
- [ ] `NUCLEUS_COUNTS` değişkenini güncelle: `[75, 100, 150, 200, 'ALL']`
- [ ] Dataset boyutları açıklamalarını güncelle
- [ ] Eski boyutları (250, 300, 350) kaldır

**Dosya:** `core_modules/constants.py`

**Değişiklik:**
```python
# ESKI:
# NUCLEUS_COUNTS = [50, 75, 100, 150, 175, 200, 250, 300, 350, 'ALL']

# YENİ:
NUCLEUS_COUNTS = [75, 100, 150, 200, 'ALL']
```

**Açıklama Ekle:**
```python
# Dataset boyutları:
# - 75: Hızlı test ve prototipleme (~2-5 dk eğitim)
# - 100: Standart geliştirme (~5-10 dk eğitim)
# - 150: Orta ölçekli eğitim (~10-20 dk eğitim)
# - 200: Büyük ölçekli eğitim (~20-40 dk eğitim)
# - ALL: Final model (maksimum veri, ~120+ dk eğitim)
```

---

### 1.2. core_modules/constants_v1.1.0.py Güncelleme
- [ ] Aynı değişiklikleri `constants_v1.1.0.py` dosyasına da uygula
- [ ] Versiyon notlarını güncelle

---

### 1.3. config.json Güncelleme (eğer varsa)
- [ ] JSON config dosyasında nucleus_counts güncellemesi
- [ ] Dataset boyutları listesini güncelle

---

## 🎯 Faz 2: PFAZ 1 - Dataset Generation Pipeline

### 2.1. Dataset Generation Pipeline V2 Güncelleme

**Dosya:** `pfaz_modules/pfaz01_dataset_generation/dataset_generation_pipeline_v2.py`

- [x] `nucleus_counts` default değerini kontrol et (satır 84'te zaten doğru: `[75, 100, 150, 200, 'ALL']`)
- [ ] Tüm dataset generation fonksiyonlarının bu değerleri kullandığını doğrula
- [ ] QM filtreleme kurallarının doküman ile uyumlu olduğunu kontrol et
- [ ] Feature listesinin 43+ feature içerdiğini kontrol et

**Kontrol Edilecek:**
```python
# Satır 84'te zaten var:
self.nucleus_counts = nucleus_counts or [75, 100, 150, 200, 'ALL']
```

**QM Filtreleme Kuralları Kontrolü:**
- [ ] MM target: QM filtresi opsiyonel (Q feature varsa zorunlu)
- [ ] QM target: QM filtresi zorunlu
- [ ] MM_QM target: QM filtresi zorunlu
- [ ] Beta_2 target: QM filtresi opsiyonel (Q feature varsa zorunlu)

---

### 2.2. QM Filter Manager Kontrolü

**Dosya:** `pfaz_modules/pfaz01_dataset_generation/qm_filter_manager.py`

- [ ] `filter_by_target()` fonksiyonunun doküman kurallarına uygun çalıştığını doğrula
- [ ] Q-dependent features listesini kontrol et
- [ ] Filtreleme raporlarının detaylı olduğunu kontrol et

---

### 2.3. Theoretical Calculations Manager Kontrolü

**Dosya:** `physics_modules/theoretical_calculations_manager.py`

- [ ] SEMF features (11 adet) hesaplanıyor mu kontrol et
- [ ] Shell Model features (10 adet) hesaplanıyor mu kontrol et
- [ ] Schmidt Model features (2 adet) hesaplanıyor mu kontrol et
- [ ] Deformation features (4 adet) hesaplanıyor mu kontrol et
- [ ] Collective Model features (4 adet) hesaplanıyor mu kontrol et
- [ ] Woods-Saxon features (2 adet - opsiyonel) kontrol et
- [ ] Nilsson Model features (2 adet - opsiyonel) kontrol et

**Feature Sayısı Doğrulama:**
```
Eksperimental: 9 adet (A, Z, N, SPIN, PARITY, MM, Q, Beta_2, NUCLEUS)
Teorik: 35+ adet
Toplam: 44+ feature (NUCLEUS hariç 43+ training feature)
```

---

### 2.4. Dataset Generator Ana Script

**Dosya:** `dataset_generation_pipeline_v2.py` (root'ta)

- [ ] Script'in güncel nucleus_counts kullandığını kontrol et
- [ ] Backward compatibility sağlandığını doğrula

---

## 🎯 Faz 3: PFAZ 2 - AI Training Pipeline

### 3.1. Model Trainer Kontrolü

**Dosya:** `pfaz_modules/pfaz02_ai_training/model_trainer.py`

- [ ] RandomForest trainer'ı test et
- [ ] XGBoost trainer'ı test et
- [ ] DNN trainer'ı test et
- [ ] Multi-output regression (MM_QM için) çalışıyor mu kontrol et
- [ ] Feature importance hesaplaması yapılıyor mu kontrol et

---

### 3.2. Hyperparameter Tuner Kontrolü

**Dosya:** `pfaz_modules/pfaz02_ai_training/hyperparameter_tuner.py`

- [ ] 50 farklı konfigürasyon tanımlı mı kontrol et
- [ ] Her model tipi için (RF, XGB, DNN) konfigürasyonlar var mı
- [ ] Adaptive pruning stratejisi uygulanıyor mu

**Doküman Gereksinimleri:**
```
- RandomForest: 10 config
- XGBoost: 10 config
- GBM: 10 config
- DNN: 10 config
- BNN: 10 config (opsiyonel)
- PINN: 10 config (opsiyonel)
```

---

### 3.3. Parallel AI Trainer Kontrolü

**Dosya:** `pfaz_modules/pfaz02_ai_training/parallel_ai_trainer.py`

- [ ] Paralel eğitim desteği çalışıyor mu
- [ ] Adaptive pruning 3-stage (0.70, 0.80, 0.85 eşikleri) doğru mu
- [ ] Train-val-test split 60-20-20 oranında mı
- [ ] Overfitting kontrolü yapılıyor mu (train-check gap > 0.15)

---

### 3.4. Model Validator Kontrolü

**Dosya:** `pfaz_modules/pfaz02_ai_training/model_validator.py`

- [ ] R² metrikleri doğru hesaplanıyor mu
- [ ] RMSE metrikleri doğru hesaplanıyor mu
- [ ] MAE metrikleri doğru hesaplanıyor mu
- [ ] Overfitting detection çalışıyor mu

---

## 🎯 Faz 4: PFAZ 4 - All Nuclei Predictions

### 4.1. All Nuclei Predictor Kontrolü

**Dosya:** `pfaz_modules/pfaz04_unknown_predictions/all_nuclei_predictor.py`

- [ ] Tüm trained modelleri yükleyebiliyor mu
- [ ] ALL datasets (MM_ALL_267nuclei, QM_ALL_219nuclei, vb.) üzerinde tahmin yapabiliyor mu
- [ ] Delta hesaplaması (Prediction - Experimental) doğru mu
- [ ] Best model identification per nucleus çalışıyor mu
- [ ] Model agreement (std) hesaplanıyor mu

---

### 4.2. Excel Reporter Kontrolü

**Kontrol Edilecek Excel Formatı:**
- [ ] Sheet 1: MM_ALL_267nuclei (267 satır)
- [ ] Sheet 2: QM_ALL_219nuclei (219 satır)
- [ ] Sheet 3: MM_QM_ALL_219nuclei (219 satır, multi-output)
- [ ] Sheet 4: Beta_2_ALL_267nuclei (267 satır)
- [ ] Summary sheets (model performance, error analysis)

**Sütun Yapısı:**
```
NUCLEUS | A | Z | N | Experimental_Value |
Model1_Pred | Model1_Delta | Model1_Error |
Model2_Pred | Model2_Delta | Model2_Error | ...
Best_Model | Best_Pred | Best_Delta | Agreement | Classification
```

- [ ] Conditional formatting uygulanıyor mu (yeşil=iyi, kırmızı=kötü)
- [ ] Classification (EXCELLENT/GOOD/MEDIUM/POOR) doğru mu

---

### 4.3. Nucleus Classification Kriterleri

**Dokümana Göre:**
```python
EXCELLENT: avg_error < 0.05 && avg_agreement < 0.02
GOOD: avg_error < 0.15 && avg_agreement < 0.05
MEDIUM: avg_error < 0.30 && avg_agreement < 0.10
POOR: diğerleri
```

- [ ] Bu kriterlerin doğru uygulandığını kontrol et

---

## 🎯 Faz 5: PFAZ 6 - Final Reporting (Basitleştirilmiş)

### 5.1. Comprehensive Excel Reporter Basitleştirme

**Dosya:** `pfaz_modules/pfaz06_final_reporting/comprehensive_excel_reporter.py`

**Basitleştirme Adımları:**
- [ ] Sadece temel metrikleri içerecek şekilde güncelle
- [ ] Karmaşık pivot tableları kaldır veya opsiyonel yap
- [ ] Sadece kritik grafikleri koru

**Temel Raporlar (Korunacak):**
- [ ] Model performance summary (R², RMSE, MAE)
- [ ] Feature importance top 10
- [ ] Error distribution per target
- [ ] Best models per dataset

**Kaldırılacak/Opsiyonel:**
- [ ] Detaylı statistical analysis raporları
- [ ] Advanced correlation matrices
- [ ] Her nucleus için detaylı breakdown (sadece summary)

---

### 5.2. Excel Charts Basitleştirme

**Dosya:** `pfaz_modules/pfaz06_final_reporting/excel_charts.py`

**Temel Grafikler (Korunacak):**
- [ ] R² comparison bar chart (modeller arası)
- [ ] RMSE comparison bar chart
- [ ] Error distribution histogram
- [ ] Prediction vs Experimental scatter plot (test set)

**Kaldırılacak/Opsiyonel:**
- [ ] Karmaşık multi-panel charts
- [ ] Interactive charts (HTML için değil Excel için)
- [ ] Advanced statistical plots

---

### 5.3. LaTeX Generator (Opsiyonel)

**Dosya:** `pfaz_modules/pfaz06_final_reporting/latex_generator.py`

- [ ] Bu modülü opsiyonel yap (sadece thesis için kullanılacak)
- [ ] Default olarak devre dışı bırak

---

## 🎯 Faz 6: PFAZ 8 - Visualization (Basitleştirilmiş)

### 6.1. AI Visualizer Basitleştirme

**Dosya:** `pfaz_modules/pfaz08_visualization/ai_visualizer.py`

**Temel Görseller (Korunacak):**
- [ ] Training curves (loss vs epoch)
- [ ] R² evolution (train/val/test)
- [ ] Prediction scatter plots
- [ ] Feature importance bar charts (top 10)

**Kaldırılacak/Opsiyonel:**
- [ ] 3D plots
- [ ] Interactive dashboards (opsiyonel)
- [ ] Animation'lar
- [ ] Complex multi-panel figures

---

### 6.2. SHAP Analysis Kontrolü

**Dosya:** `pfaz_modules/pfaz08_visualization/shap_analysis.py`

- [ ] SHAP summary plot çalışıyor mu
- [ ] SHAP waterfall plot (individual predictions) opsiyonel yap
- [ ] SHAP dependence plots opsiyonel yap

**Temel SHAP (Korunacak):**
- [ ] Global feature importance (summary_plot)
- [ ] Top 10 features SHAP values

---

### 6.3. Model Comparison Dashboard (Opsiyonel)

**Dosya:** `pfaz_modules/pfaz08_visualization/model_comparison_dashboard.py`

- [ ] Bu modülü tamamen opsiyonel yap
- [ ] Sadece production deployment için aktifleştir

---

### 6.4. Diğer Visualization Modülleri

- [ ] `anomaly_visualizations_complete.py` - Opsiyonel yap
- [ ] `log_analytics_visualizations_complete.py` - Opsiyonel yap
- [ ] `interactive_html_visualizer.py` - Opsiyonel yap
- [ ] `robustness_visualizations_complete.py` - Opsiyonel yap

---

## 🎯 Faz 7: Ana Script'ler Güncelleme

### 7.1. main.py Güncelleme

**Dosya:** `main.py`

- [ ] NUCLEUS_COUNTS değişkenini kontrol et
- [ ] Dataset generation çağrılarını kontrol et
- [ ] AI training çağrılarını kontrol et
- [ ] All nuclei prediction çağrılarını kontrol et

---

### 7.2. main_revised.py Güncelleme

**Dosya:** `main_revised.py`

- [ ] Güncel workflow'u kullandığını kontrol et
- [ ] Basitleştirilmiş raporlama çağrılarını kullanıyor mu

---

### 7.3. Example Scripts Kontrolü

- [ ] `example_usage.py` - Güncel API kullanıyor mu
- [ ] `example_performance_pipeline.py` - Test çalışıyor mu

---

## 🎯 Faz 8: Test ve Validasyon

### 8.1. Unit Tests

**Klasör:** `tests/test_units/`

- [ ] Dataset generation unit testleri çalışıyor mu
- [ ] QM filter unit testleri çalışıyor mu
- [ ] Theoretical calculations unit testleri çalışıyor mu

---

### 8.2. Integration Tests

**Klasör:** `tests/test_integration/`

- [ ] End-to-end dataset generation testi
- [ ] End-to-end AI training testi (küçük dataset ile)
- [ ] All nuclei prediction testi

---

### 8.3. Smoke Tests

**Dosya:** `tests/test_smoke/test_basic_smoke.py`

- [ ] Tüm modüllerin import edildiğini test et
- [ ] Basit bir workflow'un çalıştığını test et

---

## 🎯 Faz 9: Dokümantasyon Güncellemeleri

### 9.1. README Güncellemeleri

**Kontrol Edilecek Dosyalar:**
- [ ] `README.md` (root)
- [ ] `DATASET_GUIDE.md`
- [ ] `pfaz_modules/pfaz01_dataset_generation/README.md`

**Güncelleme:**
- [ ] Nucleus counts bilgisini güncelle (75, 100, 150, 200, ALL)
- [ ] Feature sayısını güncelle (43+)
- [ ] Target'ları açıkla (MM, QM, MM_QM, Beta_2)

---

### 9.2. Yeni Doküman: Kullanım Kılavuzu

**Dosya:** `USAGE_GUIDE.md` (oluşturulacak)

- [ ] Quick start guide
- [ ] Dataset generation örnekleri
- [ ] AI training örnekleri
- [ ] Results interpretation

---

## 🎯 Faz 10: Performance Optimization (Opsiyonel)

### 10.1. Dataset Generation Hızlandırma

- [ ] Paralel teoretik hesaplamalar
- [ ] Caching mekanizması
- [ ] Checkpoint/resume özelliği

---

### 10.2. AI Training Hızlandırma

- [ ] GPU desteği kontrolü
- [ ] Batch processing optimization
- [ ] Model caching

---

## 📊 İlerleme Takibi

### Toplam Görevler
- **Faz 1:** 3 görev
- **Faz 2:** 4 görev
- **Faz 3:** 4 görev
- **Faz 4:** 3 görev
- **Faz 5:** 3 görev
- **Faz 6:** 4 görev
- **Faz 7:** 3 görev
- **Faz 8:** 3 görev
- **Faz 9:** 2 görev
- **Faz 10:** 2 görev (opsiyonel)

**Toplam:** ~31 ana görev

### Öncelik Sıralaması

**🔴 Kritik (Önce Yapılmalı):**
1. Faz 1: Constants güncelleme
2. Faz 2: Dataset generation kontrolü
3. Faz 4: All nuclei prediction kontrolü

**🟡 Önemli (Sonra Yapılmalı):**
4. Faz 3: AI training kontrolü
5. Faz 7: Ana scriptler güncelleme
6. Faz 8: Test ve validasyon

**🟢 İsteğe Bağlı (Son):**
7. Faz 5: Reporting basitleştirme
8. Faz 6: Visualization basitleştirme
9. Faz 9: Dokümantasyon
10. Faz 10: Performance optimization

---

## 🎯 Beklenen Sonuçlar

### Dataset Generation
- ✅ 4 target × 5 boyut = **20 dataset** (75, 100, 150, 200, ALL için)
- ✅ Her dataset için CSV + MAT + metadata.json
- ✅ QM filtreleme kuralları doğru uygulanmış

### AI Training
- ✅ ~1,000-1,500 model başarıyla eğitilmiş (20 dataset × 50 config)
- ✅ Adaptive pruning ile düşük performanslı modeller elenmiş
- ✅ Best modeller tanımlanmış (R² > 0.90)

### All Nuclei Predictions
- ✅ Excel raporu: **All_Nuclei_Predictions.xlsx**
- ✅ 4 sheet (MM, QM, MM_QM, Beta_2)
- ✅ Her çekirdek için best model belirlenmiş
- ✅ Classification: EXCELLENT/GOOD/MEDIUM/POOR

### Visualization & Reporting
- ✅ Basit ve anlaşılır grafikler
- ✅ Temel metrik raporları
- ✅ Feature importance analizi

---

## 📝 Notlar

### Önemli Kararlar
1. **Çekirdek Sayıları:** 75, 100, 150, 200, ALL (5 boyut)
2. **Görselleştirme:** Basitleştirilmiş (sadece temel grafikler)
3. **Raporlama:** Excel-based, basit ve anlaşılır
4. **Opsiyonel Modüller:** BNN, PINN, advanced visualizations

### Bağımlılıklar
```python
# Kritik:
numpy, pandas, scikit-learn, xgboost, openpyxl, joblib

# Opsiyonel:
tensorflow, torch, shap, matplotlib, seaborn
```

---

**Son Güncelleme:** 2025-11-22
**Durum:** TODO listesi oluşturuldu, implementasyon başlayabilir
**Sonraki Adım:** Faz 1'den başlayarak sistematik implementasyon
