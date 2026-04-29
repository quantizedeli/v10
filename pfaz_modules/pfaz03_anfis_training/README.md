# PFAZ 3 — ANFIS Training

Gerçek Takagi-Sugeno ANFIS ile 8 üyelik fonksiyonu konfigürasyonunu paralel eğitir.

## Ana Sınıf

**`ANFISParallelTrainerV2`** (`anfis_parallel_trainer_v2.py`)

```python
from pfaz_modules.pfaz03_anfis_training import ANFISParallelTrainerV2

trainer = ANFISParallelTrainerV2(
    datasets_dir="outputs/generated_datasets",
    output_dir="outputs/anfis_models",
    n_workers=None,
    use_config_manager=True,
    use_adaptive_strategy=False,
    use_performance_analyzer=True,
    save_datasets=True,
)
results = trainer.train_all_anfis(n_configs=50)
```

## ANFIS Implementasyonu: `TakagiSugenoANFIS`

Tip-1 Takagi-Sugeno ANFIS — MATLAB gerekmez.

### Mimari
```
Giriş Katmanı
  → Fuzzy-leştirme (MF hesaplama)
  → Kural Ateşleme Gücü (ürün veya min t-normu)
  → Normalizasyon
  → Sonuç Katmanı (lineer: p_r * [1, x1..xn])
  → Toplam Çıkış (ağırlıklı ortalama)
```

### Hibrit Öğrenme
- **İleri geçiş:** LSE (Ridge regresyon) → sonuç parametreleri (p_r)
- **Geri geçiş:** L-BFGS-B → öncül parametreler (MF merkezleri, genişlikleri)
- EarlyStopping: val loss bazlı, patience=30
- Dahili StandardScaler normalizasyonu

### Yapıcı Parametreler
| Parametre | Açıklama |
|-----------|---------|
| `n_inputs` | Girdi sayısı |
| `n_mfs` | Girdi başına üyelik fonksiyonu sayısı (adaptif) |
| `mf_type` | `gaussian`, `bell`, `triangle`, `trapezoid` |
| `method` | `grid` (ızgara bölümleme) veya `subclust` (KMeans tabanlı) |
| `max_iter` | Maksimum iterasyon (varsayılan 300) |
| `patience` | EarlyStopping patience (30) |

## 8 Konfigürasyon

| Konfigürasyon | Yöntem | MF Tipi | n_mfs | Kural Sayısı (3 girdi) |
|---|---|---|---|---|
| CFG_Grid_2MF_Gauss | grid | gaussian | 2 | 8 |
| CFG_Grid_2MF_Bell | grid | bell | 2 | 8 |
| CFG_Grid_2MF_Tri | grid | triangle | 2 | 8 |
| CFG_Grid_2MF_Trap | grid | trapezoid | 2 | 8 |
| CFG_Grid_3MF_Gauss | grid | gaussian | 3 | 27 |
| CFG_Grid_3MF_Bell | grid | bell | 3 | 27 |
| CFG_SubClust_5 | subclust | gaussian | 5 küme | 5 |
| CFG_SubClust_8 | subclust | gaussian | 8 küme | 8 |

## Adaptif Kural Sayısı Yönetimi

`_adaptive_n_mfs()` — kural sayısı kural: `n_rules < n_train / 3`

| Girdi | Eğitim örneği | n_mfs | Kural |
|-------|--------------|-------|-------|
| 3 | ~52 | 2 | 8 (OK) |
| 3 | ~52 | 3 | 27 (sınırda) |
| 4 | ~52 | 2 | 16 (OK) |
| 5 | ~52 | 2 | 32 (adaptif düşürür) |

## Anomali Temizleme (Eğitim Sonrası)

`OutlierDetector` sınıfı:
1. Artık (residual) tabanlı IQR (multiplier=2.5) + Z-score (threshold=2.5) tespiti
2. İki yöntemin kesişimi aykırı olarak işaretlenir; <80% kalırsa sadece IQR kullanılır
3. Aykırı örnekler çıkarılır, model yeniden eğitilir
4. Val R² karşılaştırması → daha iyi model saklanır
5. Metadata'ya: n_outliers_detected, outlier_cleaning_applied kaydedilir

## Veri Yükleme

- `_load_split_csv(dataset_path, split)` — train/val/test.csv'yi yükler
- Başlıksız CSV (PFAZ1 formatı) + metadata.json desteği
- Hedef sütun tespiti dataset adındaki MM/QM/Beta_2 prefix'inden
- Özellik sütunları: NUCLEUS, target_cols dışındaki tüm sayısal sütunlar

## I/O Yapısı

**Giriş:** `outputs/generated_datasets/{dataset}/train.csv`, `val.csv`, `test.csv`

**Çıkış:**
```
outputs/anfis_models/
└── {dataset_adi}/
    └── {config_id}/
        ├── model_{config_id}.pkl       <- pickle'lanmış TakagiSugenoANFIS
        └── metrics_{config_id}.json    <- R2, RMSE, MAE, n_rules, mf_type, n_outliers
```

## Modüller

| Dosya | Sınıf | Görev |
|-------|-------|-------|
| `anfis_parallel_trainer_v2.py` | ANFISParallelTrainerV2, TakagiSugenoANFIS, OutlierDetector | Ana orkestratör + gerçek ANFIS + anomali temizleme |
| `anfis_config_manager.py` | ANFISConfigManager | 8 konfigürasyon tanımları |
| `anfis_adaptive_strategy.py` | ANFISAdaptiveStrategy | n_mfs adaptif seçim stratejileri |
| `anfis_dataset_selector.py` | ANFISDatasetSelector | ANFIS uygun dataset seçimi |
| `anfis_model_saver.py` | ANFISModelSaver | Model kayıt/yükleme |
| `anfis_performance_analyzer.py` | ANFISPerformanceAnalyzer | Sonuç analizi ve Excel raporu |
| `anfis_robustness_tester.py` | ANFISRobustnessTester | Gürültü ve pertürbasyon testi |
| `anfis_visualizer.py` | ANFISVisualizer | MF grafikleri, öğrenme eğrisi |
| `matlab_anfis_trainer.py` | MatlabANFISTrainer | MATLAB engine entegrasyonu (opsiyonel fallback) |
