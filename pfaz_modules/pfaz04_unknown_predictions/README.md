# PFAZ 4 — Unknown Nuclei Predictions

Eğitim setinde yer almayan (test split) çekirdekler için tahmin yapar; val R² ile test R² karşılaştırarak genelleme degradasyonunu ölçer.

## Ana Sınıf

**`UnknownNucleiPredictor`** (`unknown_nuclei_predictor.py`)

```python
from pfaz_modules.pfaz04_unknown_predictions import UnknownNucleiPredictor

predictor = UnknownNucleiPredictor(
    ai_models_dir="outputs/trained_models",
    anfis_models_dir="outputs/anfis_models",
    splits_dir="outputs/generated_datasets",
    output_dir="outputs/unknown_predictions",
)
results = predictor.predict_unknown_nuclei()
```

## Çalışma Akışı

```
1. Her dataset klasöründe test.csv varlığını tara
2. Hedef sütunlarını tespit et (MM / Q veya QM / Beta_2)
3. _load_ai_models_for_dataset()   -> tüm PKL modelleri + metrics
4. _load_anfis_models_for_dataset() -> tüm ANFIS PKL modelleri
5. Her model için test.csv üzerinde predict()
6. _calculate_metrics() -> R2, RMSE, MAE
7. Degradasyon = val_R2 - test_R2
8. _process_predictions() -> all_results[] + per_nucleus_results[]
9. generate_excel_report()
```

## Depolanan Veriler

**`all_results`** listesi (model bazlı):
```
Dataset, Target, Model_Category (AI/ANFIS),
Model_Type, Config_ID,
Train_R2, Val_R2 (Known), Test_R2 (Unknown),
Degradation, Test_RMSE, Test_MAE, N_Test
```

**`per_nucleus_results`** listesi (çekirdek bazlı):
```
Dataset, Target, Model_Category, Model_Type, Config_ID,
NUCLEUS, y_true, y_pred, error, abs_error
```

## Excel Çıktısı: `Unknown_Nuclei_Results.xlsx`

| Sayfa | İçerik |
|-------|--------|
| All_Results | Tüm model-dataset kombinasyonları |
| Best_Per_Dataset | Dataset × Target × Kategori başına en iyi R² |
| Degradation_Analysis | Val-Test R² farkı (en kötüden en iyiye sıralı) |
| AI_vs_ANFIS | Yan yana karşılaştırma + kazanan sütunu |
| Pivot_By_Target | Target bazlı ortalama Test R² |
| Pivot_By_ModelType | Model türü bazlı ortalama Test R² |
| Per_Nucleus_Predictions | Bireysel çekirdek hataları |
| AAA2_Comparison | Her target için orijinal vs tahmin |

## I/O Yapısı

**Giriş:**
- `outputs/trained_models/{dataset}/{model_turu}/{config}/model_*.pkl`
- `outputs/anfis_models/{dataset}/{config}/model_*.pkl`
- `outputs/generated_datasets/{dataset}/test.csv` + `metadata.json`

**Çıkış:**
```
outputs/unknown_predictions/
├── Unknown_Nuclei_Results.xlsx
└── generalization_analysis/    <- GeneralizationAnalyzer çıktıları
```

## Modüller

| Dosya | Sınıf | Görev |
|-------|-------|-------|
| `unknown_nuclei_predictor.py` | UnknownNucleiPredictor | Ana sınıf, Excel üretimi |
| `single_nucleus_predictor.py` | SingleNucleusPredictor | Tek çekirdek CLI tahmini |
| `generalization_analyzer.py` | GeneralizationAnalyzer | Bölge bazlı genelleme analizi |
