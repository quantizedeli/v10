# PFAZ 1 — Dataset Generation

Ham nükleer veriyi (aaa2.txt) 848 eğitim/doğrulama/test veri setine dönüştürür.

## Ana Sınıf

**`DatasetGenerationPipelineV2`** (`dataset_generation_pipeline_v2.py`)

```python
pipeline = DatasetGenerationPipelineV2(
    source_data_path="data/aaa2.txt",
    output_dir="outputs/generated_datasets",
    targets=["MM", "QM", "Beta_2", "MM_QM"],
    feature_sets=None,          # None → TARGET_RECOMMENDED_SETS kullanılır
    scenarios=["S70", "S80"],
    nucleus_counts=[75, 100, 150, 200, "ALL"],
    scaling_methods=["NoScaling"],
    sampling_methods=["Random"],
)
pipeline.run_complete_pipeline()
```

## Çalışma Adımları

| Adım | Metod | Açıklama |
|------|-------|---------|
| 1 | `_load_raw_data()` | CSV/XLSX/TXT yükle, sütun adlarını standartlaştır |
| 2 | `_add_theoretical_calculations()` | SEMF, Woods-Saxon, Nilsson, Kabuk modeli özellikleri ekle |
| 3 | `_apply_qm_filtering()` | QM hedefi için QM=NaN satırları çıkar (219 çekirdek kalır) |
| 4 | `_perform_quality_control()` | IQR tabanlı aykırı değer tespiti (threshold=3.0) |
| 5 | `_generate_all_datasets()` | Tüm kombinasyonları üret, böl, kaydet |
| 6 | `_create_metadata_and_reports()` | metadata.json, exclusion tracker, datasets_summary.xlsx |

## Veri Seti İsimlendirme

```
{HEDEF}_{BOYUT}_{SENARYO}_{ÖZELLIK_KODU}_{ÖLÇEKLEME}_{ÖRNEKLEME}[_NoAnomaly]
```

Örnekler:
- `MM_150_S70_AZSMC_NoScaling_Random`
- `QM_200_S80_AZB2EMC_NoScaling_Random_NoAnomaly`
- `Beta_2_ALL_S70_MCZMNM_NoScaling_Random_NoAnomaly`

### Kısıtlar
- `SMALL_NUCLEUS_THRESHOLD = 100` — boyut ≤100 yalnızca S70 + Basic/Standard feature set üretir
- `NOANOMALY_SIZES = {150, 200, "ALL"}` — NoAnomaly varyantı yalnızca bu boyutlarda
- S70: %70 train / %15 val / %15 test
- S80: %80 train / %10 val / %10 test

## Özellik Setleri (`feature_combination_manager.py`)

60+ kombinasyon, hedef bazlı önerilen setler:

| Hedef | Set Sayısı | Örnek Setler |
|-------|-----------|--------------|
| MM | 13 | AZS, AZSMC, AZSMCBEPA, AZSNNNP |
| QM | 13 | AZB2E, AZB2EMC, AZB2EMCS |
| Beta_2 | 13 | MCZMNM, MCZMNMZV, MCZMNMZVNV |
| MM_QM | 11 | AZS, AZSMC, AZSMCB2E |

**FEATURE_ABBREV** — kısaltma → gerçek sütun adı:

| Kısaltma | Sütun | Kısaltma | Sütun |
|---|---|---|---|
| A | A | ZV | Z_valence |
| Z | Z | NV | N_valence |
| S | SPIN | ZSG | Z_shell_gap |
| MC | magic_character | NSG | N_shell_gap |
| BEPA | BE_per_A | BEP | BE_pairing |
| B2E | Beta_2_estimated | SPHI | spherical_index |
| ZMD | Z_magic_dist | CP | Q0_intrinsic |
| NMD | N_magic_dist | NN | Nn |
| BEA | BE_asymmetry | NP | Np |

Ayrık özellikler (A, Z, N, SPIN, PARITY, magic_character) **hiçbir zaman ölçeklenmez.**

## Ölçekleme (`scaling_manager.py`)

| Yöntem | Formül |
|--------|--------|
| NoScaling | ham değer (varsayılan) |
| Standard | (X − μ) / σ |
| Robust | (X − median) / IQR |

Ölçekleme yalnızca train özellikleri üzerinde fit edilir; val/test aynı parametreyle dönüştürülür. Hedef sütunları ölçeklenmez.

## I/O Yapısı

**Giriş:** `data/aaa2.txt` (267 çekirdek, 12 ham sütun)

**Çıkış:**
```
outputs/generated_datasets/
├── MM_75_S70_AZS_NoScaling_Random/
│   ├── train.csv       <- başlıksız, sayısal (özellikler + hedef)
│   ├── val.csv
│   ├── test.csv
│   ├── train.xlsx      <- NUCLEUS, A, Z, N + özellikler + hedef
│   ├── val.xlsx
│   ├── test.xlsx
│   └── metadata.json   <- feature_names, target_col, split_ratios, scaling...
├── ...
├── datasets_summary.xlsx
└── AAA2_enriched_all_nuclei.xlsx
```

**Toplam: 848 veri seti**

| Hedef | Standart | NoAnomaly | Toplam |
|-------|----------|-----------|--------|
| MM | 140 | 84 | 224 |
| QM | 150 | 90 | 240 |
| Beta_2 | 130 | 78 | 208 |
| MM_QM | 110 | 66 | 176 |

## Modüller

| Dosya | Sınıf | Görev |
|-------|-------|-------|
| `dataset_generation_pipeline_v2.py` | DatasetGenerationPipelineV2 | Ana orkestratör |
| `feature_combination_manager.py` | FeatureCombinationManager | 60+ özellik seti tanımları, SHAP öncelikleri |
| `io_config_manager.py` | InputOutputConfigManager, ScenarioManager | I/O config ve bölme oranları |
| `scaling_manager.py` | ScalingManager | Ayrık özellikleri koruyarak ölçekleme |
| `data_loader.py` | DataLoader | Ham veri yükleme, otomatik format tespiti |
| `data_quality_modules.py` | DataQualityChecker | IQR aykırı değer analizi |
| `qm_filter_manager.py` | QMFilterManager | QM=NaN filtreleme |
| `sampling_manager.py` | SamplingManager | Stratified / Random örnekleme |
| `data_enricher.py` | DataEnricher | Teorik özellik zenginleştirme |
| `control_group_generator.py` | ControlGroupGenerator | AAA2 kontrol grubu üretimi |
