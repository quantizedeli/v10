# PFAZ 10 — Thesis Compilation

Tüm PFAZ çıktılarını toplayarak 8 adımda tam LaTeX tezi oluşturur.

## Ana Sınıf

**`MasterThesisIntegration`** (`pfaz10_master_integration.py`)

```python
from pfaz_modules.pfaz10_thesis_compilation import MasterThesisIntegration

thesis = MasterThesisIntegration(
    project_dir="outputs",
    output_dir="outputs/thesis",
    pfaz_outputs={
        1: "outputs/generated_datasets",
        2: "outputs/trained_models",
        3: "outputs/anfis_models",
        4: "outputs/unknown_predictions",
        5: "outputs/cross_model_analysis",
        6: "outputs/reports",
        7: "outputs/ensemble_results",
        8: "outputs/visualizations",
        9: "outputs/aaa2_pfaz9_complete_results",
        10: "outputs/thesis",
        12: "outputs/advanced_analytics",
        13: "outputs/automl_results",
    },
    metadata={
        "title": "Machine Learning and ANFIS-Based Prediction of Nuclear Properties",
        "author": "...",
        "supervisor": "Prof. ...",
        "university": "Sakarya Universitesi",
        "department": "Fizik Bolumu",
        "thesis_type": "Master of Science",
    },
)
results = thesis.execute_full_pipeline(compile_pdf=False)
```

## 8 Adımlı Pipeline

| Adım | Metod | Açıklama |
|------|-------|---------|
| 1 | `_step1_collect_all_data()` | Tüm PFAZ çıktılarından figure, Excel, JSON topla |
| 2 | `_step2_generate_chapters()` | 14 bölüm + 4 ek → .tex dosyaları |
| 3 | `_step3_copy_figures()` | PFAZ8 PNG'lerini thesis/figures/ dizinine kopyala |
| 4 | `_step4_generate_tables()` | PFAZ çıktılarından LaTeX tabloları üret |
| 5 | `_step5_bibliography()` | BibTeX dosyası oluştur |
| 6 | `_step6_main_document()` | main.tex (\\include{chapters/*} ile) |
| 7 | `_step7_quality_checks()` | Eksik figür/bölüm kontrolü |
| 8 | `_step8_compile_pdf()` | pdflatex çalıştır (compile_pdf=True ise) |

## Adım 1: Veri Toplama Kaynakları

| PFAZ | Kaynak | Toplanan |
|------|--------|---------|
| 1 | `generation_summary.json` | Toplam veri seti sayısı |
| 2 | `metrics_*.json` (ilk 50) | AI model metrikleri |
| 3 | `metrics_*.json` (ilk 30) | ANFIS metrikleri |
| 4 | `Unknown_Nuclei_Results.xlsx` + `prediction_summary.json` | Test tahmini |
| 5 | `MASTER_CROSS_MODEL_REPORT*.xlsx` + `cross_model_summary.json` | Çapraz model |
| 6 | `THESIS_COMPLETE_RESULTS*.xlsx` | Nihai rapor |
| 7 | `ensemble_results*.json` + `ensemble_report*.xlsx` | Ensemble |
| 8 | `**/*.png` (rglob) | Tüm görselleştirmeler |
| 9 | `AAA2_Complete_*.xlsx` + `aaa2_analysis_summary.json` | Kontrol grubu |
| 12 | `pfaz12_statistical_tests*.xlsx` | İstatistiksel testler |
| 13 | `automl_summary.json` + `automl_retraining_log.json` | AutoML |

## Üretilen Bölümler (14 adet)

| Dosya | İçerik |
|-------|--------|
| `abbreviations.tex` | 40+ kısaltma tablosu (longtable) |
| `symbols.tex` | Nükleer fizik + ML + ANFIS semboller |
| `abstract.tex` | Abstract (EN) + Özet (TR) |
| `nuclear_theory.tex` | Kabuk modeli, SEMF, β₂, büyü sayıları |
| `introduction.tex` | Motivasyon, 11 hedef, bölüm planı |
| `methodology.tex` | 13 PFAZ pipeline, metrikler, CV, IQR, MC |
| `dataset.tex` | PFAZ1 — veri seti üretimi |
| `ai_training.tex` | PFAZ2 — AI model eğitimi |
| `anfis.tex` | PFAZ3 — ANFIS eğitimi |
| `results.tex` | PFAZ2/3 toplu sonuçlar |
| `unknown_predictions.tex` | PFAZ4 — bilinmeyen çekirdek tahminleri |
| `cross_model.tex` | PFAZ5 — çapraz model analizi |
| `ensemble.tex` | PFAZ7 — ensemble sonuçları |
| `statistical.tex` | PFAZ12 — istatistiksel testler |
| `automl.tex` | PFAZ13 — AutoML iyileştirme |
| `discussion.tex` | Yorumlama ve karşılaştırma |
| `conclusion.tex` | Özet + gelecek çalışmalar |

## 4 Ek (Appendix)

| Ek | İçerik |
|----|--------|
| A | Hiperparametre arama uzayları |
| B | Veri seti istatistikleri |
| C | Özellik açıklamaları |
| D | Excel rapor referansları |

## Yardımcı Fonksiyonlar

- `_safe_json(path)` — hata durumunda {} döner
- `_safe_excel_first_sheet(path, nrows)` — hata durumunda boş DataFrame
- `_df_to_latex(df, caption, label)` — booktabs formatında LaTeX tablo
- `_pfaz_path(pfaz_id, fallback_name)` — PFAZ çıktı dizinini çöz

## I/O Yapısı

**Çıkış:**
```
outputs/thesis/
├── main.tex                    <- Ana LaTeX belgesi
├── compile.bat / compile.sh    <- pdflatex + bibtex derleme betikleri
├── chapters/
│   ├── abbreviations.tex
│   ├── abstract.tex
│   ├── introduction.tex
│   └── ... (toplam 17 .tex)
├── appendices/
│   ├── A_hyperparams.tex
│   └── ...
├── figures/                    <- PFAZ8 PNG kopyaları
├── tables/                     <- LaTeX tablo dosyaları
├── bibliography/
│   └── references.bib
├── logs/
│   └── execution_report.json
└── thesis.pdf                  <- compile_pdf=True ise
```

## Modüller

| Dosya | Sınıf | Görev |
|-------|-------|-------|
| `pfaz10_master_integration.py` | MasterThesisIntegration | Ana 8-adım pipeline |
| `pfaz10_chapter_generator.py` | ChapterGenerator | Bölüm üretimi |
| `pfaz10_content_generator.py` | ContentGenerator | İçerik hazırlama |
| `pfaz10_latex_integration.py` | LatexIntegration | LaTeX dönüşüm araçları |
| `pfaz10_thesis_compilation_system.py` | ThesisCompilationSystem | Derleme sistemi |
| `pfaz10_thesis_orchestrator.py` | ThesisOrchestrator | Orkestrasyon |
| `pfaz10_visualization_qa.py` | VisualizationQA | Figür kalite kontrolü |
| `pfaz10_discussion_conclusion.py` | DiscussionConclusionGenerator | Tartışma/sonuç bölümleri |
