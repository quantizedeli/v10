# -*- coding: utf-8 -*-
"""
Supplemental Visualizer — PFAZ9, PFAZ12, PFAZ13 Grafikler
===========================================================

PFAZ8 ana sistemi PFAZ1-7 verisini görselleştirir.
Bu modül PFAZ8'in ikinci geçişinde (tüm fazlar bittikten sonra)
PFAZ9/12/13 çıktılarını görselleştirir.

Grafik grupları:
  MC9  — Monte Carlo belirsizlik grafikleri (PFAZ9)
  ST12 — İstatistiksel test sonuçları (PFAZ12)
  AM13 — AutoML optimizasyon geçmişi ve iyileştirme raporları (PFAZ13)

Author: Nuclear Physics AI Project
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import logging
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

COLORS = {
    'MM':     '#2196F3',
    'QM':     '#4CAF50',
    'Beta_2': '#FF9800',
    'MM_QM':  '#FF5722',
}
DPI = 150


# =============================================================================
# MC9 — Monte Carlo Uncertainty (PFAZ9)
# =============================================================================

class MonteCarlо9Visualizer:
    """PFAZ9 Monte Carlo belirsizlik grafikleri."""

    def __init__(self, output_dir: Path):
        self.out = Path(output_dir) / 'mc9'
        self.out.mkdir(parents=True, exist_ok=True)

    def plot_uncertainty_distribution(self, pfaz9_dir: Path) -> list:
        """
        Hedef başına MC belirsizlik dağılımı (std, CV, CI genişliği).
        Kaynak: AAA2_Complete_{target}.xlsx → Uncertainty sheet
        """
        generated = []
        targets_data = {}

        for target in ['MM', 'QM']:
            excel = pfaz9_dir / f'AAA2_Complete_{target}.xlsx'
            if not excel.exists():
                continue
            try:
                df = pd.read_excel(excel, sheet_name='Uncertainty')
                if df.empty:
                    continue
                targets_data[target] = df
            except Exception as e:
                logger.warning(f"MC9 read {target}: {e}")

        if not targets_data:
            return generated

        # --- MC9-A: Std dağılımı violin ---
        fig, axes = plt.subplots(1, len(targets_data), figsize=(4 * len(targets_data), 5),
                                  sharey=False)
        if len(targets_data) == 1:
            axes = [axes]
        for ax, (target, df) in zip(axes, targets_data.items()):
            col = 'Std_Prediction'
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            parts = ax.violinplot([vals], showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(COLORS.get(target, '#888'))
                pc.set_alpha(0.7)
            ax.set_title(target, fontsize=11, fontweight='bold')
            ax.set_ylabel('Prediction Std', fontsize=9)
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle('MC9-A: Monte Carlo Tahmin Belirsizliği (Std)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = self.out / 'mc9a_uncertainty_std.png'
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        generated.append(str(path))
        logger.info(f"  [OK] {path.name}")

        # --- MC9-B: Yüksek belirsizlikli çekirdek sayısı bar ---
        fig, ax = plt.subplots(figsize=(8, 4))
        target_names, high_unc_counts, total_counts = [], [], []
        for target, df in targets_data.items():
            cv_col = 'CV'
            if cv_col not in df.columns:
                continue
            high = int((df[cv_col].dropna() > 0.3).sum())
            total = int(df[cv_col].dropna().count())
            target_names.append(target)
            high_unc_counts.append(high)
            total_counts.append(total)

        if target_names:
            x = np.arange(len(target_names))
            bars_total = ax.bar(x, total_counts, color='#BBDEFB', label='Toplam', width=0.6)
            bars_high  = ax.bar(x, high_unc_counts, color=[COLORS.get(t, '#888') for t in target_names],
                                label='Yüksek Belirsizlik (CV>0.3)', width=0.6, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(target_names)
            ax.set_ylabel('Çekirdek Sayısı')
            ax.set_title('MC9-B: Yüksek Belirsizlikli Çekirdek Sayısı', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            for bar, count in zip(bars_high, high_unc_counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        path = self.out / 'mc9b_high_uncertainty_nuclei.png'
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        generated.append(str(path))
        logger.info(f"  [OK] {path.name}")

        # --- MC9-C: CI genişliği vs Mean tahmin scatter ---
        for target, df in targets_data.items():
            if 'CI_Upper' not in df.columns or 'CI_Lower' not in df.columns:
                continue
            if 'Mean_Prediction' not in df.columns:
                continue
            ci_width = (df['CI_Upper'] - df['CI_Lower']).dropna()
            mean_pred = df.loc[ci_width.index, 'Mean_Prediction']

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(mean_pred.values, ci_width.values,
                       c=COLORS.get(target, '#888'), alpha=0.5, s=20, edgecolors='none')
            ax.set_xlabel(f'Ortalama Tahmin ({target})')
            ax.set_ylabel('95% CI Genişliği')
            ax.set_title(f'MC9-C: {target} CI Genişliği vs Tahmin', fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path = self.out / f'mc9c_ci_width_{target}.png'
            fig.savefig(path, dpi=DPI, bbox_inches='tight')
            plt.close(fig)
            generated.append(str(path))
            logger.info(f"  [OK] {path.name}")

        return generated


# =============================================================================
# ST12 — Statistical Tests (PFAZ12)
# =============================================================================

class Statistical12Visualizer:
    """PFAZ12 istatistiksel test sonuç grafikleri."""

    def __init__(self, output_dir: Path):
        self.out = Path(output_dir) / 'st12'
        self.out.mkdir(parents=True, exist_ok=True)

    def plot_statistical_tests(self, pfaz12_dir: Path) -> list:
        """
        PFAZ12 istatistiksel test Excel'inden grafik üret.
        Kaynak: pfaz12_statistical_tests.xlsx
        """
        generated = []

        # Find the Excel file
        excel_candidates = list(pfaz12_dir.rglob('pfaz12_statistical_tests.xlsx'))
        if not excel_candidates:
            excel_candidates = list(pfaz12_dir.rglob('statistical_tests.xlsx'))
        if not excel_candidates:
            logger.info("  [INFO] ST12: Statistical test Excel bulunamadı")
            return generated

        excel_path = excel_candidates[0]

        try:
            xl = pd.ExcelFile(excel_path)
        except Exception as e:
            logger.warning(f"  ST12 Excel açılamadı: {e}")
            return generated

        # --- ST12-A: Pairwise p-value heatmap ---
        for sheet_name in xl.sheet_names:
            if 'pairwise' in sheet_name.lower() or 'wilcoxon' in sheet_name.lower():
                try:
                    df = pd.read_excel(xl, sheet_name=sheet_name)
                    if df.empty:
                        continue
                    # Try to build a p-value matrix
                    p_cols = [c for c in df.columns if 'p_value' in c.lower() or 'p-value' in c.lower()]
                    model_a_col = next((c for c in df.columns if 'model_a' in c.lower()), None)
                    model_b_col = next((c for c in df.columns if 'model_b' in c.lower()), None)
                    if not (p_cols and model_a_col and model_b_col):
                        continue

                    # Pivot into matrix
                    models = sorted(set(df[model_a_col].tolist() + df[model_b_col].tolist()))
                    mat = pd.DataFrame(np.ones((len(models), len(models))),
                                       index=models, columns=models)
                    for _, row in df.iterrows():
                        ma, mb = row[model_a_col], row[model_b_col]
                        pv = float(row[p_cols[0]])
                        if ma in mat.index and mb in mat.columns:
                            mat.loc[ma, mb] = pv
                            mat.loc[mb, ma] = pv

                    fig, ax = plt.subplots(figsize=(max(6, len(models)), max(5, len(models) - 1)))
                    import matplotlib.colors as mcolors
                    cmap = plt.cm.RdYlGn_r
                    im = ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=0.2, aspect='auto')
                    ax.set_xticks(range(len(models)))
                    ax.set_yticks(range(len(models)))
                    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
                    ax.set_yticklabels(models, fontsize=9)
                    plt.colorbar(im, ax=ax, label='p-value')
                    ax.axhline(-0.5, color='k', lw=0.5)

                    # Annotate cells
                    for i in range(len(models)):
                        for j in range(len(models)):
                            v = mat.values[i, j]
                            color = 'white' if v < 0.05 else 'black'
                            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                                    fontsize=7, color=color)

                    ax.set_title('ST12-A: Pairwise İstatistiksel Test p-değerleri\n(α=0.05 altı yeşil = anlamlı)',
                                 fontweight='bold')
                    plt.tight_layout()
                    path = self.out / 'st12a_pvalue_heatmap.png'
                    fig.savefig(path, dpi=DPI, bbox_inches='tight')
                    plt.close(fig)
                    generated.append(str(path))
                    logger.info(f"  [OK] {path.name}")
                    break
                except Exception as e:
                    logger.warning(f"  ST12-A failed: {e}")

        # --- ST12-B: Model Val R² box plots (distribution comparison) ---
        # Try to load raw R² arrays from AI metrics directly
        ai_dir_candidates = [
            pfaz12_dir.parent / 'trained_models',
            pfaz12_dir.parent.parent / 'outputs' / 'trained_models',
        ]
        ai_models_dir = next((p for p in ai_dir_candidates if p.exists()), None)

        if ai_models_dir:
            model_scores = {}
            for metrics_file in ai_models_dir.rglob('metrics_*.json'):
                try:
                    with open(metrics_file, encoding='utf-8') as f:
                        m = json.load(f)
                    val_r2 = m.get('val', {}).get('r2', None)
                    if val_r2 is None or np.isnan(val_r2) or val_r2 < -10:
                        continue
                    mtype = metrics_file.parts[-3] if len(metrics_file.parts) >= 3 else 'unknown'
                    model_scores.setdefault(mtype, []).append(float(val_r2))
                except Exception:
                    continue

            if len(model_scores) >= 2:
                scores_filtered = {k: v for k, v in model_scores.items() if len(v) >= 3}
                if scores_filtered:
                    fig, ax = plt.subplots(figsize=(max(8, len(scores_filtered) * 1.5), 5))
                    data = [scores_filtered[k] for k in scores_filtered]
                    labels = list(scores_filtered.keys())
                    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
                    colors_list = plt.cm.Set2(np.linspace(0, 1, len(labels)))
                    for patch, color in zip(bp['boxes'], colors_list):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.8)
                    ax.axhline(0.7, color='green', linestyle='--', alpha=0.6, label='R²=0.7 (iyi)')
                    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.6, label='R²=0.5 (orta)')
                    ax.set_ylabel('Val R²')
                    ax.set_title('ST12-B: Model Tipi Başına Val R² Dağılımı', fontweight='bold')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    path = self.out / 'st12b_model_r2_boxplot.png'
                    fig.savefig(path, dpi=DPI, bbox_inches='tight')
                    plt.close(fig)
                    generated.append(str(path))
                    logger.info(f"  [OK] {path.name}")

        return generated

    def plot_band_analysis(self, pfaz12_dir: Path) -> list:
        """
        NuclearMomentBandAnalyzer ciktisinden bant grafikleri uretir.
        Kaynak: band_analysis/nuclear_band_analysis.xlsx
        Sheets: Bant_Ozeti, Korelasyon, Sicrama_Analizi, Capraz_Kutle
        """
        generated = []
        pfaz12_dir = Path(pfaz12_dir)

        band_excel_candidates = list(pfaz12_dir.rglob('nuclear_band_analysis.xlsx'))
        if not band_excel_candidates:
            logger.info("  [INFO] ST12-Band: nuclear_band_analysis.xlsx bulunamadi")
            return generated

        excel_path = band_excel_candidates[0]
        try:
            xl = pd.ExcelFile(excel_path)
        except Exception as e:
            logger.warning(f"  ST12-Band: Excel acilamadi: {e}")
            return generated

        # --- BA-1: Korelasyon bar chart (top-15 Spearman |r|) ---
        if 'Korelasyon' in xl.sheet_names:
            try:
                df_cor = pd.read_excel(xl, sheet_name='Korelasyon')
                targets_in_cor = df_cor['Target'].unique() if 'Target' in df_cor.columns else []
                if len(targets_in_cor) == 0 and not df_cor.empty:
                    targets_in_cor = ['all']

                fig_w = max(10, len(targets_in_cor) * 4)
                ncols = min(len(targets_in_cor), 4)
                nrows = (len(targets_in_cor) + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, 4 * nrows), squeeze=False)
                axes_flat = axes.flatten()

                for idx, target in enumerate(targets_in_cor):
                    ax = axes_flat[idx]
                    if 'Target' in df_cor.columns:
                        sub = df_cor[df_cor['Target'] == target]
                    else:
                        sub = df_cor
                    spear_col = next((c for c in sub.columns if 'spearman' in c.lower()), None)
                    feat_col  = next((c for c in sub.columns if 'feature' in c.lower()), None)
                    if spear_col is None or feat_col is None:
                        ax.set_visible(False)
                        continue
                    top = sub.nlargest(15, spear_col).sort_values(spear_col)
                    colors_bar = ['#EF5350' if v < 0 else '#42A5F5' for v in top[spear_col]]
                    ax.barh(range(len(top)), top[spear_col].values, color=colors_bar, alpha=0.85)
                    ax.set_yticks(range(len(top)))
                    ax.set_yticklabels(top[feat_col].values, fontsize=7)
                    ax.axvline(0, color='k', linewidth=0.8)
                    ax.set_xlabel('Spearman r')
                    ax.set_title(f'BA-1: {target} - Top-15 Korelasyon', fontsize=9, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='x')

                for idx in range(len(targets_in_cor), len(axes_flat)):
                    axes_flat[idx].set_visible(False)

                plt.suptitle('Moment Bandi - Ozellik Korelasyonlari (Spearman)', fontsize=12, fontweight='bold')
                plt.tight_layout()
                path = self.out / 'ba1_band_correlation.png'
                fig.savefig(path, dpi=DPI, bbox_inches='tight')
                plt.close(fig)
                generated.append(str(path))
                logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  BA-1 failed: {e}")

        # --- BA-2: Sicrama (jump) ozeti bar chart ---
        if 'Sicrama_Analizi' in xl.sheet_names:
            try:
                df_jump = pd.read_excel(xl, sheet_name='Sicrama_Analizi')
                chain_col = next((c for c in df_jump.columns if 'chain' in c.lower() or 'zincir' in c.lower()), None)
                mag_col   = next((c for c in df_jump.columns if 'magn' in c.lower() or 'buyukluk' in c.lower()), None)
                if chain_col and mag_col:
                    agg = df_jump.groupby(chain_col)[mag_col].agg(['count', 'mean']).reset_index()
                    agg.columns = ['Chain', 'Count', 'Mean_Mag']
                    fig, ax = plt.subplots(figsize=(8, 4))
                    x = np.arange(len(agg))
                    bars = ax.bar(x, agg['Count'], color='#FF7043', alpha=0.85, label='Sicrama Sayisi')
                    ax2 = ax.twinx()
                    ax2.plot(x, agg['Mean_Mag'], 'D-', color='#1A237E', markersize=6, label='Ort. Buyukluk')
                    ax.set_xticks(x)
                    ax.set_xticklabels(agg['Chain'], rotation=15)
                    ax.set_ylabel('Sicrama Sayisi', color='#FF7043')
                    ax2.set_ylabel('Ort. Sicrama Buyuklugu', color='#1A237E')
                    ax.set_title('BA-2: Zincir Turu Basina Ani Sicrama Analizi', fontweight='bold')
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    path = self.out / 'ba2_jump_summary.png'
                    fig.savefig(path, dpi=DPI, bbox_inches='tight')
                    plt.close(fig)
                    generated.append(str(path))
                    logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  BA-2 failed: {e}")

        # --- BA-3: Bant ozeti — cekirdek sayisi per band stacked by mass region ---
        if 'Bant_Ozeti' in xl.sheet_names:
            try:
                df_band = pd.read_excel(xl, sheet_name='Bant_Ozeti')
                band_col  = next((c for c in df_band.columns if 'band' in c.lower()), None)
                count_col = next((c for c in df_band.columns if 'count' in c.lower() or 'sayi' in c.lower()), None)
                light_col = next((c for c in df_band.columns if 'light' in c.lower() or 'hafif' in c.lower()), None)
                med_col   = next((c for c in df_band.columns if 'medium' in c.lower() or 'orta' in c.lower()), None)
                heavy_col = next((c for c in df_band.columns if 'heavy' in c.lower() or 'agir' in c.lower()), None)

                if band_col and count_col:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    x = np.arange(len(df_band))
                    bottom = np.zeros(len(df_band))
                    for col, label, color in [
                        (light_col, 'Hafif (A<100)',   '#81D4FA'),
                        (med_col,   'Orta (100-200)',  '#4DB6AC'),
                        (heavy_col, 'Agir (A>200)',    '#EF9A9A'),
                    ]:
                        if col and col in df_band.columns:
                            vals = df_band[col].fillna(0).values.astype(float)
                            ax.bar(x, vals, bottom=bottom, label=label, alpha=0.9)
                            bottom += vals

                    if light_col is None or light_col not in df_band.columns:
                        ax.bar(x, df_band[count_col].fillna(0).values, color='#90CAF9', alpha=0.85)

                    ax.set_xticks(x)
                    ax.set_xticklabels(df_band[band_col].values, rotation=20, ha='right', fontsize=8)
                    ax.set_ylabel('Cekirdek Sayisi')
                    ax.set_title('BA-3: Moment Bandi Basina Cekirdek Dagilimi (Kutle Bolgesine Gore)',
                                 fontweight='bold')
                    if light_col:
                        ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    path = self.out / 'ba3_band_nucleus_dist.png'
                    fig.savefig(path, dpi=DPI, bbox_inches='tight')
                    plt.close(fig)
                    generated.append(str(path))
                    logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  BA-3 failed: {e}")

        return generated


# =============================================================================
# AM13 — AutoML Improvements (PFAZ13)
# =============================================================================

class AutoML13Visualizer:
    """PFAZ13 AutoML optimizasyon ve iyileştirme grafikleri."""

    def __init__(self, output_dir: Path):
        self.out = Path(output_dir) / 'am13'
        self.out.mkdir(parents=True, exist_ok=True)

    def plot_automl_improvements(self, pfaz13_dir: Path) -> list:
        """
        PFAZ13 before/after Excel raporundan iyileştirme grafikleri.
        Kaynak: automl_improvement_report.xlsx
        """
        generated = []
        pfaz13_dir = Path(pfaz13_dir)

        # --- AM13-A: Before vs After R² karşılaştırma ---
        report_candidates = list(pfaz13_dir.rglob('automl_improvement_report.xlsx'))
        if report_candidates:
            try:
                df = pd.read_excel(report_candidates[0], sheet_name='Improvements')
                if not df.empty and 'Before_Val_R2' in df.columns and 'After_Val_R2' in df.columns:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                    # Scatter: before vs after
                    ax = axes[0]
                    targets_in_df = df['Target'].unique() if 'Target' in df.columns else ['all']
                    for target in targets_in_df:
                        sub = df[df['Target'] == target] if 'Target' in df.columns else df
                        ax.scatter(sub['Before_Val_R2'], sub['After_Val_R2'],
                                   c=COLORS.get(target, '#888'), label=target,
                                   alpha=0.7, s=40, edgecolors='k', linewidths=0.5)
                    # Diagonal line (no improvement)
                    lim = [min(df['Before_Val_R2'].min(), df['After_Val_R2'].min()) - 0.05,
                           max(df['Before_Val_R2'].max(), df['After_Val_R2'].max()) + 0.05]
                    ax.plot(lim, lim, 'k--', alpha=0.4, label='Değişim yok')
                    ax.set_xlabel('Öncesi Val R²')
                    ax.set_ylabel('Sonrası Val R²')
                    ax.set_title('AM13-A: AutoML Öncesi vs Sonrası R²', fontweight='bold')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                    # Bar: improvement per target
                    ax = axes[1]
                    if 'Target' in df.columns:
                        imp_by_target = df.groupby('Target').agg(
                            Delta_R2=('Delta_R2', 'mean'),
                            N=('Delta_R2', 'count')
                        ).reset_index()
                        bar_colors = [COLORS.get(t, '#888') for t in imp_by_target['Target']]
                        bars = ax.bar(imp_by_target['Target'], imp_by_target['Delta_R2'],
                                      color=bar_colors, alpha=0.85, edgecolor='k', linewidth=0.5)
                        ax.axhline(0, color='k', linewidth=0.8)
                        ax.set_ylabel('Ortalama ΔR² (Sonrası - Öncesi)')
                        ax.set_title('AM13-B: Hedef Başına Ortalama AutoML İyileştirme', fontweight='bold')
                        ax.grid(True, alpha=0.3, axis='y')
                        for bar, (_, row) in zip(bars, imp_by_target.iterrows()):
                            ax.text(bar.get_x() + bar.get_width() / 2,
                                    bar.get_height() + 0.003,
                                    f'+{row["Delta_R2"]:.3f}\n(n={int(row["N"])})',
                                    ha='center', va='bottom', fontsize=8)

                    plt.tight_layout()
                    path = self.out / 'am13a_automl_improvement.png'
                    fig.savefig(path, dpi=DPI, bbox_inches='tight')
                    plt.close(fig)
                    generated.append(str(path))
                    logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  AM13-A failed: {e}")

        # --- AM13-B: Optuna optimizasyon geçmişi (JSON'dan) ---
        json_files = list(pfaz13_dir.glob('*_automl.json'))
        if json_files:
            try:
                all_trials = {}
                for jf in json_files:
                    with open(jf, encoding='utf-8') as f:
                        data = json.load(f)
                    if 'trials_data' not in data:
                        continue
                    label = jf.stem.replace('_automl', '')
                    vals = [t.get('value') for t in data['trials_data']
                            if t.get('value') is not None and t.get('state') == 'COMPLETE']
                    if vals:
                        all_trials[label] = vals

                if all_trials:
                    n_plots = len(all_trials)
                    cols = min(3, n_plots)
                    rows = (n_plots + cols - 1) // cols
                    fig, axes = plt.subplots(rows, cols,
                                             figsize=(5 * cols, 3.5 * rows), squeeze=False)
                    axes_flat = axes.flatten()

                    for idx, (label, vals) in enumerate(all_trials.items()):
                        ax = axes_flat[idx]
                        cummax = np.maximum.accumulate(vals)
                        ax.plot(vals, 'o-', alpha=0.4, color='#90CAF9', markersize=3, label='Trial R²')
                        ax.plot(cummax, '-', color='#1565C0', linewidth=2, label='Best so far')
                        ax.axhline(cummax[-1], color='green', linestyle='--', alpha=0.5,
                                   label=f'Best={cummax[-1]:.3f}')
                        ax.set_title(label, fontsize=9, fontweight='bold')
                        ax.set_xlabel('Trial', fontsize=8)
                        ax.set_ylabel('Val R²', fontsize=8)
                        ax.legend(fontsize=7)
                        ax.grid(True, alpha=0.3)

                    # Hide unused axes
                    for idx in range(len(all_trials), len(axes_flat)):
                        axes_flat[idx].set_visible(False)

                    fig.suptitle('AM13-C: AutoML Optuna Optimizasyon Geçmişi', fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    path = self.out / 'am13c_optuna_history.png'
                    fig.savefig(path, dpi=DPI, bbox_inches='tight')
                    plt.close(fig)
                    generated.append(str(path))
                    logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  AM13-C failed: {e}")

        # --- AM13-D: Improved vs Not-improved model count ---
        if report_candidates:
            try:
                df = pd.read_excel(report_candidates[0], sheet_name='Improvements')
                if not df.empty and 'Delta_R2' in df.columns and 'Target' in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    improved     = df[df['Delta_R2'] > 0.01].groupby('Target').size()
                    not_improved = df[df['Delta_R2'] <= 0.01].groupby('Target').size()
                    all_targets  = sorted(set(list(improved.index) + list(not_improved.index)))
                    x = np.arange(len(all_targets))
                    w = 0.35
                    ax.bar(x - w/2, [improved.get(t, 0) for t in all_targets],
                           w, label='İyileşti (ΔR²>0.01)', color='#4CAF50', alpha=0.85)
                    ax.bar(x + w/2, [not_improved.get(t, 0) for t in all_targets],
                           w, label='Değişmedi/Kötüleşti', color='#EF9A9A', alpha=0.85)
                    ax.set_xticks(x)
                    ax.set_xticklabels(all_targets)
                    ax.set_ylabel('Model Sayısı')
                    ax.set_title('AM13-D: AutoML İyileştirme Başarısı', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    path = self.out / 'am13d_improvement_counts.png'
                    fig.savefig(path, dpi=DPI, bbox_inches='tight')
                    plt.close(fig)
                    generated.append(str(path))
                    logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  AM13-D failed: {e}")

        return generated

    def plot_automl_model_breakdown(self, pfaz13_dir: Path) -> list:
        """AM13-E/F: Model-tipi bazli AutoML dagilim + iyilesme hizi grafikleri."""
        generated = []
        pfaz13_dir = Path(pfaz13_dir)
        report_candidates = list(pfaz13_dir.rglob('automl_improvement_report.xlsx'))
        if not report_candidates:
            return generated

        try:
            df = pd.read_excel(report_candidates[0], sheet_name='Improvements')
        except Exception:
            return generated

        if df.empty or 'Model_Type' not in df.columns:
            return generated

        # --- AM13-E: Model-tipi bazli before/after violin ---
        try:
            model_types = sorted(df['Model_Type'].unique())
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            before_data = [df[df['Model_Type'] == m]['Before_Val_R2'].dropna().values for m in model_types]
            after_data  = [df[df['Model_Type'] == m]['After_Val_R2'].dropna().values  for m in model_types]

            for ax, data, label, color in [
                (axes[0], before_data, 'Oncesi Val R2',  '#90CAF9'),
                (axes[1], after_data,  'Sonrasi Val R2', '#A5D6A7'),
            ]:
                valid = [(d, m) for d, m in zip(data, model_types) if len(d) > 0]
                if valid:
                    vp = ax.violinplot([v[0] for v in valid], showmedians=True, showextrema=True)
                    for pc in vp['bodies']:
                        pc.set_facecolor(color)
                        pc.set_alpha(0.75)
                    ax.set_xticks(range(1, len(valid) + 1))
                    ax.set_xticklabels([v[1] for v in valid], rotation=20, ha='right', fontsize=8)
                    ax.set_ylabel('Val R2')
                    ax.set_title(f'AM13-E: {label} - Model Tipi Violin', fontweight='bold')
                    ax.axhline(0.9, color='g', linestyle='--', alpha=0.5, label='R2=0.9')
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.legend(fontsize=7)

            plt.tight_layout()
            path = self.out / 'am13e_model_type_violin.png'
            fig.savefig(path, dpi=DPI, bbox_inches='tight')
            plt.close(fig)
            generated.append(str(path))
            logger.info(f"  [OK] {path.name}")
        except Exception as e:
            logger.warning(f"  AM13-E failed: {e}")

        # --- AM13-F: Delta R2 heatmap (Model_Type x Target) ---
        if 'Target' in df.columns and 'Delta_R2' in df.columns:
            try:
                pivot = df.pivot_table(values='Delta_R2', index='Model_Type', columns='Target', aggfunc='mean')
                if not pivot.empty:
                    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2), max(4, len(pivot) * 0.7)))
                    import matplotlib.colors as mcolors
                    vmax = max(abs(pivot.values[~np.isnan(pivot.values)].min()),
                               abs(pivot.values[~np.isnan(pivot.values)].max())) if pivot.size > 0 else 0.1
                    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                                   vmin=-vmax, vmax=vmax)
                    ax.set_xticks(range(len(pivot.columns)))
                    ax.set_yticks(range(len(pivot.index)))
                    ax.set_xticklabels(pivot.columns, fontsize=9)
                    ax.set_yticklabels(pivot.index, fontsize=9)
                    plt.colorbar(im, ax=ax, label='Ort. Delta R2')
                    for i in range(len(pivot.index)):
                        for j in range(len(pivot.columns)):
                            v = pivot.values[i, j]
                            if not np.isnan(v):
                                ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                                        fontsize=8, color='white' if abs(v) > vmax * 0.5 else 'black')
                    ax.set_title('AM13-F: AutoML Delta R2 (Model Tipi x Hedef)', fontweight='bold')
                    plt.tight_layout()
                    path = self.out / 'am13f_delta_r2_heatmap.png'
                    fig.savefig(path, dpi=DPI, bbox_inches='tight')
                    plt.close(fig)
                    generated.append(str(path))
                    logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  AM13-F failed: {e}")

        return generated


class MonteCarlo9Extended:
    """MC9 genisletilmis grafikleri — Z-N haritasi ve hedef bazli hata dagilimi."""

    def __init__(self, output_dir: Path):
        self.out = Path(output_dir) / 'mc9'
        self.out.mkdir(parents=True, exist_ok=True)

    def plot_nucleus_error_map(self, pfaz4_dir: Path) -> list:
        """
        MC9-D: Z-N haritasinda renk = ortalama tahmin hatasi.
        MC9-E: Hedef basina hata histogram karsilastirmasi (AI vs ANFIS).
        Kaynak: PFAZ4 AAA2_Original_vs_Predictions.xlsx
        """
        generated = []
        pfaz4_dir = Path(pfaz4_dir)
        excel_candidates = list(pfaz4_dir.rglob('AAA2_Original_vs_Predictions.xlsx'))
        if not excel_candidates:
            return generated

        try:
            xl = pd.ExcelFile(excel_candidates[0])
        except Exception:
            return generated

        all_dfs = {}
        for sheet in xl.sheet_names:
            try:
                df = pd.read_excel(xl, sheet_name=sheet)
                if not df.empty:
                    all_dfs[sheet] = df
            except Exception:
                continue

        # --- MC9-D: Z-N scatter haritasi (renk = ort abs hata) ---
        for target, df in all_dfs.items():
            z_col = next((c for c in df.columns if c.upper() == 'Z'), None)
            n_col = next((c for c in df.columns if c.upper() == 'N'), None)
            err_cols = [c for c in df.columns if 'abs_error' in c.lower() or 'abserror' in c.lower()]
            orig_col = next((c for c in df.columns if 'original' in c.lower() or 'true' in c.lower()), None)
            if z_col is None or n_col is None or not err_cols:
                continue
            try:
                df = df.copy()
                df['_mean_err'] = df[err_cols].mean(axis=1)
                valid = df.dropna(subset=[z_col, n_col, '_mean_err'])
                if valid.empty:
                    continue
                fig, ax = plt.subplots(figsize=(10, 7))
                sc = ax.scatter(valid[n_col], valid[z_col],
                                c=valid['_mean_err'], cmap='YlOrRd',
                                s=50, edgecolors='k', linewidths=0.3, alpha=0.85)
                plt.colorbar(sc, ax=ax, label='Ort. Abs Hata')
                ax.set_xlabel('Notron Sayisi (N)')
                ax.set_ylabel('Proton Sayisi (Z)')
                ax.set_title(f'MC9-D: {target} — Z-N Haritasinda Tahmin Hatasi', fontweight='bold')
                ax.grid(True, alpha=0.2)
                plt.tight_layout()
                path = self.out / f'mc9d_zn_error_map_{target}.png'
                fig.savefig(path, dpi=DPI, bbox_inches='tight')
                plt.close(fig)
                generated.append(str(path))
                logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  MC9-D {target}: {e}")

        # --- MC9-E: Hata histogrami (tum hedefler icin) ---
        try:
            if all_dfs:
                n_tgts = len(all_dfs)
                fig, axes = plt.subplots(1, n_tgts, figsize=(5 * n_tgts, 4), squeeze=False)
                for ax, (target, df) in zip(axes[0], all_dfs.items()):
                    err_cols = [c for c in df.columns if 'abs_error' in c.lower()]
                    if not err_cols:
                        ax.set_visible(False)
                        continue
                    all_errs = df[err_cols].values.flatten()
                    all_errs = all_errs[~np.isnan(all_errs)]
                    ax.hist(all_errs, bins=30, color=COLORS.get(target, '#888'),
                            alpha=0.8, edgecolor='white', linewidth=0.4)
                    ax.axvline(np.median(all_errs), color='k', linestyle='--',
                               label=f'Medyan={np.median(all_errs):.3f}')
                    ax.set_xlabel('Abs Hata')
                    ax.set_ylabel('Frekans')
                    ax.set_title(f'MC9-E: {target} Hata Dagilimi', fontsize=9, fontweight='bold')
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3, axis='y')
                fig.suptitle('MC9-E: Tahmin Hatasi Histogram Karsilastirmasi', fontsize=11, fontweight='bold')
                plt.tight_layout()
                path = self.out / 'mc9e_error_histogram.png'
                fig.savefig(path, dpi=DPI, bbox_inches='tight')
                plt.close(fig)
                generated.append(str(path))
                logger.info(f"  [OK] {path.name}")
        except Exception as e:
            logger.warning(f"  MC9-E: {e}")

        # --- MC9-F: Gercek vs Tahmin scatter (hedef basina) ---
        for target, df in all_dfs.items():
            orig_col = next((c for c in df.columns if 'original' in c.lower() or
                             f'original_{target}' in c.lower()), None)
            pred_cols = [c for c in df.columns if 'y_pred' in c.lower() or '_pred_mean' in c.lower()]
            if orig_col is None or not pred_cols:
                continue
            try:
                df = df.dropna(subset=[orig_col])
                best_pred_col = pred_cols[0]
                valid = df[[orig_col, best_pred_col]].dropna()
                if len(valid) < 5:
                    continue
                from scipy import stats as _sp_stats
                slope, intercept, r, p, _ = _sp_stats.linregress(valid[orig_col], valid[best_pred_col])
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(valid[orig_col], valid[best_pred_col],
                           c=COLORS.get(target, '#888'), alpha=0.65, s=30,
                           edgecolors='k', linewidths=0.3)
                lim = [min(valid[orig_col].min(), valid[best_pred_col].min()) - 0.2,
                       max(valid[orig_col].max(), valid[best_pred_col].max()) + 0.2]
                ax.plot(lim, lim, 'k--', alpha=0.5, label='Ideal (y=x)')
                x_line = np.linspace(lim[0], lim[1], 100)
                ax.plot(x_line, slope * x_line + intercept, 'r-', alpha=0.7,
                        label=f'Regresyon (R={r:.3f})')
                ax.set_xlim(lim)
                ax.set_ylim(lim)
                ax.set_xlabel(f'Gercek {target}')
                ax.set_ylabel(f'Tahmin {target}')
                ax.set_title(f'MC9-F: {target} Gercek vs Tahmin', fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                path = self.out / f'mc9f_actual_vs_predicted_{target}.png'
                fig.savefig(path, dpi=DPI, bbox_inches='tight')
                plt.close(fig)
                generated.append(str(path))
                logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  MC9-F {target}: {e}")

        return generated


class Statistical12Extended:
    """ST12 genisletilmis: tum cekirdek dogruluk + Z-N band haritasi + model karsilastirma."""

    def __init__(self, output_dir: Path):
        self.out = Path(output_dir) / 'st12'
        self.out.mkdir(parents=True, exist_ok=True)

    def plot_all_nucleus_accuracy(self, pfaz4_dir: Path, pfaz12_dir: Path) -> list:
        """
        ST12-C: Tum 267 cekirdek icin sicrama vs normal vs bant dogruluk karsilastirmasi.
        ST12-D: Bant x Model_Tipi heatmap (ortalama abs hata).
        ST12-E: Z-N scatter renk = bant atamasi (band_idx).
        Kaynak: AAA2_Original_vs_Predictions.xlsx (PFAZ4) + nuclear_band_analysis.xlsx (PFAZ12)
        """
        generated = []
        pfaz4_dir  = Path(pfaz4_dir)
        pfaz12_dir = Path(pfaz12_dir)

        # Tahmin_Dogrulugu sayfasini oku (band_analyzer tarafindan uretildi)
        band_excel_cands = list(pfaz12_dir.rglob('nuclear_band_analysis.xlsx'))
        if not band_excel_cands:
            logger.info("  [INFO] ST12-C: nuclear_band_analysis.xlsx bulunamadi")
            return generated

        try:
            xl_band = pd.ExcelFile(band_excel_cands[0])
        except Exception:
            return generated

        # --- ST12-C: Sicrama vs Normal vs Bant dogruluk (box + bar) ---
        if 'Tahmin_Dogrulugu' in xl_band.sheet_names:
            try:
                df_acc = pd.read_excel(xl_band, sheet_name='Tahmin_Dogrulugu')
                sinif_col = next((c for c in df_acc.columns if 'sinif' in c.lower() or 'jump' in c.lower()), None)
                err_col   = next((c for c in df_acc.columns if 'ort_abs' in c.lower() or 'mean_abs' in c.lower()), None)
                tgt_col   = next((c for c in df_acc.columns if 'target' in c.lower()), None)

                if sinif_col and err_col and tgt_col:
                    targets = sorted(df_acc[tgt_col].dropna().unique())
                    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 5), squeeze=False)
                    for ax, tgt in zip(axes[0], targets):
                        sub = df_acc[df_acc[tgt_col] == tgt]
                        groups = sub.groupby(sinif_col)[err_col].apply(list)
                        colors_g = ['#EF5350' if 'sicrama' in k.lower() else '#42A5F5'
                                    for k in groups.index]
                        bp = ax.boxplot(groups.values, patch_artist=True, notch=False)
                        for patch, color in zip(bp['boxes'], colors_g):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        ax.set_xticklabels(groups.index, rotation=15, ha='right', fontsize=8)
                        ax.set_ylabel('Ort. Abs Hata')
                        ax.set_title(f'ST12-C: {tgt}\nSicrama vs Normal Tahmin Hatasi', fontsize=9, fontweight='bold')
                        ax.grid(True, alpha=0.3, axis='y')
                    fig.suptitle('Tum Cekirdekler — Sicrama Nukleusu Tahmin Dogrulugu', fontsize=11, fontweight='bold')
                    plt.tight_layout()
                    path = self.out / 'st12c_jump_vs_normal_accuracy.png'
                    fig.savefig(path, dpi=DPI, bbox_inches='tight')
                    plt.close(fig)
                    generated.append(str(path))
                    logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  ST12-C: {e}")

        # --- ST12-D: Ozet — bant bazli hata heatmap (Band x Target) ---
        if 'Tahmin_Ozeti' in xl_band.sheet_names:
            try:
                df_sum = pd.read_excel(xl_band, sheet_name='Tahmin_Ozeti')
                band_col = next((c for c in df_sum.columns if 'sinif' in c.lower() or 'band' in c.lower()), None)
                err_col  = next((c for c in df_sum.columns if 'ort_abs' in c.lower() or 'mean' in c.lower()), None)
                tgt_col  = next((c for c in df_sum.columns if 'target' in c.lower()), None)
                if band_col and err_col and tgt_col:
                    pivot = df_sum[df_sum[band_col].str.startswith('Bant:')].pivot_table(
                        index=band_col, columns=tgt_col, values=err_col, aggfunc='mean')
                    if not pivot.empty:
                        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2.5), max(4, len(pivot) * 0.6)))
                        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
                        ax.set_xticks(range(len(pivot.columns)))
                        ax.set_yticks(range(len(pivot.index)))
                        ax.set_xticklabels(pivot.columns, fontsize=9)
                        ax.set_yticklabels([str(b)[:20] for b in pivot.index], fontsize=7)
                        plt.colorbar(im, ax=ax, label='Ort. Abs Hata')
                        for i in range(len(pivot.index)):
                            for j in range(len(pivot.columns)):
                                v = pivot.values[i, j]
                                if not np.isnan(v):
                                    ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                                            fontsize=7, color='white' if v > pivot.values[~np.isnan(pivot.values)].mean() else 'black')
                        ax.set_title('ST12-D: Bant x Hedef Tahmin Hatasi Heatmap', fontweight='bold')
                        plt.tight_layout()
                        path = self.out / 'st12d_band_accuracy_heatmap.png'
                        fig.savefig(path, dpi=DPI, bbox_inches='tight')
                        plt.close(fig)
                        generated.append(str(path))
                        logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  ST12-D: {e}")

        # --- ST12-E: Z-N scatter renk = bant indeksi ---
        if 'Cekirdek_Detay' in xl_band.sheet_names:
            try:
                df_det = pd.read_excel(xl_band, sheet_name='Cekirdek_Detay')
                z_col   = next((c for c in df_det.columns if c.upper() == 'Z'), None)
                n_col   = next((c for c in df_det.columns if c.upper() == 'N'), None)
                bi_col  = next((c for c in df_det.columns if 'band_idx' in c.lower()), None)
                bl_col  = next((c for c in df_det.columns if 'band_label' in c.lower()), None)
                tgt_col = next((c for c in df_det.columns if 'target' in c.lower()), None)

                if z_col and n_col and bi_col and tgt_col:
                    targets = sorted(df_det[tgt_col].dropna().unique())
                    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 6), squeeze=False)
                    for ax, tgt in zip(axes[0], targets):
                        sub = df_det[df_det[tgt_col] == tgt].dropna(subset=[z_col, n_col, bi_col])
                        if sub.empty:
                            ax.set_visible(False)
                            continue
                        sc = ax.scatter(sub[n_col], sub[z_col],
                                        c=sub[bi_col], cmap='tab10',
                                        s=55, edgecolors='k', linewidths=0.3, alpha=0.85)
                        plt.colorbar(sc, ax=ax, label='Bant Indeksi')
                        ax.set_xlabel('N (Notron)')
                        ax.set_ylabel('Z (Proton)')
                        ax.set_title(f'ST12-E: {tgt} — Z-N Haritasinda Bant Atamasi', fontsize=9, fontweight='bold')
                        ax.grid(True, alpha=0.2)
                    fig.suptitle('Cekirdeklerin Moment Bant Atamasi (Z-N Haritasi)', fontsize=11, fontweight='bold')
                    plt.tight_layout()
                    path = self.out / 'st12e_zn_band_map.png'
                    fig.savefig(path, dpi=DPI, bbox_inches='tight')
                    plt.close(fig)
                    generated.append(str(path))
                    logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  ST12-E: {e}")

        # --- ST12-F: AI vs ANFIS Val R2 hedef bazli violin karsilastirma ---
        ai_dir_cands = [pfaz12_dir.parent / 'trained_models',
                        pfaz12_dir.parent.parent / 'outputs' / 'trained_models']
        anfis_dir_cands = [pfaz12_dir.parent / 'anfis_models',
                           pfaz12_dir.parent.parent / 'outputs' / 'anfis_models']
        ai_dir    = next((p for p in ai_dir_cands    if p.exists()), None)
        anfis_dir = next((p for p in anfis_dir_cands if p.exists()), None)

        try:
            ai_scores_by_tgt    = {}
            anfis_scores_by_tgt = {}
            if ai_dir:
                for mf in ai_dir.rglob('metrics_*.json'):
                    try:
                        with open(mf, encoding='utf-8') as f:
                            m = json.load(f)
                        tgt = m.get('target', mf.parts[-4] if len(mf.parts) >= 4 else 'all')
                        r2  = m.get('val', {}).get('r2', None)
                        if r2 is not None and not np.isnan(r2) and r2 > -10:
                            ai_scores_by_tgt.setdefault(tgt, []).append(float(r2))
                    except Exception:
                        continue
            if anfis_dir:
                for mf in anfis_dir.rglob('metrics_*.json'):
                    try:
                        with open(mf, encoding='utf-8') as f:
                            m = json.load(f)
                        tgt = m.get('target', 'all')
                        r2  = m.get('val', {}).get('r2', None)
                        if r2 is not None and not np.isnan(r2) and r2 > -10:
                            anfis_scores_by_tgt.setdefault(tgt, []).append(float(r2))
                    except Exception:
                        continue

            all_tgts = sorted(set(list(ai_scores_by_tgt.keys()) + list(anfis_scores_by_tgt.keys())))
            if all_tgts:
                fig, ax = plt.subplots(figsize=(max(10, len(all_tgts) * 2.5), 5))
                x_base = np.arange(len(all_tgts))
                w = 0.35
                for offset, scores_dict, label, color in [
                    (-w/2, ai_scores_by_tgt,    'AI',    '#2196F3'),
                    (+w/2, anfis_scores_by_tgt, 'ANFIS', '#FF9800'),
                ]:
                    means = [np.mean(scores_dict.get(t, [np.nan])) for t in all_tgts]
                    stds  = [np.std(scores_dict.get(t,  [0.0]))    for t in all_tgts]
                    ax.bar(x_base + offset, means, w, yerr=stds, label=label,
                           color=color, alpha=0.8, capsize=4, edgecolor='k', linewidth=0.4)
                ax.set_xticks(x_base)
                ax.set_xticklabels(all_tgts)
                ax.set_ylabel('Ortalama Val R2 (+-std)')
                ax.set_title('ST12-F: AI vs ANFIS — Hedef Bazli Val R2 Karsilastirma', fontweight='bold')
                ax.axhline(0.9, color='g', linestyle='--', alpha=0.5, label='R2=0.9 (iyi)')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                path = self.out / 'st12f_ai_vs_anfis_r2.png'
                fig.savefig(path, dpi=DPI, bbox_inches='tight')
                plt.close(fig)
                generated.append(str(path))
                logger.info(f"  [OK] {path.name}")
        except Exception as e:
            logger.warning(f"  ST12-F: {e}")

        return generated

    def plot_nuclear_correlation_matrix(self, pfaz12_dir: Path, aaa2_path: Path = None) -> list:
        """
        ST12-G: Fizik ozellikleri korelasyon heatmap (aaa2.txt'den).
        ST12-H: Hedef basina izotop zinciri moment profili.
        """
        generated = []
        pfaz12_dir = Path(pfaz12_dir)

        # aaa2 yolunu bul
        if aaa2_path is None or not Path(aaa2_path).exists():
            for cand in [pfaz12_dir.parent.parent / 'aaa2.txt',
                         pfaz12_dir.parent.parent / 'data' / 'aaa2.txt',
                         pfaz12_dir.parent.parent.parent / 'aaa2.txt']:
                if cand.exists():
                    aaa2_path = cand
                    break

        if aaa2_path is None or not Path(aaa2_path).exists():
            return generated

        try:
            df = pd.read_csv(aaa2_path, sep='\t', encoding='utf-8')
            if 'NUCLEUS' not in df.columns:
                df = pd.read_csv(aaa2_path, encoding='utf-8')
        except Exception as e:
            logger.warning(f"  ST12-G: aaa2.txt okunamadi: {e}")
            return generated

        PHYSICS_COLS = ['Z', 'N', 'A', 'SPIN', 'MAGNETIC MOMENT [µ]',
                        'QUADRUPOLE MOMENT [Q]', 'Beta_2']
        extra_cols = [c for c in df.columns if any(k in c.lower() for k in
                      ['semf', 'woods', 'shell', 'nilsson', 'binding', 'magic',
                       'valence', 'deform', 'radius', 'pair'])]
        cols_use = [c for c in PHYSICS_COLS + extra_cols if c in df.columns]
        num_df = df[cols_use].select_dtypes(include=[np.number]).dropna(how='all')

        # --- ST12-G: Korelasyon heatmap ---
        if len(num_df.columns) >= 3:
            try:
                from scipy import stats as _sp
                corr_mat = num_df.corr(method='spearman')
                fig, ax = plt.subplots(figsize=(max(8, len(corr_mat) * 0.7),
                                                max(7, len(corr_mat) * 0.6)))
                im = ax.imshow(corr_mat.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                ax.set_xticks(range(len(corr_mat.columns)))
                ax.set_yticks(range(len(corr_mat.index)))
                ax.set_xticklabels(corr_mat.columns, rotation=45, ha='right', fontsize=7)
                ax.set_yticklabels(corr_mat.index, fontsize=7)
                plt.colorbar(im, ax=ax, label='Spearman r', shrink=0.8)
                ax.set_title('ST12-G: Nuklear Ozellik Korelasyon Matrisi (Spearman)',
                             fontweight='bold')
                plt.tight_layout()
                path = self.out / 'st12g_feature_correlation_matrix.png'
                fig.savefig(path, dpi=DPI, bbox_inches='tight')
                plt.close(fig)
                generated.append(str(path))
                logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  ST12-G: {e}")

        # --- ST12-H: Izotop zinciri moment profili (Z=50 gibi; en cok N'e sahip Z) ---
        mm_col = next((c for c in df.columns if 'magnetic' in c.lower() or c == 'MM'), None)
        z_col  = next((c for c in df.columns if c.upper() == 'Z'), None)
        n_col  = next((c for c in df.columns if c.upper() == 'N'), None)
        if mm_col and z_col and n_col:
            try:
                top_z_vals = df.groupby(z_col).size().nlargest(3).index.tolist()
                if top_z_vals:
                    fig, axes = plt.subplots(1, len(top_z_vals),
                                             figsize=(5 * len(top_z_vals), 4), squeeze=False)
                    for ax, z_val in zip(axes[0], top_z_vals):
                        chain = df[df[z_col] == z_val].dropna(subset=[n_col, mm_col])
                        chain = chain.sort_values(n_col)
                        if len(chain) < 2:
                            ax.set_visible(False)
                            continue
                        ax.plot(chain[n_col], chain[mm_col], 'o-',
                                color='#1565C0', markersize=6, linewidth=1.5)
                        for _, row in chain.iterrows():
                            nuc = row.get('NUCLEUS', f"Z{z_val}N{int(row[n_col])}")
                            ax.annotate(str(nuc), (row[n_col], row[mm_col]),
                                        textcoords='offset points', xytext=(0, 5), fontsize=6)
                        ax.set_xlabel('N (Notron Sayisi)')
                        ax.set_ylabel('MM (Manyetik Moment)')
                        ax.set_title(f'ST12-H: Z={z_val} Izotop Zinciri MM Profili',
                                     fontsize=9, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                    fig.suptitle('Izotop Zincirlerinde Manyetik Moment Degisimi',
                                 fontsize=11, fontweight='bold')
                    plt.tight_layout()
                    path = self.out / 'st12h_isotope_mm_profile.png'
                    fig.savefig(path, dpi=DPI, bbox_inches='tight')
                    plt.close(fig)
                    generated.append(str(path))
                    logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  ST12-H: {e}")

        return generated


# =============================================================================
# SUPPLEMENTAL VISUALIZATION RUNNER
# =============================================================================

class SupplementalVisualizer:
    """
    PFAZ8 ikinci geçiş — PFAZ9/12/13 verilerinden grafik üretir.
    main.py'de PFAZ13 tamamlandıktan sonra çağrılır.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.mc9_viz    = MonteCarlо9Visualizer(self.output_dir)
        self.mc9_ext    = MonteCarlo9Extended(self.output_dir)
        self.st12_viz   = Statistical12Visualizer(self.output_dir)
        self.st12_ext   = Statistical12Extended(self.output_dir)
        self.am13_viz   = AutoML13Visualizer(self.output_dir)

    def run(self, pfaz9_dir=None, pfaz12_dir=None, pfaz13_dir=None,
            pfaz4_dir=None, aaa2_path=None) -> dict:
        """
        Tum supplemental grafikleri uret.

        Args:
            pfaz9_dir:  PFAZ9 cikti dizini
            pfaz12_dir: PFAZ12 cikti dizini
            pfaz13_dir: PFAZ13 cikti dizini
            pfaz4_dir:  PFAZ4 cikti dizini (AAA2_Original_vs_Predictions.xlsx)
            aaa2_path:  aaa2.txt dosya yolu (korelasyon grafikleri icin)
        """
        logger.info("\n" + "=" * 70)
        logger.info("PFAZ8 SUPPLEMENTAL VISUALIZATIONS (PFAZ4/9/12/13)")
        logger.info("=" * 70)

        all_files = []

        # PFAZ9 — Monte Carlo (temel + genisletilmis)
        if pfaz9_dir and Path(pfaz9_dir).exists():
            logger.info("\n[MC9] Monte Carlo grafikleri...")
            for method, label in [
                (lambda: self.mc9_viz.plot_uncertainty_distribution(Path(pfaz9_dir)), "MC9 temel"),
            ]:
                try:
                    files = method()
                    all_files.extend(files)
                    logger.info(f"  [OK] {len(files)} {label} grafigi")
                except Exception as e:
                    logger.warning(f"  [{label} ERROR] {e}")

        # PFAZ4 dir bul (pfaz4_dir verilmediyse varsayilan)
        _pfaz4 = pfaz4_dir
        if _pfaz4 is None:
            for cand in [Path(self.output_dir).parent.parent / 'outputs' / 'unknown_predictions',
                         Path(self.output_dir).parent / 'unknown_predictions']:
                if cand.exists():
                    _pfaz4 = cand
                    break

        if _pfaz4 and Path(_pfaz4).exists():
            logger.info("\n[MC9-EXT] Genisletilmis hata ve Z-N haritasi grafikleri...")
            try:
                files = self.mc9_ext.plot_nucleus_error_map(Path(_pfaz4))
                all_files.extend(files)
                logger.info(f"  [OK] {len(files)} MC9-EXT grafigi")
            except Exception as e:
                logger.warning(f"  [MC9-EXT ERROR] {e}")
        else:
            logger.info("  [INFO] PFAZ4 dizini bulunamadi -- MC9 genisletilmis grafikleri atlandi")

        # PFAZ12 — Statistical Tests + Band Analysis + Extended
        if pfaz12_dir and Path(pfaz12_dir).exists():
            logger.info("\n[ST12] Istatistiksel test grafikleri...")
            try:
                files = self.st12_viz.plot_statistical_tests(Path(pfaz12_dir))
                all_files.extend(files)
                logger.info(f"  [OK] {len(files)} ST12 temel grafigi")
            except Exception as e:
                logger.warning(f"  [ST12 ERROR] {e}")

            logger.info("\n[BA] Moment bant analizi grafikleri...")
            try:
                files = self.st12_viz.plot_band_analysis(Path(pfaz12_dir))
                all_files.extend(files)
                logger.info(f"  [OK] {len(files)} BA grafigi")
            except Exception as e:
                logger.warning(f"  [BA ERROR] {e}")

            logger.info("\n[ST12-EXT] Tum cekirdek dogruluk + Z-N band haritasi + AI vs ANFIS...")
            try:
                files = self.st12_ext.plot_all_nucleus_accuracy(
                    pfaz4_dir=Path(_pfaz4) if _pfaz4 else Path(pfaz12_dir),
                    pfaz12_dir=Path(pfaz12_dir)
                )
                all_files.extend(files)
                logger.info(f"  [OK] {len(files)} ST12-EXT grafigi")
            except Exception as e:
                logger.warning(f"  [ST12-EXT ERROR] {e}")

            logger.info("\n[ST12-COR] Nuklear korelasyon matrisi + izotop profili...")
            try:
                files = self.st12_ext.plot_nuclear_correlation_matrix(
                    pfaz12_dir=Path(pfaz12_dir),
                    aaa2_path=Path(aaa2_path) if aaa2_path else None
                )
                all_files.extend(files)
                logger.info(f"  [OK] {len(files)} ST12-COR grafigi")
            except Exception as e:
                logger.warning(f"  [ST12-COR ERROR] {e}")
        else:
            logger.info("  [INFO] PFAZ12 dizini bulunamadi -- istatistik/band grafikleri atlandi")

        # PFAZ13 — AutoML (temel + genisletilmis)
        if pfaz13_dir and Path(pfaz13_dir).exists():
            logger.info("\n[AM13] AutoML grafikleri...")
            try:
                files = self.am13_viz.plot_automl_improvements(Path(pfaz13_dir))
                all_files.extend(files)
                logger.info(f"  [OK] {len(files)} AM13 temel grafigi")
            except Exception as e:
                logger.warning(f"  [AM13 ERROR] {e}")

            logger.info("\n[AM13-EXT] Model tipi violin + delta heatmap...")
            try:
                files = self.am13_viz.plot_automl_model_breakdown(Path(pfaz13_dir))
                all_files.extend(files)
                logger.info(f"  [OK] {len(files)} AM13-EXT grafigi")
            except Exception as e:
                logger.warning(f"  [AM13-EXT ERROR] {e}")
        else:
            logger.info("  [INFO] PFAZ13 dizini bulunamadi -- AutoML grafikleri atlandi")

        logger.info(f"\n[OK] Supplemental viz tamamlandi: {len(all_files)} grafik")
        logger.info(f"     Dizin: {self.output_dir}")

        return {
            'n_files': len(all_files),
            'files': all_files,
            'output_dir': str(self.output_dir),
        }
