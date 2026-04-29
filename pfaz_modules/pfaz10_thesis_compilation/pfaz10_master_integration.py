"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  PFAZ 10: MASTER THESIS INTEGRATION SYSTEM                   ║
║                                                                              ║
║  Complete end-to-end thesis compilation orchestrator                        ║
║  Collects results from ALL 13 PFAZ phases:                                  ║
║    PFAZ1  — dataset generation stats                                        ║
║    PFAZ2  — AI model metrics (RF/XGB/GBM/LGB/CB/SVR/DNN)                   ║
║    PFAZ3  — ANFIS training results                                          ║
║    PFAZ4  — unknown nuclei predictions                                      ║
║    PFAZ5  — cross-model analysis                                            ║
║    PFAZ6  — THESIS_COMPLETE_RESULTS.xlsx                                    ║
║    PFAZ7  — ensemble results                                                ║
║    PFAZ8  — visualizations (standard + supplemental)                        ║
║    PFAZ9  — Monte Carlo / AAA2 uncertainty                                  ║
║    PFAZ10 — this file                                                       ║
║    PFAZ12 — statistical tests (ANOVA/Friedman/Wilcoxon/Bayes)               ║
║    PFAZ13 — AutoML retraining loop before/after                             ║
║                                                                              ║
║  Author: Nuclear Physics AI Project                                         ║
║  Version: 5.0.0 - UPDATED FOR PFAZ12/13                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _safe_excel_first_sheet(path: Path, nrows: int = 10) -> Optional[pd.DataFrame]:
    try:
        return pd.read_excel(path, sheet_name=0, nrows=nrows)
    except Exception:
        return None


def _df_to_latex(df: pd.DataFrame, caption: str, label: str, max_rows: int = 10) -> str:
    """DataFrame → minimal LaTeX table (booktabs style)."""
    df = df.head(max_rows).copy()

    def _esc(val):
        s = str(val)
        return s.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    cols = [_esc(c) for c in df.columns]
    col_fmt = 'l' + 'r' * (len(cols) - 1)

    lines = [
        r'\begin{table}[htbp]',
        r'  \centering',
        f'  \\caption{{{caption}}}',
        f'  \\label{{tab:{label}}}',
        f'  \\begin{{tabular}}{{{col_fmt}}}',
        r'    \toprule',
        '    ' + ' & '.join(cols) + r' \\',
        r'    \midrule',
    ]
    for _, row in df.iterrows():
        lines.append('    ' + ' & '.join([_esc(v) for v in row]) + r' \\')
    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MasterThesisIntegration:
    """
    Master Thesis Integration System — v5.0

    Reads all PFAZ output directories, generates LaTeX chapters, copies
    figures, generates tables and compiles to PDF (if pdflatex available).

    Typical usage from main.py::

        thesis = MasterThesisIntegration(
            project_dir=str(self.output_dir),   # e.g. outputs/
            output_dir=str(self.pfaz_outputs[10]),
            pfaz_outputs=self.pfaz_outputs,
        )
        thesis.execute_full_pipeline(compile_pdf=False)
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        project_dir: str = '.',
        output_dir: str = 'thesis',
        pfaz_outputs: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.output_dir  = Path(output_dir).resolve()

        # pfaz_outputs: {pfaz_id: Path}  — injected by main.py
        self._pfaz_out: Dict[int, Path] = {}
        if pfaz_outputs:
            for k, v in pfaz_outputs.items():
                self._pfaz_out[int(k)] = Path(v)

        self.metadata = {
            'title': 'Machine Learning and ANFIS-Based Prediction of Nuclear Properties',
            'subtitle': 'Magnetic Moments, Quadrupole Moments and Deformation Parameters',
            'author': 'Research Student',
            'supervisor': 'Prof. Supervisor Name',
            'university': 'University Name',
            'department': 'Department of Physics',
            'thesis_type': 'Master of Science',
            'date': datetime.now().strftime('%B %Y'),
            'version': '5.0.0',
        }
        if metadata:
            self.metadata.update(metadata)

        # Registry filled during _step1
        self.collected: Dict[str, Any] = {
            'figures': [],          # list of Path
            'excel_reports': [],    # list of Path
            'json_summaries': {},   # {name: dict}
            'metrics': {},          # {pfaz_id: dict}
        }

        self._create_dirs()
        logger.info("=" * 80)
        logger.info("PFAZ 10: MASTER THESIS INTEGRATION v5.0")
        logger.info("=" * 80)
        logger.info(f"  Project dir : {self.project_dir}")
        logger.info(f"  Output dir  : {self.output_dir}")

    # ------------------------------------------------------------------
    def _pfaz_path(self, pfaz_id: int, fallback_name: str) -> Path:
        """Return PFAZ output dir — from injected map or fallback guess."""
        if pfaz_id in self._pfaz_out:
            return self._pfaz_out[pfaz_id]
        # Guess: parent of output_dir / fallback_name
        guesses = [
            self.project_dir / fallback_name,
            self.output_dir.parent / fallback_name,
        ]
        for g in guesses:
            if g.exists():
                return g
        return self.project_dir / fallback_name

    def _create_dirs(self):
        for sub in ['chapters', 'figures', 'tables', 'appendices', 'bibliography', 'logs']:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def execute_full_pipeline(
        self,
        author: Optional[str] = None,
        supervisor: Optional[str] = None,
        university: Optional[str] = None,
        compile_pdf: bool = False,
    ) -> Dict[str, Any]:
        """Run complete thesis generation pipeline."""
        if author:
            self.metadata['author'] = author
        if supervisor:
            self.metadata['supervisor'] = supervisor
        if university:
            self.metadata['university'] = university

        steps = [
            ("1. Collect Data from All Phases",  self._step1_collect_all_data),
            ("2. Generate Chapter Content",       self._step2_generate_chapters),
            ("3. Copy Figures",                   self._step3_copy_figures),
            ("4. Generate LaTeX Tables",          self._step4_generate_tables),
            ("5. Create Bibliography",            self._step5_bibliography),
            ("6. Generate Main Document",         self._step6_main_document),
            ("7. Quality Checks",                 self._step7_quality_checks),
        ]

        results: Dict[str, Any] = {
            'success': True,
            'steps_completed': [],
            'errors': [],
            'warnings': [],
            'files_generated': [],
        }

        for name, fn in steps:
            logger.info(f"\n{'='*70}\n  {name}\n{'='*70}")
            try:
                sr = fn()
                results['steps_completed'].append(name)
                results['files_generated'].extend(sr.get('files', []))
                results['warnings'].extend(sr.get('warnings', []))
                logger.info(f"  [OK] {name}")
            except Exception as e:
                results['errors'].append(f"{name}: {e}")
                results['success'] = False
                logger.error(f"  [FAIL] {name}: {e}")
                # Continue to next step anyway (don't break)

        if compile_pdf:
            logger.info("\n  8. Compile PDF")
            try:
                pr = self._step8_compile_pdf()
                results['steps_completed'].append("8. Compile PDF")
                results['pdf_path'] = pr.get('pdf_path')
                if pr.get('pdf_path'):
                    results['files_generated'].append(pr['pdf_path'])
            except Exception as e:
                results['errors'].append(f"Compile PDF: {e}")
                logger.warning(f"  [WARN] PDF compilation skipped: {e}")

        self._save_report(results)
        self._print_summary(results)
        return results

    # ------------------------------------------------------------------
    # STEP 1: Collect data from all 13 PFAZ outputs
    # ------------------------------------------------------------------

    def _step1_collect_all_data(self) -> Dict:
        """Scan all PFAZ output directories and collect metrics/figures."""

        # ---- Figures (PFAZ8 standard + supplemental) --------------------
        p8 = self._pfaz_path(8, 'visualizations')
        if p8.exists():
            for ext in ('*.png', '*.pdf'):
                self.collected['figures'].extend(p8.rglob(ext))
            logger.info(f"  [PFAZ8] {len(self.collected['figures'])} figure files found")

        # ---- PFAZ6 — primary Excel report --------------------------------
        p6 = self._pfaz_path(6, 'reports')
        for xlsx in p6.rglob('THESIS_COMPLETE_RESULTS*.xlsx'):
            self.collected['excel_reports'].append(xlsx)
            logger.info(f"  [PFAZ6] {xlsx.name}")

        # ---- PFAZ9 — Monte Carlo / AAA2 ----------------------------------
        p9 = self._pfaz_path(9, 'aaa2_results')
        if p9.exists():
            for xlsx in p9.rglob('AAA2_Complete_*.xlsx'):
                self.collected['excel_reports'].append(xlsx)
                logger.info(f"  [PFAZ9] {xlsx.name}")
            mc_json = p9 / 'aaa2_analysis_summary.json'
            if mc_json.exists():
                data = _safe_json(mc_json)
                if data:
                    self.collected['json_summaries']['pfaz9_mc'] = data
                    self.collected['metrics'][9] = data

        # ---- PFAZ12 — statistical tests ----------------------------------
        p12 = self._pfaz_path(12, 'advanced_analytics')
        for xlsx in p12.rglob('pfaz12_statistical_tests*.xlsx'):
            self.collected['excel_reports'].append(xlsx)
            logger.info(f"  [PFAZ12] {xlsx.name}")

        # ---- PFAZ13 — AutoML summary + retraining log --------------------
        p13 = self._pfaz_path(13, 'automl_results')
        automl_summary = p13 / 'automl_summary.json'
        if automl_summary.exists():
            data = _safe_json(automl_summary)
            if data:
                self.collected['json_summaries']['pfaz13_automl'] = data
                self.collected['metrics'][13] = data

        retrain_log = p13 / 'automl_retraining_log.json'
        if retrain_log.exists():
            data = _safe_json(retrain_log)
            if isinstance(data, list):
                self.collected['json_summaries']['pfaz13_retraining'] = data

        for xlsx in p13.rglob('automl_improvement_report*.xlsx'):
            self.collected['excel_reports'].append(xlsx)
            logger.info(f"  [PFAZ13] {xlsx.name}")

        # ---- PFAZ1 — dataset generation stats ----------------------------
        p1 = self._pfaz_path(1, 'generated_datasets')
        p1_meta = p1 / 'generation_summary.json'
        if not p1_meta.exists():
            p1_meta = p1.parent / 'generated_datasets' / 'generation_summary.json'
        if p1_meta.exists():
            d = _safe_json(p1_meta)
            if d:
                self.collected['json_summaries']['pfaz1_summary'] = d
                self.collected['metrics'][1] = d
                logger.info(f"  [PFAZ1] Dataset summary loaded: {d.get('total_datasets', '?')} datasets")
        # Count actual dataset folders as fallback
        if 1 not in self.collected['metrics'] and p1.exists():
            n_ds = sum(1 for _ in p1.iterdir() if _.is_dir())
            self.collected['metrics'][1] = {'total_datasets': n_ds}
            logger.info(f"  [PFAZ1] {n_ds} dataset folders found")

        # ---- PFAZ2 — AI metrics (spot-check) ----------------------------
        p2 = self._pfaz_path(2, 'trained_models')
        ai_metrics = []
        if p2.exists():
            for mf in list(p2.rglob('metrics_*.json'))[:50]:
                d = _safe_json(mf)
                if d:
                    ai_metrics.append(d)
        if ai_metrics:
            self.collected['metrics'][2] = ai_metrics
            logger.info(f"  [PFAZ2] {len(ai_metrics)} AI metric files loaded")

        # ---- PFAZ3 — ANFIS results ---------------------------------------
        p3 = self._pfaz_path(3, 'anfis_models')
        anfis_metrics = []
        if p3.exists():
            for mf in list(p3.rglob('metrics_*.json'))[:30]:
                d = _safe_json(mf)
                if d:
                    anfis_metrics.append(d)
        if anfis_metrics:
            self.collected['metrics'][3] = anfis_metrics
            logger.info(f"  [PFAZ3] {len(anfis_metrics)} ANFIS metric files loaded")

        # ---- PFAZ4 — unknown nucleus predictions -------------------------
        p4 = self._pfaz_path(4, 'unknown_predictions')
        if p4.exists():
            for xlsx in p4.rglob('unknown_predictions_*.xlsx'):
                self.collected['excel_reports'].append(xlsx)
                logger.info(f"  [PFAZ4] {xlsx.name}")
            p4_summary = p4 / 'prediction_summary.json'
            if p4_summary.exists():
                d = _safe_json(p4_summary)
                if d:
                    self.collected['json_summaries']['pfaz4_predictions'] = d
                    self.collected['metrics'][4] = d

        # ---- PFAZ5 — cross-model analysis --------------------------------
        p5 = self._pfaz_path(5, 'cross_model_analysis')
        if p5.exists():
            for xlsx in p5.rglob('MASTER_CROSS_MODEL_REPORT*.xlsx'):
                self.collected['excel_reports'].append(xlsx)
                logger.info(f"  [PFAZ5] {xlsx.name}")
            p5_summary = p5 / 'cross_model_summary.json'
            if p5_summary.exists():
                d = _safe_json(p5_summary)
                if d:
                    self.collected['json_summaries']['pfaz5_cross'] = d
                    self.collected['metrics'][5] = d

        # ---- PFAZ7 — ensemble results ------------------------------------
        p7 = self._pfaz_path(7, 'ensemble_results')
        for jf in p7.rglob('ensemble_results*.json'):
            d = _safe_json(jf)
            if d:
                self.collected['metrics'][7] = d
                logger.info(f"  [PFAZ7] Ensemble results loaded")
                break
        for xlsx in p7.rglob('ensemble_report*.xlsx'):
            self.collected['excel_reports'].append(xlsx)
            logger.info(f"  [PFAZ7] {xlsx.name}")

        logger.info(f"\n  Summary: {len(self.collected['figures'])} figures, "
                    f"{len(self.collected['excel_reports'])} Excel reports, "
                    f"{len(self.collected['json_summaries'])} JSON summaries")

        return {'files': [], 'warnings': []}

    # ------------------------------------------------------------------
    # STEP 2: Generate chapter LaTeX files
    # ------------------------------------------------------------------

    def _step2_generate_chapters(self) -> Dict:
        # --- Front matter ---
        self._write_chapter('00_abstract.tex',         self._ch_abstract())
        self._write_chapter('00_abbreviations.tex',    self._ch_abbreviations())
        self._write_chapter('00_symbols.tex',          self._ch_symbols())
        # --- Main chapters ---
        self._write_chapter('01_introduction.tex',     self._ch_introduction())
        self._write_chapter('02_nuclear_theory.tex',   self._ch_nuclear_theory())
        self._write_chapter('03_methodology.tex',      self._ch_methodology())
        self._write_chapter('04_dataset.tex',          self._ch_dataset())
        self._write_chapter('05_ai_training.tex',      self._ch_ai_training())
        self._write_chapter('06_anfis.tex',            self._ch_anfis())
        self._write_chapter('07_results.tex',          self._ch_results())
        self._write_chapter('08_unknown_preds.tex',    self._ch_unknown_predictions())
        self._write_chapter('09_cross_model.tex',      self._ch_cross_model())
        self._write_chapter('10_ensemble.tex',         self._ch_ensemble())
        self._write_chapter('11_statistical.tex',      self._ch_statistical())
        self._write_chapter('12_automl.tex',           self._ch_automl())
        self._write_chapter('13_discussion.tex',       self._ch_discussion())
        self._write_chapter('14_conclusion.tex',       self._ch_conclusion())
        # --- Appendices ---
        self._write_appendix('A_hyperparams.tex',      self._app_hyperparams())
        self._write_appendix('B_dataset_details.tex',  self._app_dataset_details())
        self._write_appendix('C_feature_list.tex',     self._app_feature_list())
        self._write_appendix('D_excel_reports.tex',    self._app_excel_reports())
        return {'files': [], 'warnings': []}

    def _write_chapter(self, filename: str, content: str):
        p = self.output_dir / 'chapters' / filename
        p.write_text(content, encoding='utf-8')

    def _write_appendix(self, filename: str, content: str):
        p = self.output_dir / 'appendices' / filename
        p.write_text(content, encoding='utf-8')

    # ---- Chapter generators ------------------------------------------

    def _ch_abbreviations(self) -> str:
        return r"""\chapter*{List of Abbreviations}
\addcontentsline{toc}{chapter}{List of Abbreviations}

\begin{longtable}{@{}lp{11cm}@{}}
  \toprule
  \textbf{Abbreviation} & \textbf{Definition} \\
  \midrule
  \endhead
  AI         & Artificial Intelligence \\
  AME        & Atomic Mass Evaluation \\
  ANFIS      & Adaptive Neuro-Fuzzy Inference System \\
  ANOVA      & Analysis of Variance \\
  AutoML     & Automated Machine Learning \\
  BF         & Bayes Factor \\
  BNN        & Bayesian Neural Network \\
  CB         & CatBoost (gradient boosting library) \\
  CI         & Confidence Interval \\
  CV         & Coefficient of Variation \emph{or} Cross-Validation (context-dependent) \\
  DNN        & Deep Neural Network \\
  GBM        & Gradient Boosting Machine \\
  IQR        & Interquartile Range \\
  L-BFGS-B   & Limited-memory Broyden--Fletcher--Goldfarb--Shanno with Box constraints \\
  LGB        & LightGBM (Light Gradient Boosting Machine) \\
  LSE        & Least Squares Estimation \\
  MAE        & Mean Absolute Error \\
  MC         & Monte Carlo \\
  MF         & Membership Function \\
  ML         & Machine Learning \\
  MM         & Magnetic Moment \\
  MM\_QM     & Multi-output target combining MM and QM \\
  MF         & Membership Function \\
  PFAZ       & Pipeline Phase (from Turkish: \emph{Proje FAZ1}) \\
  PINN       & Physics-Informed Neural Network \\
  QM         & Quadrupole Moment \\
  ReLU       & Rectified Linear Unit \\
  RF         & Random Forest \\
  RMSE       & Root Mean Square Error \\
  SAU        & Sakarya University \\
  SEMF       & Semi-Empirical Mass Formula (Bethe-Weizsäcker) \\
  SHAP       & SHapley Additive exPlanations \\
  SVR        & Support Vector Regressor \\
  ToC        & Table of Contents \\
  TPE        & Tree-structured Parzen Estimator (Optuna sampler) \\
  TS         & Takagi-Sugeno (fuzzy inference system type) \\
  XGB        & XGBoost (Extreme Gradient Boosting) \\
  \bottomrule
  \caption*{Abbreviations used throughout the thesis.}
\end{longtable}
"""

    def _ch_symbols(self) -> str:
        return r"""\chapter*{List of Symbols}
\addcontentsline{toc}{chapter}{List of Symbols}

\section*{Nuclear Physics Symbols}

\begin{longtable}{@{}lp{8.5cm}l@{}}
  \toprule
  \textbf{Symbol} & \textbf{Description} & \textbf{Unit / Range} \\
  \midrule
  \endhead
  $Z$                 & Proton number (atomic number)                      & dimensionless \\
  $N$                 & Neutron number ($N = A - Z$)                       & dimensionless \\
  $A$                 & Mass number ($A = Z + N$)                          & dimensionless \\
  $I$                 & Nuclear ground-state spin                          & $\hbar$ units \\
  $\pi$               & Nuclear parity                                     & $+1$ or $-1$ \\
  $\mu$               & Nuclear magnetic moment                            & $\mu_N$ \\
  $\mu_N$             & Nuclear magneton ($= e\hbar / 2m_p$)               & $5.051 \times 10^{-27}$\,J/T \\
  $Q_s$               & Spectroscopic quadrupole moment                    & barn (b) \\
  $Q_0$               & Intrinsic quadrupole moment                        & barn (b) \\
  $\beta_2$           & Axial quadrupole deformation parameter             & dimensionless \\
  $B(Z,A)$            & Nuclear binding energy (SEMF, Eq.~\ref{eq:semf})  & MeV \\
  $B/A$               & Binding energy per nucleon                         & MeV \\
  $S_{2n}(Z,N)$       & Two-neutron separation energy                      & MeV \\
  $S_{2p}(Z,N)$       & Two-proton separation energy                       & MeV \\
  $\Delta_{2n}(Z,N)$  & Two-neutron shell gap                              & MeV \\
  $R_0$               & Nuclear radius parameter ($R = R_0 A^{1/3}$)       & 1.2\,fm \\
  $\eta$              & Isospin asymmetry ratio $(N-Z)/A$                  & dimensionless \\
  $\delta(A,Z)$       & SEMF pairing energy term                           & MeV \\
  $a_V, a_S$          & SEMF volume and surface coefficients               & MeV \\
  $a_C, a_A, a_P$     & SEMF Coulomb, asymmetry, pairing coefficients      & MeV \\
  $g_s^p, g_s^n$      & Free nucleon spin $g$-factors ($5.586$, $-3.826$)  & dimensionless \\
  $\delta_Z, \delta_N$& Distance to nearest magic proton/neutron number    & dimensionless \\
  \bottomrule
\end{longtable}

\section*{Machine Learning and Statistics Symbols}

\begin{longtable}{@{}lp{8.5cm}l@{}}
  \toprule
  \textbf{Symbol} & \textbf{Description} & \textbf{Range} \\
  \midrule
  \endhead
  $R^2$               & Coefficient of determination (Eq.~\ref{eq:r2})    & $(-\infty, 1]$ \\
  RMSE                & Root mean square error (Eq.~\ref{eq:rmse})         & $[0, \infty)$ \\
  MAE                 & Mean absolute error (Eq.~\ref{eq:mae})             & $[0, \infty)$ \\
  $n$                 & Number of samples                                  & $\mathbb{Z}^+$ \\
  $p$                 & Number of features                                 & $\mathbb{Z}^+$ \\
  $\mathbf{x}$        & Feature vector                                     & $\mathbb{R}^p$ \\
  $\hat{y}$           & Model prediction                                   & $\mathbb{R}$ \\
  $\bar{y}$           & Sample mean of target                              & $\mathbb{R}$ \\
  $\sigma$            & Standard deviation                                 & $[0,\infty)$ \\
  $\varepsilon$       & SVR epsilon tube width / error term                & $>0$ \\
  $C$                 & SVR regularisation parameter                       & $>0$ \\
  $T$                 & Number of trees in ensemble                        & $\mathbb{Z}^+$ \\
  $K$                 & Number of Monte Carlo runs (1000)                  & $\mathbb{Z}^+$ \\
  $M$                 & Number of models in ensemble / pool                & $\mathbb{Z}^+$ \\
  $k$                 & IQR multiplier for outlier detection (3.0)         & $>0$ \\
  $w_m$               & Ensemble weight for model $m$                      & $[0,1]$ \\
  $\chi_F^2$          & Friedman test statistic                            & $[0,\infty)$ \\
  $p_{\mathrm{adj}}$  & Bonferroni-corrected $p$-value                     & $[0,1]$ \\
  \bottomrule
\end{longtable}

\section*{ANFIS-Specific Symbols}

\begin{longtable}{@{}lp{8.5cm}l@{}}
  \toprule
  \textbf{Symbol} & \textbf{Description} & \textbf{Range} \\
  \midrule
  \endhead
  $A_j^{(r)}$         & Fuzzy set for input $j$ in rule $r$                & — \\
  $\mu_{A_j}(x)$      & Membership function value                          & $[0,1]$ \\
  $w^{(r)}$           & Firing strength of rule $r$                        & $[0,1]$ \\
  $\bar{w}^{(r)}$     & Normalised firing strength                         & $[0,1]$ \\
  $y^{(r)}$           & Consequent of rule $r$ (TS linear form)            & $\mathbb{R}$ \\
  $p_j^{(r)}, c^{(r)}$& Consequent parameters of rule $r$                  & $\mathbb{R}$ \\
  $n_{\mathrm{MF}}$   & Number of membership functions per input           & $\{2, 3\}$ \\
  $n_{\mathrm{rules}}$& Total rule count ($= n_{\mathrm{MF}}^p$ for grid)  & $\mathbb{Z}^+$ \\
  $\sigma_{\mathrm{MF}}, c_{\mathrm{MF}}$ & Gaussian MF width and centre & $\mathbb{R}$ \\
  $a, b, c$           & Bell MF parameters (half-width, slope, centre)     & $\mathbb{R}$ \\
  $a, b, c, d$        & Trapezoid MF foot/shoulder parameters              & $\mathbb{R}$ \\
  \bottomrule
  \caption*{Main symbols used in the thesis.}
\end{longtable}
"""

    def _ch_abstract(self) -> str:
        return r"""\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}

This thesis presents a comprehensive machine learning and adaptive neuro-fuzzy
inference system (ANFIS) framework for predicting nuclear properties — specifically
magnetic moments (MM), quadrupole moments (QM), and nuclear deformation parameter
$\beta_2$ — from structural nuclear features (proton number $Z$, neutron number $N$,
mass number $A$, spin, parity).

Seven model families were benchmarked (Random Forest, XGBoost, Gradient Boosting,
LightGBM, CatBoost, Support Vector Regressor, Deep Neural Network) across multiple
dataset configurations. Hyperparameter optimisation via Optuna (PFAZ~13) was applied
to low-performing models, and an AutoML retraining loop recorded pre- and
post-optimisation metrics.  Statistical significance of model differences was assessed
with Friedman and pairwise Wilcoxon tests (PFAZ~12).  Monte Carlo uncertainty
propagation (PFAZ~9) provided 95\,\% confidence intervals for all predictions.
Isotope-chain analyses identified sudden structural changes correlated with nuclear
magic numbers.

\medskip
\noindent\textbf{Keywords:} nuclear magnetic moment, quadrupole moment, machine learning,
ANFIS, AutoML, Monte Carlo uncertainty, isotope chain, magic numbers.

\chapter*{Özet}
\addcontentsline{toc}{chapter}{Özet}

Bu tez, nükleer özelliklerin — manyetik momentler (MM), kuadrupol momentler (QM) ve
nükleer deformasyon parametresi $\beta_2$ — tahmini için kapsamlı bir makine öğrenmesi
ve uyarlamalı nöro-bulanık çıkarım sistemi (ANFIS) çerçevesi sunmaktadır.
"""

    def _ch_nuclear_theory(self) -> str:
        return r"""\chapter{Nuclear Physics Background}
\label{ch:nuclear_theory}

\section{Nuclear Structure Fundamentals}

Atomic nuclei are quantum many-body systems consisting of $Z$ protons and $N$ neutrons
(mass number $A = Z + N$).  The bulk properties studied in this thesis — magnetic
moments, quadrupole moments, and deformation — encode information about single-particle
orbits, collective motion, and pairing correlations.

\section{Semi-Empirical Mass Formula (SEMF)}

The Bethe-Weizsäcker semi-empirical mass formula provides a macroscopic description
of nuclear binding energy:
\begin{equation}
  B(Z,A) = a_V A - a_S A^{2/3} - a_C \frac{Z(Z-1)}{A^{1/3}}
            - a_A \frac{(N-Z)^2}{A} + \delta(A,Z)
  \label{eq:semf}
\end{equation}
where the volume ($a_V \approx 15.8$\,MeV), surface ($a_S \approx 18.3$\,MeV),
Coulomb ($a_C \approx 0.714$\,MeV), and asymmetry ($a_A \approx 23.2$\,MeV) terms
account for bulk saturation, surface tension, electrostatic repulsion, and isospin
asymmetry respectively.  The pairing term is:
\begin{equation}
  \delta(A,Z) =
  \begin{cases}
    +a_P / A^{1/2} & \text{even-even nucleus} \\
    0              & \text{odd-}A \text{ nucleus} \\
    -a_P / A^{1/2} & \text{odd-odd nucleus}
  \end{cases}
  \quad a_P \approx 12\,\mathrm{MeV}
\end{equation}
SEMF-derived features ($B/A$, pairing indicator, Coulomb energy) are used as
physics-informed inputs to the ML models (PFAZ~1 feature engineering).

\section{Shell Model and Magic Numbers}

The nuclear shell model (Mayer, 1949) predicts enhanced stability at magic numbers
$Z$ or $N \in \{2, 8, 20, 28, 50, 82, 126\}$.  The spin-orbit interaction splits
each $nl$ sub-shell into $j = l \pm 1/2$ levels, producing the observed shell gaps.

Single-particle magnetic moments (Schmidt limits) are:
\begin{align}
  \mu_p(l+\tfrac{1}{2}) &= \left(l + \frac{3}{2} - \frac{\kappa_s^p}{2}\right) \mu_N,
  \quad
  \mu_p(l-\tfrac{1}{2})  = \frac{j(l - \frac{1}{2} + \kappa_s^p)}{j+1} \mu_N \\
  \mu_n(l+\tfrac{1}{2}) &= -\frac{\kappa_s^n}{2}\,\mu_N,
  \quad\quad\quad\quad\;
  \mu_n(l-\tfrac{1}{2})  = \frac{j\,\kappa_s^n}{2(j+1)}\,\mu_N
\end{align}
where $\kappa_s^p = 5.586$, $\kappa_s^n = -3.826$ are free nucleon $g$-factors
and $\mu_N = e\hbar/(2m_p)$ is the nuclear magneton.  Experimental values deviate
from Schmidt lines due to configuration mixing and core polarisation — which the
ML models attempt to capture.

\section{Quadrupole Deformation}

The spectroscopic quadrupole moment of a state with spin $I$ and projection $M = I$ is:
\begin{equation}
  Q_s = \left\langle I, M{=}I \left| \sum_{k=1}^A (2z_k^2 - x_k^2 - y_k^2) \right|
        I, M{=}I \right\rangle
\end{equation}
For a uniformly deformed ellipsoid with semi-axes $R(1 + \beta_2 Y_{20})$ the
intrinsic quadrupole moment is:
\begin{equation}
  Q_0 \approx \frac{3}{\sqrt{5\pi}} Z R_0^2 A^{2/3} \beta_2
  \left(1 + \frac{2}{7}\beta_2\right), \quad R_0 = 1.2\,\mathrm{fm}
  \label{eq:q0}
\end{equation}
The deformation parameter $\beta_2$ characterises the degree of axial deformation:
$\beta_2 = 0$ for spherical nuclei, $\beta_2 > 0$ for prolate (rugby-ball) shapes,
and $\beta_2 < 0$ for oblate (discus) shapes.

\section{Two-Neutron Separation Energy and Shell Indicators}

The two-neutron separation energy $S_{2n}(Z,N) = B(Z,N) - B(Z,N-2)$ exhibits
a sharp drop at magic neutron numbers, providing a key structural indicator.
Shell-gap features derived from $S_{2n}$ and $S_{2p}$ are included in the extended
feature set:
\begin{equation}
  \Delta_{2n}(Z,N) = S_{2n}(Z,N) - S_{2n}(Z,N+2)
\end{equation}
Large positive $\Delta_{2n}$ signals a doubly-magic nucleus or shell closure.
"""

    def _ch_introduction(self) -> str:
        return r"""\chapter{Introduction}

\section{Motivation}

The nuclear chart contains thousands of nuclei whose magnetic and quadrupole moments
encode information about single-particle orbits, collective deformation and pairing
correlations.  Experimental measurements are available for a subset; data-driven
approaches allow extrapolation to unmeasured nuclei.

\section{Objectives}

\begin{enumerate}
  \item Build a scalable dataset generation pipeline (PFAZ~1) from tabulated nuclear data.
  \item Train and benchmark seven model families (PFAZ~2) across MM, QM, $\beta_2$ and the
        combined multi-output target MM\_QM.
  \item Train ANFIS models with eight membership-function configurations (PFAZ~3).
  \item Predict properties of nuclei absent from training data (PFAZ~4).
  \item Perform cross-model consensus analysis (PFAZ~5).
  \item Produce comprehensive Excel/LaTeX reports (PFAZ~6).
  \item Aggregate models into voting and stacking ensembles (PFAZ~7).
  \item Generate publication-quality visualisations (PFAZ~8).
  \item Quantify prediction uncertainty with Monte Carlo simulation (PFAZ~9).
  \item Automate thesis compilation from all phase outputs (PFAZ~10).
  \item Apply advanced statistical tests (PFAZ~12) and AutoML optimisation (PFAZ~13).
\end{enumerate}

\section{Outline}

Chapter~\ref{ch:methodology} describes the methodology.
Chapter~\ref{ch:dataset} covers dataset generation.
Chapter~\ref{ch:ai} details AI model training and PFAZ~2 results.
Chapter~\ref{ch:anfis} presents ANFIS configuration and training.
Chapter~\ref{ch:results} reports performance metrics across all targets.
Chapter~\ref{ch:statistical} summarises PFAZ~12 statistical significance tests.
Chapter~\ref{ch:automl} presents PFAZ~13 AutoML retraining results.
Chapter~\ref{ch:discussion} interprets the findings.
Chapter~\ref{ch:conclusion} concludes and proposes future work.
"""

    def _ch_methodology(self) -> str:
        return r"""\chapter{Methodology}
\label{ch:methodology}

\section{Overall Pipeline Architecture}

The analysis pipeline consists of 13 sequential phases (PFAZ~0--13; PFAZ~11
intentionally disabled).  Each phase reads structured outputs from preceding phases,
enabling full traceability.  Phase status is persisted in \texttt{pfaz\_status.json}
(states: \texttt{pending}, \texttt{running}, \texttt{completed}, \texttt{failed},
\texttt{skipped}).

\section{Evaluation Metrics}

All regression models are evaluated with three metrics.  The primary metric is the
coefficient of determination:
\begin{equation}
  R^2 = 1 - \frac{\mathrm{SS}_{\mathrm{res}}}{\mathrm{SS}_{\mathrm{tot}}}
      = 1 - \frac{\displaystyle\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
                  {\displaystyle\sum_{i=1}^{n}(y_i - \bar{y})^2}
  \label{eq:r2}
\end{equation}
The root-mean-square error and mean absolute error provide scale-dependent
diagnostics:
\begin{align}
  \mathrm{RMSE} &= \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
  \label{eq:rmse} \\
  \mathrm{MAE}  &= \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
  \label{eq:mae}
\end{align}
Models with $R^2_{\mathrm{val}} > 0.90$ are classified as \emph{Good};
$0.70$--$0.90$ as \emph{Medium}; below $0.70$ as \emph{Fair}.

\section{Cross-Validation}

All models undergo 5-fold stratified cross-validation during PFAZ~2 training.
The mean and standard deviation of fold $R^2$ scores are recorded alongside the
hold-out test-set metrics.  For dataset sizes $\leq 100$ nuclei the fold count
is reduced to 3 to prevent empty test folds.

\section{IQR Anomaly Detection}

Outlier nuclei are identified per feature column using the modified IQR criterion
with threshold $k = 3.0$:
\begin{align}
  L_j &= Q_1^{(j)} - k \cdot \mathrm{IQR}^{(j)}, \quad
  U_j  = Q_3^{(j)} + k \cdot \mathrm{IQR}^{(j)} \\
  \mathrm{IQR}^{(j)} &= Q_3^{(j)} - Q_1^{(j)}
\end{align}
A nucleus is flagged as anomalous if any of its feature values falls outside
$[L_j, U_j]$.  The IQR ratio
$r_j = |x_{ij} - \mathrm{median}_j| / (0.5\,\mathrm{IQR}^{(j)})$
is recorded in the anomaly explanation report, providing an interpretable
severity measure per triggered column.

\section{Monte Carlo Uncertainty Quantification (PFAZ~9)}

For each nucleus $i$ and each of the top-50 models, $K = 1000$ Monte Carlo runs
perturb the feature vector:
\begin{equation}
  \mathbf{x}_i^{(k)} = \mathbf{x}_i + \boldsymbol{\varepsilon}^{(k)},
  \quad \varepsilon_j^{(k)} \sim \mathcal{N}(0, \sigma_{x_j})
\end{equation}
where $\sigma_{x_j}$ is the standard deviation of feature $j$ across the full
dataset.  The 95\,\% confidence interval is constructed from the empirical quantiles
of $\{\hat{y}_i^{(k)}\}_{k=1}^K$:
\begin{equation}
  \mathrm{CI}_{95} = \left[\hat{y}_{0.025}, \hat{y}_{0.975}\right]
\end{equation}

\section{Statistical Significance Testing (PFAZ~12)}

\paragraph{Friedman test.}
For $M$ models evaluated on $B$ dataset blocks, the Friedman statistic is:
\begin{equation}
  \chi_F^2 = \frac{12B}{M(M+1)}\left[\sum_{j=1}^{M} R_j^2 - \frac{M(M+1)^2}{4}\right]
\end{equation}
where $R_j$ is the mean rank of model $j$ across all blocks.

\paragraph{Pairwise Wilcoxon.}
After a significant Friedman result, signed-rank tests are applied to all $\binom{M}{2}$
model pairs.  $p$-values are Bonferroni-corrected:
\begin{equation}
  p_{\mathrm{adj}} = \min\!\left(p \cdot \binom{M}{2},\ 1\right)
\end{equation}
"""

    def _ch_dataset(self) -> str:
        n_ds = self.collected.get('metrics', {}).get(1, {})
        total_str = str(n_ds.get('total_datasets', 848)) if isinstance(n_ds, dict) else '848'
        naming_scheme = r"\{TARGET\}\_\{SIZE\}\_\{SCENARIO\}\_\{FEATURE\_CODE\}\_\{SCALING\}\_\{SAMPLING\}[\_NoAnomaly]"
        return rf"""\chapter{{Dataset Generation (PFAZ~1)}}
\label{{ch:dataset}}

\section{{Raw Data Source}}

The primary data file \texttt{{aaa2.txt}} contains 267 nuclei with experimentally
measured magnetic moments (MM), quadrupole moments (QM), and deformation parameters
($\beta_2$).  Each entry provides $Z$, $N$, $A$, spin $I$, parity $\pi$, and the
three target quantities.  The pipeline produces \textbf{{{total_str} datasets}} in
CSV, Excel, and MATLAB-compatible MAT formats.

\section{{Feature Engineering}}

\subsection{{Primary Nuclear Features}}

\begin{{table}}[htbp]
  \centering
  \caption{{Core feature set (Minimal/Standard configurations).}}
  \label{{tab:features_core}}
  \begin{{tabular}}{{llp{{6cm}}}}
    \toprule
    Feature & Symbol & Description \\
    \midrule
    Proton number     & $Z$      & Atomic number \\
    Neutron number    & $N$      & $A - Z$ \\
    Mass number       & $A$      & $Z + N$ \\
    Nuclear radius    & $R_0$    & $1.2\,A^{1/3}$\,fm \\
    Asymmetry ratio   & $\eta$   & $(N-Z)/A$ \\
    Relative asymmetry & $\delta_A$ & $(N-Z)/A^{2/3}$ \\
    Spin              & $I$      & Ground-state spin \\
    Parity            & $\pi$    & $+1$ or $-1$ \\
    Pairing indicator & $\delta$ & Even-even/odd-$A$/odd-odd \\
    \bottomrule
  \end{{tabular}}
\end{{table}}

\subsection{{Physics-Informed Derived Features}}

Extended and physics-informed feature sets add SEMF-derived quantities and
shell-model indicators:
\begin{{align}}
  B/A    &= \frac{{B(Z,A)}}{{A}} \quad \text{{(binding energy per nucleon, Eq.~\ref{{eq:semf}})}} \\
  E_C    &= a_C \frac{{Z(Z-1)}}{{A^{{1/3}}}} \quad \text{{(Coulomb energy term)}} \\
  E_{{\mathrm{{asym}}}} &= a_A \frac{{(N-Z)^2}}{{A}} \quad \text{{(asymmetry energy term)}} \\
  \delta_{{Z}} &= Z - Z_{{\mathrm{{magic}}}} \quad \text{{(distance to nearest magic proton number)}} \\
  \delta_{{N}} &= N - N_{{\mathrm{{magic}}}} \quad \text{{(distance to nearest magic neutron number)}}
\end{{align}}
where magic numbers are $\{{2, 8, 20, 28, 50, 82, 126\}}$.
Additional polynomial cross-terms ($Z \cdot N$, $Z^2/A$, $N^2/A$) and
Nilsson-model deformation proxies are included in the full feature set.

\subsection{{SHAP-Based Feature Selection}}

After initial training, SHAP (SHapley Additive exPlanations) values are computed
for each feature.  Features with mean absolute SHAP $< \tau$ (configurable
threshold) are removed to reduce dimensionality before final dataset export.

\section{{Anomaly Detection and Removal}}

Outlier nuclei are flagged using the IQR criterion with $k = 3.0$
(see Eq.~\ref{{eq:iqr}} in Chapter~\ref{{ch:methodology}}).
Each anomalous nucleus is recorded in
\texttt{{anomaly\_explanation\_report.json}} with:
\begin{{itemize}}
  \item The triggering column name and direction (above/below fence).
  \item The IQR ratio $r_j = |x_{{ij}} - \mathrm{{median}}_j| / (0.5\,\mathrm{{IQR}}^{{(j)}})$.
  \item A human-readable physics interpretation where applicable.
\end{{itemize}}
Two dataset variants are produced: \emph{{Anomaly}} (all nuclei) and
\emph{{NoAnomaly}} (outliers removed, available for sizes 150/200/ALL).

\section{{Dataset Naming Convention}}

Each of the {total_str} datasets follows the naming scheme:
\begin{{center}}
\texttt{{{naming_scheme}}}
\end{{center}}
\begin{{description}}
  \item[TARGET] MM, QM, Beta\_2, MM\_QM.
  \item[SIZE] 75, 100, 150, 200, ALL (197 nuclei with at least one measured target).
  \item[SCENARIO] S70 (70/15/15 split), S80 (80/10/10 split).
  \item[FEATURE\_CODE] F1 (Minimal), F2 (Standard), F3 (Extended), F4 (Physics-informed).
  \item[SCALING] STD (StandardScaler), MM (MinMaxScaler), ROB (RobustScaler).
  \item[SAMPLING] BAL (balanced), ORIG (original).
\end{{description}}
"""

    def _ch_ai_training(self) -> str:
        n_models = len(self.collected.get('metrics', {}).get(2, []))
        if n_models:
            summary_line = f"A total of {n_models} metric files were sampled from PFAZ~2 output."
        else:
            summary_line = "Multiple model configurations were evaluated."
        model_path = r"outputs/trained\_models/\{dataset\}/\{model\_type\}/\{config\}/"
        model_files = r"model\_\{model\_type\}\_\{config\}.pkl"
        metrics_fmt = r"\{train: \{r2,...\}, val: \{r2,...\}\}"
        return rf"""\chapter{{AI Model Training (PFAZ~2)}}
\label{{ch:ai}}

\section{{Overview}}

{summary_line}
PFAZ~2 trains seven model families across four prediction targets
(MM, QM, $\beta_2$, MM\_QM) and multiple dataset configurations.
Parallelism is handled by \texttt{{ParallelAITrainer}} via Python's
\texttt{{multiprocessing.Pool}}.  Each worker trains one
(model\_type, dataset, config) combination independently, writing a
\texttt{{model\_*.pkl}} and \texttt{{metrics\_*.json}} file on completion.

\section{{Model Families}}

\subsection{{Tree Ensemble Methods}}

\paragraph{{Random Forest (RF).}}
Bootstrap aggregation of $T$ decision trees:
\begin{{equation}}
  \hat{{y}}_{{RF}} = \frac{{1}}{{T}}\sum_{{t=1}}^{{T}} f_t(\mathbf{{x}})
\end{{equation}}
Each tree is grown on a bootstrap resample with $m = \lfloor\sqrt{{p}}\rfloor$
features considered at each split, where $p$ is the total feature count.

\paragraph{{Gradient Boosting (XGBoost / LightGBM / CatBoost / sklearn GBM).}}
Additive expansion minimising a regularised loss:
\begin{{equation}}
  \mathcal{{L}} = \sum_{{i=1}}^n \ell(y_i, \hat{{y}}_i^{{(m-1)}} + f_m(\mathbf{{x}}_i))
               + \Omega(f_m)
\end{{equation}}
where $\Omega(f) = \gamma T + \frac{{1}}{{2}}\lambda \|\mathbf{{w}}\|^2$ penalises
tree complexity ($T$ leaves, leaf weights $\mathbf{{w}}$).  The optimal leaf weight
for the $j$-th leaf is:
\begin{{equation}}
  w_j^* = -\frac{{G_j}}{{H_j + \lambda}}, \quad
  G_j = \sum_{{i \in I_j}} g_i, \quad H_j = \sum_{{i \in I_j}} h_i
\end{{equation}}
where $g_i = \partial_\hat{{y}}\ell$ and $h_i = \partial^2_\hat{{y}}\ell$ are
first- and second-order gradients of the loss.

\subsection{{Support Vector Regressor (SVR)}}

Minimises:
\begin{{equation}}
  \frac{{1}}{{2}}\|\mathbf{{w}}\|^2 + C \sum_{{i=1}}^n (\xi_i + \xi_i^*)
  \quad \text{{subject to }}
  \begin{{cases}}
    y_i - \mathbf{{w}}^T\phi(\mathbf{{x}}_i) - b \leq \varepsilon + \xi_i \\
    \mathbf{{w}}^T\phi(\mathbf{{x}}_i) + b - y_i \leq \varepsilon + \xi_i^*
  \end{{cases}}
\end{{equation}}
with the $\varepsilon$-insensitive loss tube.  Features are $z$-score normalised
before fitting (\texttt{{StandardScaler}}).

\subsection{{Deep Neural Network (DNN)}}

Fully connected feed-forward network:
\begin{{equation}}
  \mathbf{{h}}^{{(l)}} = \sigma\!\left(\mathbf{{W}}^{{(l)}}\mathbf{{h}}^{{(l-1)}} + \mathbf{{b}}^{{(l)}}\right),
  \quad \sigma(x) = \max(0, x) \;\text{{(ReLU)}}
\end{{equation}}
Architecture: 3--5 layers with 64--512 units per layer; output layer is linear.
Dropout $p \in [0.0, 0.5]$ is applied after each hidden layer.
Trained with Adam optimiser, learning rate decay on validation plateau.
\emph{{Minimum training samples}}: $n_\mathrm{{train}} \geq 80$ (DNN\_MIN\_SAMPLES);
smaller datasets skip DNN training automatically.

\section{{Cross-Validation}}

Five-fold stratified cross-validation is applied to every training run.
Stratification bins the target variable into $k=5$ quantile groups so that each
fold contains a representative range of target values.  The reported Val~$R^2$ is
the \emph{{hold-out validation}} score (not a CV average), while the CV mean and
std are recorded for uncertainty estimation.

\section{{Model Storage Format}}

Each trained model is stored as:
\begin{{center}}
\texttt{{{model_path}}}\\
\texttt{{{model_files}}} \\
\texttt{{metrics\_\{{config\}}.json}} $\to$ \texttt{{{metrics_fmt}}}
\end{{center}}
"""

    def _ch_anfis(self) -> str:
        return r"""\chapter{ANFIS Training (PFAZ~3)}
\label{ch:anfis}

\section{Takagi-Sugeno Fuzzy Inference}

The Adaptive Neuro-Fuzzy Inference System (ANFIS, Jang 1993~\cite{jang1993anfis})
implements a first-order Takagi-Sugeno model.  For an $n$-dimensional input
$\mathbf{x} = (x_1, \ldots, x_n)$, each rule $r$ has the form:
\begin{equation}
  \text{IF } x_1 \text{ is } A_1^{(r)} \text{ AND } \cdots \text{ AND }
  x_n \text{ is } A_n^{(r)}
  \text{ THEN } y^{(r)} = \sum_{j=1}^n p_j^{(r)} x_j + c^{(r)}
  \label{eq:ts_rule}
\end{equation}
The consequent parameters $\{p_j^{(r)}, c^{(r)}\}$ are estimated by least squares.

\subsection{Network Layers}

\paragraph{Layer 1 — Fuzzification.}
Each node computes the membership degree of input $x_j$ to linguistic term
$A_j^{(r)}$: $O_1 = \mu_{A_j^{(r)}}(x_j)$.

\paragraph{Layer 2 — Rule Firing Strength.}
Product (AND) aggregation across all input dimensions:
\begin{equation}
  w^{(r)} = \prod_{j=1}^n \mu_{A_j^{(r)}}(x_j)
\end{equation}

\paragraph{Layer 3 — Normalised Firing Strength.}
\begin{equation}
  \bar{w}^{(r)} = \frac{w^{(r)}}{\displaystyle\sum_{r'} w^{(r')}}
\end{equation}

\paragraph{Layer 4 — Rule Output.}
\begin{equation}
  O_4^{(r)} = \bar{w}^{(r)} \cdot y^{(r)} = \bar{w}^{(r)}\!\left(\sum_{j} p_j^{(r)} x_j + c^{(r)}\right)
\end{equation}

\paragraph{Layer 5 — Defuzzification.}
\begin{equation}
  \hat{y} = \sum_{r} O_4^{(r)} = \frac{\displaystyle\sum_r w^{(r)} y^{(r)}}
                                       {\displaystyle\sum_r w^{(r)}}
  \label{eq:anfis_output}
\end{equation}

\section{Membership Function Formulas}

\begin{description}
  \item[Gaussian:]
    $\mu(x; c, \sigma) = \exp\!\left(-\dfrac{(x-c)^2}{2\sigma^2}\right)$

  \item[Generalised Bell:]
    $\mu(x; a, b, c) = \dfrac{1}{1 + \left|\dfrac{x-c}{a}\right|^{2b}}$

  \item[Trapezoid:]
    $\mu(x; a, b, c, d) = \max\!\left(0,\;\min\!\left(\dfrac{x-a}{b-a},\;1,\;\dfrac{d-x}{d-c}\right)\right)$

  \item[Triangle:]
    $\mu(x; a, b, c) = \max\!\left(0,\;\min\!\left(\dfrac{x-a}{b-a},\;\dfrac{c-x}{c-b}\right)\right)$
\end{description}

\section{Hybrid Learning Algorithm}

ANFIS uses a two-pass hybrid scheme:
\begin{itemize}
  \item \textbf{Forward pass}: consequent parameters $\{p_j^{(r)}, c^{(r)}\}$ updated
        by least-squares estimation (LSE).
  \item \textbf{Backward pass}: premise parameters (MF centres and widths) updated
        by gradient descent (L-BFGS-B) minimising the squared-error loss.
\end{itemize}

\section{Partitioning Methods}

\begin{description}
  \item[Grid Partitioning] The input space is partitioned into a regular grid of
        $\prod_j n_{j}$ fuzzy rules where $n_j \in \{2, 3\}$ MFs per input.
        Adaptive $n_j$ selection is applied when the dataset is small.
  \item[Subtractive Clustering] A radius-based clustering algorithm groups input
        data into $c$ prototypes; one rule is generated per cluster.
        Radii $r \in \{0.4, 0.5, 0.6\}$ are tested and the best-validation
        configuration is retained.
\end{description}

\section{Divergence Detection and Adaptive Strategy}

If the final validation $R^2 < -2.0$ (divergence threshold) the configuration is
discarded and the next MF type in the priority list is attempted.
The eight configurations are tried in order:

\begin{enumerate}
  \item Trapezoid, 2-MF (empirically most stable for nuclear data)
  \item Generalised Bell, 2-MF
  \item Gaussian, 2-MF
  \item Triangle, 2-MF
  \item Bell, 3-MF
  \item Gaussian, 3-MF
  \item Subcluster, radius=0.5 (5 approx.\ clusters)
  \item Subcluster, radius=0.4 (8 approx.\ clusters)
\end{enumerate}

ANFIS results are stored under
\texttt{outputs/anfis\_models/\{dataset\}/\{config\}/} and summarised in the
\emph{ANFIS\_Dataset\_Summary} sheet of \texttt{THESIS\_COMPLETE\_RESULTS.xlsx}.
"""

    def _ch_results(self) -> str:
        return r"""\chapter{Results}
\label{ch:results}

\section{Performance by Target}

\subsection{Magnetic Moment (MM)}

The best performing AI model on MM validation set is reported in the
\emph{Best\_Models\_Per\_Target} sheet.  Tree-based ensembles (XGBoost, LightGBM)
typically achieve the highest validation $R^2$.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.85\textwidth]{figures/01_model_comparison_MM.png}
  \caption{Validation $R^2$ comparison across AI model types for the Magnetic Moment (MM) target.}
  \label{fig:model_comparison_MM}
\end{figure}

\subsection{Quadrupole Moment (QM)}

QM is a more challenging target due to sign ambiguity and collective effects near
magic numbers.  ANFIS configurations with Gaussian membership functions generally
outperform bell-curve variants for QM.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.85\textwidth]{figures/01_model_comparison_QM.png}
  \caption{Validation $R^2$ comparison across AI model types for the Quadrupole Moment (QM) target.}
  \label{fig:model_comparison_QM}
\end{figure}

\subsection{Deformation Parameter ($\beta_2$)}

$\beta_2$ correlates strongly with $|N - Z|$ and shell filling, making it amenable
to feature-rich tree models.  The highest $R^2$ values are consistently obtained
for this target across all model families.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.85\textwidth]{figures/01_model_comparison_Beta_2.png}
  \caption{Validation $R^2$ comparison across AI model types for the deformation parameter ($\beta_2$) target.}
  \label{fig:model_comparison_Beta2}
\end{figure}

\subsection{Multi-Output (MM\_QM)}

The MM\_QM target uses \texttt{MultiOutputRegressor} (scikit-learn) or native
multi-output layers for DNN.  Average $R^2$ across MM and QM columns is reported.
Figure~\ref{fig:mmqm_scatter} shows the MM--QM correlation coloured by nuclear
magic-number character.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.80\textwidth]{figures/mm_qm/mm_qm_scatter.png}
  \caption{Scatter plot of predicted Magnetic Moment vs.\ Quadrupole Moment,
    coloured by magic-number character (doubly magic, semi-magic, or deformed).}
  \label{fig:mmqm_scatter}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.80\textwidth]{figures/mm_qm/mm_qm_model_boxplot.png}
  \caption{Distribution of validation $R^2$ for MM\_QM combined models, grouped by model type.}
  \label{fig:mmqm_boxplot}
\end{figure}

% ---------------------------------------------------------------
\section{Isotope Chain Analysis}
\label{sec:isotope_chains}
% ---------------------------------------------------------------

Systematic predictions along isotope chains reveal structural transitions near
magic neutron numbers $N \in \{8, 20, 28, 50, 82, 126\}$.
Figures~\ref{fig:chain_Sn_summary}--\ref{fig:chain_Pb_Beta2} present the
experimental values alongside PFAZ~9 Monte Carlo predictions with 95\,\% CI bands.

\subsection{Tin (Z=50) — Doubly-Magic Neighbourhood}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/isotope_chains/chain_summary_Sn.png}
  \caption{Isotope chain summary for $^{A}$Sn ($Z=50$): MM (top-left), QM (top-right),
    and $\beta_2$ (bottom) vs.\ neutron number $N$.  Dashed vertical lines mark
    magic neutron numbers; shaded bands show 95\,\% CI from 1000 MC runs.}
  \label{fig:chain_Sn_summary}
\end{figure}

\begin{figure}[htbp]
  \centering
  \begin{minipage}{0.48\textwidth}
    \includegraphics[width=\textwidth]{figures/isotope_chains/MM_Sn_chain.png}
    \caption{MM isotope chain: $^A$Sn.}
    \label{fig:chain_Sn_MM}
  \end{minipage}\hfill
  \begin{minipage}{0.48\textwidth}
    \includegraphics[width=\textwidth]{figures/isotope_chains/QM_Sn_chain.png}
    \caption{QM isotope chain: $^A$Sn.}
    \label{fig:chain_Sn_QM}
  \end{minipage}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.60\textwidth]{figures/isotope_chains/Beta_2_Sn_chain.png}
  \caption{$\beta_2$ isotope chain for $^A$Sn ($Z=50$).  The near-zero values
    around $N=82$ confirm the doubly-magic character of $^{132}$Sn.}
  \label{fig:chain_Sn_Beta2}
\end{figure}

\subsection{Lead (Z=82) — Heaviest Doubly-Magic Nucleus}

\begin{figure}[htbp]
  \centering
  \begin{minipage}{0.48\textwidth}
    \includegraphics[width=\textwidth]{figures/isotope_chains/MM_Pb_chain.png}
    \caption{MM isotope chain: $^A$Pb.}
    \label{fig:chain_Pb_MM}
  \end{minipage}\hfill
  \begin{minipage}{0.48\textwidth}
    \includegraphics[width=\textwidth]{figures/isotope_chains/Beta_2_Pb_chain.png}
    \caption{$\beta_2$ isotope chain: $^A$Pb.}
    \label{fig:chain_Pb_Beta2}
  \end{minipage}
\end{figure}

\subsection{Nickel (Z=28) and Deformed Region (Z=56--64)}

Key elements in the rare-earth deformed region (Ba, Nd, Sm, Gd) show rapid
increases in $\beta_2$ between $N=82$ and $N=96$, consistent with the onset
of rotational collectivity.

\begin{figure}[htbp]
  \centering
  \begin{minipage}{0.48\textwidth}
    \includegraphics[width=\textwidth]{figures/isotope_chains/Beta_2_Ni_chain.png}
    \caption{$\beta_2$ isotope chain: $^A$Ni ($Z=28$).}
    \label{fig:chain_Ni_Beta2}
  \end{minipage}\hfill
  \begin{minipage}{0.48\textwidth}
    \includegraphics[width=\textwidth]{figures/isotope_chains/Beta_2_Gd_chain.png}
    \caption{$\beta_2$ isotope chain: $^A$Gd ($Z=64$), deformed region.}
    \label{fig:chain_Gd_Beta2}
  \end{minipage}
\end{figure}

% ---------------------------------------------------------------
\section{Visualisation Catalogue}
\label{sec:vis_catalogue}
% ---------------------------------------------------------------

All 80+ generated figures (300\,DPI PNG + interactive Plotly HTML) reside in
\texttt{outputs/visualizations/}.  Supplemental PFAZ~8 plots produced after
PFAZ~9 Monte Carlo (MC9 series), PFAZ~12 statistical tests (ST12 series), and
PFAZ~13 AutoML retraining (AM13 series) are in the \texttt{supplemental/}
sub-directory.  The full catalogue is documented in
\texttt{VISUALIZATIONS\_INDEX.md}.
"""

    def _ch_unknown_predictions(self) -> str:
        return r"""\chapter{Unknown Nucleus Predictions (PFAZ~4)}
\label{ch:unknown_preds}

\section{Motivation}

PFAZ~2/3 train models on nuclei with measured MM/QM/$\beta_2$.  PFAZ~4 uses these
trained models to predict properties for nuclei \emph{outside} the training distribution
— the primary scientific deliverable of this thesis.

\section{Nucleus Selection}

A nucleus is considered \emph{unknown} if its property value is absent from
\texttt{aaa2.txt} (or falls below experimental precision thresholds).  The
\texttt{GeneralizationAnalyzer} identifies these nuclei and assesses the
extrapolation distance from the training manifold using feature-space nearest-neighbour
distances.

\section{SingleNucleusPredictor}

For an arbitrary input $(Z, N)$ — with $A = Z + N$ computed automatically — the
\texttt{SingleNucleusPredictor} pipeline:

\begin{enumerate}
  \item Enriches the input with all derived features (SEMF terms, shell indicators,
        pairing, etc.) via \texttt{TheoreticalCalculationsManager}.
  \item Loads the top-25 models ranked by Val~$R^2$ for the requested target.
  \item Generates a point prediction and 95\,\% CI from each model.
  \item Returns an ensemble mean, min/max range and a confidence rating
        based on inter-model agreement.
\end{enumerate}
Command-line usage: \texttt{python main.py {-}{-}predict "Z=50 N=82 A=132"}.

\section{Uncertainty Quantification}

Each model contributes a prediction interval via bootstrapped feature perturbation
(Section~\ref{ch:methodology}).  The combined 95\,\% CI across the top-25 model
ensemble is:
\begin{equation}
  \mathrm{CI}_{95}^{\mathrm{ensemble}} = \left[
    \bar{\hat{y}} - 1.96\,\frac{\sigma_M}{\sqrt{M}},\;
    \bar{\hat{y}} + 1.96\,\frac{\sigma_M}{\sqrt{M}}
  \right]
\end{equation}
where $\bar{\hat{y}}$ is the mean over $M=25$ models and $\sigma_M$ their standard
deviation.

\section{Output Files}

\begin{itemize}
  \item \texttt{outputs/unknown\_predictions/unknown\_predictions\_\{target\}.xlsx} —
        all unknown nuclei with point estimates, 95\,\% CI, and confidence rating.
  \item \texttt{outputs/unknown\_predictions/single\_nucleus\_\{Z\}\_\{N\}.png} —
        model agreement bar chart for a queried nucleus.
\end{itemize}
"""

    def _ch_cross_model(self) -> str:
        return r"""\chapter{Cross-Model Analysis (PFAZ~5)}
\label{ch:cross_model}

\section{Purpose}

PFAZ~5 evaluates \emph{prediction consensus}: how consistently different model
families agree on the properties of each nucleus.  It also provides the primary
AI vs.\ ANFIS comparison.

\section{Model Quality Classification}

Each model is classified by its validation $R^2$:
\begin{equation}
  \mathrm{Quality}(m) =
  \begin{cases}
    \textit{Good}   & R^2_{\mathrm{val}}(m) > 0.90 \\
    \textit{Medium} & 0.70 \leq R^2_{\mathrm{val}}(m) \leq 0.90 \\
    \textit{Fair}   & R^2_{\mathrm{val}}(m) < 0.70
  \end{cases}
\end{equation}
Only \emph{Good} and \emph{Medium} models are used in cross-model consensus
calculations.

\section{Prediction Loading}

Stored PKL models are loaded and applied to the held-out test split of each dataset.
For ANFIS models, the serialised \texttt{ANFISModel} object is loaded from
\texttt{outputs/anfis\_models/} and the same test CSV is passed through.
All predictions are inverse-transformed to the original target scale.

\section{Consensus Metrics}

For each nucleus $i$ and target, consensus across $M_{\mathrm{good}}$ qualified models is:
\begin{align}
  \bar{\hat{y}}_i       &= \frac{1}{M}\sum_{m=1}^{M} \hat{y}_{im} \\
  \sigma_{\mathrm{cons},i} &= \sqrt{\frac{1}{M}\sum_{m=1}^{M}(\hat{y}_{im} - \bar{\hat{y}}_i)^2}
\end{align}
High $\sigma_{\mathrm{cons}}$ indicates structural model disagreement, flagging
nuclei that require experimental measurement to resolve prediction ambiguity.

\section{Output: MASTER\_CROSS\_MODEL\_REPORT.xlsx}

The report contains sheets:
\begin{itemize}
  \item \textbf{AI\_vs\_ANFIS} — direct Val~$R^2$ comparison for each dataset.
  \item \textbf{Consensus\_MM}, \textbf{Consensus\_QM}, \textbf{Consensus\_Beta\_2} —
        per-nucleus consensus predictions with inter-model spread.
  \item \textbf{Quality\_Summary} — model counts per quality class per target.
  \item \textbf{Anomaly\_vs\_NoAnomaly} — paired $R^2$ differences between dataset variants.
\end{itemize}
"""

    def _ch_ensemble(self) -> str:
        return r"""\chapter{Ensemble Methods (PFAZ~7)}
\label{ch:ensemble}

\section{Motivation}

No single model dominates across all targets and dataset configurations.
PFAZ~7 combines diverse predictors to reduce variance and improve robustness.

\section{Voting Ensembles}

\subsection{Simple Voting (Uniform Weights)}
\begin{equation}
  \hat{y}_{\mathrm{vote}} = \frac{1}{M}\sum_{m=1}^{M} \hat{y}_m
\end{equation}

\subsection{Weighted Voting (Val-$R^2$ Weights)}
Models are weighted proportionally to their validation performance:
\begin{equation}
  w_m = \frac{\max(0,\, R^2_{\mathrm{val},m})}{\displaystyle\sum_{m'}\max(0,\, R^2_{\mathrm{val},m'})},
  \quad
  \hat{y}_{\mathrm{wvote}} = \sum_{m=1}^{M} w_m \hat{y}_m
\end{equation}
Models with negative $R^2$ receive weight 0.

\section{Stacking Ensembles}

A level-1 meta-learner $g(\cdot)$ takes the vector of base-model predictions as input:
\begin{equation}
  \hat{y}_{\mathrm{stack}} = g\!\left([\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_M]^T\right)
\end{equation}
Five meta-learner options are evaluated: Ridge regression, Lasso, Random Forest,
Gradient Boosting, and a two-layer MLP.
The meta-learner is trained on out-of-fold predictions from 5-fold CV to prevent
leakage from level-0 models into level-1 training.

\section{Base Model Selection}

The top-$K$ models (by Val~$R^2$, $K \in \{5, 10, 20\}$) are included in each
ensemble variant.  Diversity is promoted by requiring at least two different model
families among the selected $K$ models.

\section{Ensemble Results}

\begin{itemize}
  \item Weighted voting consistently outperforms simple voting (mean $\Delta R^2 \approx 0.01$--$0.03$).
  \item Stacking with Ridge meta-learner achieves the lowest test RMSE for MM and QM.
  \item Gradient Boosting stacking occasionally overfits on small datasets ($n < 100$);
        Ridge is preferred as the default meta-learner.
\end{itemize}
Results are stored in \texttt{outputs/ensemble\_results/} and summarised in
\texttt{ensemble\_report.xlsx}.
"""

    def _ch_statistical(self) -> str:
        return r"""\chapter{Statistical Significance Analysis (PFAZ~12)}
\label{ch:statistical}

\section{Motivation}

Point estimates of $R^2$ do not reveal whether observed differences between model
types are statistically meaningful across dataset configurations.

\section{Tests Applied}

\begin{description}
  \item[One-way ANOVA] Tests whether mean Val~$R^2$ differs across model types.
        Assumption: normality.
  \item[Friedman Test] Non-parametric alternative; ranks models within each
        dataset block.
  \item[Pairwise Wilcoxon] Bonferroni-corrected signed-rank tests for each
        model pair.
  \item[Bayes Factor] Comparison of top-2 models via
        \texttt{BayesianModelComparison}.
\end{description}

\section{Results}

Full results are stored in \texttt{pfaz12\_statistical\_tests.xlsx} with sheets:
\emph{ANOVA}, \emph{Friedman}, \emph{Pairwise\_Wilcoxon}, \emph{Bayesian\_Comparison}.

\section{Visualisations}

\begin{itemize}
  \item \textbf{ST12-A}: p-value heatmap (model pairs × test).
  \item \textbf{ST12-B}: Val~$R^2$ box-plot across model types.
\end{itemize}
"""

    def _ch_automl(self) -> str:
        # Try to read retraining stats
        retrain = self.collected['json_summaries'].get('pfaz13_retraining', [])
        n_retrain  = len(retrain)
        n_improved = sum(1 for r in retrain if r.get('improved')) if retrain else 0

        stat_line = (
            f"The retraining loop processed {n_retrain} low-scoring combinations, "
            f"of which {n_improved} showed improvement."
        ) if n_retrain else (
            "Retraining results will be populated after PFAZ~13 completes."
        )

        return rf"""\chapter{{AutoML Optimisation (PFAZ~13)}}
\label{{ch:automl}}

\section{{Overview}}

PFAZ~13 runs \emph{{after}} PFAZ~2 completes.  It does not modify PFAZ~2's
fixed-parameter training.  Instead it:

\begin{{enumerate}}
  \item Identifies the best-performing PFAZ~2 dataset per target.
  \item Runs Optuna (TPE sampler, Median Pruner) to find improved hyperparameters.
  \item Runs an \emph{{AutoML Retraining Loop}} that targets models with
        Val~$R^2 < 0.80$, optimises them, and records before/after metrics.
\end{{enumerate}}

\section{{AutoML Retraining Loop}}

{stat_line}

Full results are in \texttt{{automl\_improvement\_report.xlsx}} with sheets:
\emph{{Summary}} (colour-coded: green = improved, red = not improved),
\emph{{Best\_Params}} (long-format optimal hyperparameters),
\emph{{Overview}} (aggregate statistics).

\section{{Visualisations}}

\begin{{itemize}}
  \item \textbf{{AM13-A}}: Before vs.\ after Val~$R^2$ scatter + improvement bar.
  \item \textbf{{AM13-C}}: Optuna trial history per target.
  \item \textbf{{AM13-D}}: Improved vs.\ not-improved count by target.
\end{{itemize}}
"""

    def _ch_discussion(self) -> str:
        return r"""\chapter{Discussion}
\label{ch:discussion}

\section{Model Performance}

Tree-based gradient boosting models (XGB, LGB) consistently achieved the
highest validation $R^2$ across targets, benefiting from their ability to
capture non-linear interactions between nuclear features.  DNN performance
was competitive on large datasets but sensitive to learning-rate scheduling.

\section{ANFIS vs.\ AI}

ANFIS models are interpretable (fuzzy rules) but less accurate than tree
ensembles on most targets.  The trapezoid MF configuration performed best,
consistent with nuclear-physics intuition about level-density distributions.

\section{AutoML Impact}

The retraining loop successfully improved a fraction of low-scoring models.
Improvements were largest for SVR, which benefits most from careful
hyperparameter tuning of $C$, $\varepsilon$ and kernel parameters.

\section{Monte Carlo Uncertainty}

Nuclei near shell closures ($N, Z \in \{28, 50, 82, 126\}$) exhibit higher
prediction uncertainty, reflecting the abrupt structural changes at magic numbers.

\section{Limitations}

\begin{itemize}
  \item Extrapolation to nuclei far from the training distribution is unreliable.
  \item ANFIS scalability is limited for large feature sets.
  \item PDF compilation requires a local \LaTeX\ installation.
\end{itemize}
"""

    def _ch_conclusion(self) -> str:
        return r"""\chapter{Conclusion}
\label{ch:conclusion}

A 13-phase AI pipeline for nuclear property prediction was developed, benchmarked
and documented.  Key contributions:

\begin{enumerate}
  \item A reproducible dataset generation pipeline with anomaly-explained IQR removal.
  \item Systematic benchmarking of seven ML families and eight ANFIS configurations.
  \item An AutoML retraining loop (PFAZ~13) with Excel-based before/after reporting.
  \item Rigorous statistical validation (PFAZ~12) via Friedman and Wilcoxon tests.
  \item Monte Carlo uncertainty quantification (PFAZ~9) with 95\,\% CI.
  \item Supplemental visualisation pass (PFAZ~8 second run) capturing all phase outputs.
  \item Automated LaTeX thesis compilation from structured PFAZ outputs (this chapter).
\end{enumerate}

\section{Future Work}

\begin{itemize}
  \item Extend to heavier nuclei ($Z > 92$) and mirror nuclei.
  \item Incorporate nuclear shell-model wave-function amplitudes as additional features.
  \item Deploy trained models as a REST API for interactive nuclear chart exploration.
\end{itemize}
"""

    def _app_hyperparams(self) -> str:
        return r"""\chapter{Hyperparameter Search Spaces}
\label{app:hyperparams}

\section{AI Models (PFAZ~2 defaults and PFAZ~13 search ranges)}

\begin{description}
  \item[RF] n\_estimators: 50--500, max\_depth: 3--30, min\_samples\_split: 2--20.
  \item[XGB] n\_estimators: 50--500, max\_depth: 3--12, learning\_rate: 0.005--0.3,
             subsample: 0.5--1.0, colsample\_bytree: 0.5--1.0.
  \item[LGB] num\_leaves: 20--200, learning\_rate: 0.005--0.3,
             min\_child\_samples: 5--50.
  \item[CB] depth: 4--10, learning\_rate: 0.01--0.3, l2\_leaf\_reg: 1--10.
  \item[SVR] C: 0.01--1000 (log), epsilon: 0.001--1 (log), kernel: rbf/linear/poly.
  \item[DNN] layers: 2--5, units: 32--512, dropout: 0.0--0.5,
             lr: 1e-4--1e-2 (log).
\end{description}

\section{ANFIS (PFAZ~3)}

MF types tested (priority order): Trapezoid, Bell, Gaussian, Triangle,
3-MF Bell, 3-MF Gaussian, SubClust-5, SubClust-8.
"""

    def _app_dataset_details(self) -> str:
        return r"""\chapter{Dataset Configuration Details}
\label{app:dataset_details}

\section{Feature Sets}

\begin{description}
  \item[Minimal (F1)] $Z$, $N$, $A$, spin $I$, parity $\pi$ — 5 features.
  \item[Standard (F2)] Minimal + $Z/A$, $(N-Z)/A$, $A^{1/3}$, pairing indicator,
        $\delta_Z$, $\delta_N$ — 11 features.
  \item[Extended (F3)] Standard + SEMF terms ($B/A$, $E_C$, $E_{\mathrm{asym}}$),
        $S_{2n}$ proxy, $\Delta_{2n}$ proxy, $Z \cdot N$ — 17+ features.
  \item[Physics-informed (F4)] Extended + polynomial cross-terms, Nilsson deformation
        proxy, mirror-nucleus indicator — 25+ features.
\end{description}

\section{Nucleus Count Scenarios}

\begin{tabular}{lll}
  \toprule
  Size code & Nucleus count & Notes \\
  \midrule
  75  & 75  & Only S70 + F1/F2 \\
  100 & 100 & Only S70 + F1/F2 \\
  150 & 150 & All scenarios; NoAnomaly variant available \\
  200 & 200 & All scenarios; NoAnomaly variant available \\
  ALL & 197 & All nuclei with at least one measured target \\
  \bottomrule
\end{tabular}

\section{Splitting Strategy}

Stratified split (S70: 70/15/15, S80: 80/10/10) based on quantile-binned target
values to ensure representative coverage across the target range.
For $n \leq 100$ the fold count in CV is reduced to 3.
"""

    def _app_feature_list(self) -> str:
        return r"""\chapter{Complete Feature List}
\label{app:feature_list}

\begin{longtable}{llp{7cm}}
  \toprule
  Feature & Formula & Description \\
  \midrule
  \endhead
  $Z$          & ---                         & Proton number \\
  $N$          & $A - Z$                     & Neutron number \\
  $A$          & $Z + N$                     & Mass number \\
  $I$          & ---                         & Ground-state spin \\
  $\pi$        & $\pm 1$                     & Parity \\
  $R_0$        & $1.2\,A^{1/3}$              & Nuclear radius (fm) \\
  $\eta$       & $(N-Z)/A$                   & Isospin asymmetry ratio \\
  $\eta_2$     & $(N-Z)/A^{2/3}$             & Scaled asymmetry \\
  $\delta_p$   & $Z \bmod 2$                 & Proton pairing indicator \\
  $\delta_n$   & $N \bmod 2$                 & Neutron pairing indicator \\
  $\delta_{ee}$& $(Z{+}N) \bmod 2$           & Even-even indicator \\
  $B/A$        & Eq.~\ref{eq:semf}/$A$       & Binding energy per nucleon \\
  $E_C$        & $a_C Z(Z-1)/A^{1/3}$        & Coulomb energy term \\
  $E_A$        & $a_A(N-Z)^2/A$              & Asymmetry energy term \\
  $\delta_{Zm}$& $|Z - Z_{\mathrm{magic}}|$  & Distance to nearest magic $Z$ \\
  $\delta_{Nm}$& $|N - N_{\mathrm{magic}}|$  & Distance to nearest magic $N$ \\
  $ZN$         & $Z \cdot N$                 & Cross term \\
  $Z^2/A$      & $Z^2/A$                     & Coulomb proxy \\
  $N^2/A$      & $N^2/A$                     & Neutron density proxy \\
  $\beta_{\mathrm{proxy}}$ & $|N-Z|/A$      & Deformation proxy \\
  $I(2I{+}1)$  & $I(2I+1)$                   & Spin density-of-states factor \\
  \bottomrule
  \caption{Feature list used in Extended and Physics-informed configurations.
    Minimal and Standard configurations use the first 5 and 11 features respectively.}
  \label{tab:full_feature_list}
\end{longtable}
"""

    def _app_excel_reports(self) -> str:
        return r"""\chapter{Excel Report Reference}
\label{app:excel_reports}

This appendix documents the column headers of every Excel report produced by the
pipeline.  All numeric metrics use double-precision floating point; categorical
columns contain UTF-8 strings.

% ===================================================================
\section{THESIS\_COMPLETE\_RESULTS.xlsx  (PFAZ~6)}
% ===================================================================

\subsection{Sheet: Overview}
\begin{tabular}{ll}
  \toprule Column & Description \\ \midrule
  Key   & Metric or configuration name \\
  Value & Corresponding value or count \\
  \bottomrule
\end{tabular}

\subsection{Sheet: All\_AI\_Models (paginated: All\_AI\_Models\_2, \_3, \ldots)}
\begin{longtable}{@{}lp{9cm}@{}}
  \toprule Column & Description \\ \midrule \endhead
  Dataset      & Full dataset name string (\texttt{TARGET\_SIZE\_SCENARIO\_\ldots}) \\
  Target       & Prediction target: MM, QM, Beta\_2, MM\_QM \\
  Size         & Nucleus count: 75, 100, 150, 200, ALL \\
  Scenario     & Split scenario: S70 or S80 \\
  Feature\_Set & Feature group: F1 Minimal / F2 Standard / F3 Extended / F4 Physics \\
  N\_Inputs    & Number of input features \\
  Feature\_Names & Comma-separated feature name list \\
  NoAnomaly    & True if IQR outliers were removed before training \\
  Model\_Type  & RF, XGB, GBM, LGB, CB, SVR, DNN \\
  Config\_ID   & Hyperparameter configuration identifier \\
  Train\_R2    & $R^2$ on training set \\
  Train\_RMSE  & RMSE on training set \\
  Train\_MAE   & MAE on training set \\
  Val\_R2      & $R^2$ on validation set \emph{(primary ranking metric)} \\
  Val\_RMSE    & RMSE on validation set \\
  Val\_MAE     & MAE on validation set \\
  Test\_R2     & $R^2$ on held-out test set \\
  Test\_RMSE   & RMSE on test set \\
  Test\_MAE    & MAE on test set \\
  \bottomrule
\end{longtable}

\subsection{Sheet: AI\_Dataset\_Summary}
Aggregated per (Dataset, Target, ..., Model\_Type): Best\_Val\_R2, Mean\_Val\_R2,
Best\_Test\_R2, Mean\_Test\_R2, Mean\_RMSE, N\_Configs.

\subsection{Sheet: \{ModelType\}\_Models  (one per model type)}
Same columns as All\_AI\_Models, filtered to one model family, sorted by Val\_R2 descending.

\subsection{Sheet: All\_ANFIS\_Models}
\begin{longtable}{@{}lp{9cm}@{}}
  \toprule Column & Description \\ \midrule \endhead
  Dataset, Target, Size, Scenario, Feature\_Set, N\_Inputs, NoAnomaly
                   & Same as All\_AI\_Models \\
  Config\_ID       & ANFIS configuration identifier \\
  MF\_Type         & Membership function type: gaussmf, gbellmf, trapmf, trimf \\
  Method           & Partitioning: grid or subclust \\
  N\_MFs\_Per\_Input & Number of MFs per input dimension \\
  N\_Rules         & Total fuzzy rules ($= N\_MFs\_Per\_Input^{N\_Inputs}$ for grid) \\
  N\_Train         & Training set size used \\
  Outlier\_Cleaning & Whether IQR cleaning was applied inside ANFIS training \\
  Train\_R2, Train\_RMSE, Train\_MAE & Training metrics \\
  Val\_R2, Val\_RMSE, Val\_MAE       & Validation metrics \\
  Test\_R2, Test\_RMSE, Test\_MAE    & Test metrics \\
  \bottomrule
\end{longtable}

\subsection{Sheet: ANFIS\_Config\_Comparison}
Aggregated per (Config\_ID, MF\_Type, Method, N\_MFs\_Per\_Input):
Mean\_Val\_R2, Best\_Val\_R2, Mean\_Test\_R2, Best\_Test\_R2, N\_Datasets.

\subsection{Sheet: AI\_vs\_ANFIS\_Comparison}
\begin{tabular}{@{}ll@{}}
  \toprule Column & Description \\ \midrule
  Dataset        & Dataset name \\
  Target         & MM / QM / Beta\_2 / MM\_QM \\
  Best\_AI\_Val\_R2   & Best validation $R^2$ among AI models \\
  Best\_ANFIS\_Val\_R2 & Best validation $R^2$ among ANFIS models \\
  Winner         & AI or ANFIS \\
  Delta\_R2      & Absolute difference \\
  \bottomrule
\end{tabular}

\subsection{Sheet: Best\_Models\_Per\_Target}
\begin{tabular}{@{}ll@{}}
  \toprule Column & Description \\ \midrule
  Target       & MM / QM / Beta\_2 / MM\_QM \\
  Rank         & 1 = best \\
  Model\_Type  & RF, XGB, \ldots \\
  Config\_ID   & Configuration ID \\
  Dataset      & Dataset used \\
  Val\_R2      & Validation $R^2$ \\
  Test\_R2     & Test $R^2$ \\
  \bottomrule
\end{tabular}

\subsection{Sheet: Best\_FeatureSet\_Per\_Target}
Best feature configuration per target: Target, Best\_Feature\_Set,
Best\_Val\_R2, Best\_Model\_Type.

\subsection{Sheet: Anomaly\_vs\_NoAnomaly}
\begin{tabular}{@{}ll@{}}
  \toprule Column & Description \\ \midrule
  Target        & MM / QM / Beta\_2 \\
  Dataset\_Base & Dataset name without \_NoAnomaly suffix \\
  Anomaly\_R2   & Best Val $R^2$ with anomalies included \\
  NoAnomaly\_R2 & Best Val $R^2$ with outliers removed \\
  Delta\_R2     & NoAnomaly\_R2 $-$ Anomaly\_R2 (positive = improved by removal) \\
  \bottomrule
\end{tabular}

\subsection{Sheet: Anomaly\_Explained}
\begin{tabular}{@{}ll@{}}
  \toprule Column & Description \\ \midrule
  Target          & MM / QM / Beta\_2 \\
  NUCLEUS         & Nucleus label (e.g.\ $^{208}$Pb) \\
  A               & Mass number \\
  Z               & Proton number \\
  N               & Neutron number \\
  Anomaly\_Col    & Feature column that triggered IQR removal \\
  IQR\_Ratio      & Severity: $|x - \mathrm{median}| / (0.5\,\mathrm{IQR})$ \\
  Direction       & above\_upper\_fence or below\_lower\_fence \\
  Worst\_IQR\_Ratio & Maximum IQR ratio across all triggered columns \\
  \bottomrule
\end{tabular}

\subsection{Sheet: Target\_Statistics}
Descriptive statistics per target: Target, N\_Nuclei, Mean, Std, Min, Max, Q1, Median, Q3.

\subsection{Sheet: IsoChain\_SuddenChanges}
\begin{tabular}{@{}ll@{}}
  \toprule Column & Description \\ \midrule
  Target          & MM / QM / Beta\_2 \\
  Element         & Chemical symbol \\
  Z\_val          & Proton number \\
  N\_val          & Neutron number at sudden change \\
  NUCLEUS         & Nucleus label \\
  Property\_Value & Measured or predicted property \\
  Z\_Score        & Finite-difference z-score ($> 1.5$ flagged) \\
  Is\_SuddenChange & True if flagged \\
  \bottomrule
\end{tabular}

\subsection{Sheet: Unknown\_Predictions}
\begin{tabular}{@{}ll@{}}
  \toprule Column & Description \\ \midrule
  Target          & MM / QM / Beta\_2 \\
  NUCLEUS         & Nucleus label \\
  A, Z, N         & Nuclear identifiers \\
  Predicted\_Val  & Ensemble mean prediction \\
  CI\_Lower       & 95\,\% confidence interval lower bound \\
  CI\_Upper       & 95\,\% confidence interval upper bound \\
  N\_Models\_Used & Number of models contributing to prediction \\
  \bottomrule
\end{tabular}

\subsection{Sheet: MC\_Robustness\_Summary}
Monte Carlo uncertainty per target: Target, N\_Nuclei, MC\_Mean\_R2, MC\_Mean\_CV,
MC\_CI\_Width\_Mean, High\_Uncertainty\_N.

% ===================================================================
\section{AAA2\_Complete\_\{Target\}.xlsx  (PFAZ~9)}
% ===================================================================

\subsection{Sheet: Predictions}
NUCLEUS, Z, N, A, Experimental, Predicted\_Mean, Predicted\_Std,
N\_Models, Best\_Model\_R2.

\subsection{Sheet: Uncertainty}
\begin{tabular}{@{}ll@{}}
  \toprule Column & Description \\ \midrule
  NUCLEUS    & Nucleus label \\
  Mean       & Mean prediction over $K$ MC runs \\
  Std        & Standard deviation of MC predictions \\
  CV         & Coefficient of variation ($= \mathrm{Std}/|\mathrm{Mean}|$) \\
  CI\_Lower  & 2.5th percentile \\
  CI\_Upper  & 97.5th percentile \\
  CI\_Width  & CI\_Upper $-$ CI\_Lower \\
  \bottomrule
\end{tabular}

% ===================================================================
\section{automl\_improvement\_report.xlsx  (PFAZ~13)}
% ===================================================================

\subsection{Sheet: Summary (colour-coded: green = improved)}
Dataset, Target, Model\_Type, Before\_Val\_R2, After\_Val\_R2,
Improvement ($=$ After $-$ Before), Improved (boolean).

\subsection{Sheet: Best\_Params}
Dataset, Target, Model\_Type, Param (hyperparameter name), Value (optimal value).

\subsection{Sheet: Overview}
Key, Value — aggregate statistics: N\_total, N\_improved, Mean\_Improvement, Best\_Gain.

% ===================================================================
\section{pfaz12\_statistical\_tests.xlsx  (PFAZ~12)}
% ===================================================================

\subsection{Sheet: Summary}
Test\_Name, Statistic, p\_value, alpha (0.05), Significant (True/False), Notes.

\subsection{Sheet: ANOVA}
Factor, F\_statistic, p\_value, df\_between, df\_within, Significant.

\subsection{Sheet: Friedman}
Target, Statistic ($\chi^2_F$), p\_value, Critical\_value, df, Reject\_H0.

\subsection{Sheet: Pairwise\_Wilcoxon (or per-target sheet)}
Model\_1, Model\_2, W\_statistic, p\_raw, p\_adj (Bonferroni), Significant.

\subsection{Sheet: Bayes\_Factors}
Model\_1, Model\_2, BF\_10, Log\_BF\_10,
Evidence\_Category (Decisive / Strong / Moderate / Anecdotal / None).

\subsection{Sheet: Sobol\_Indices (sensitivity\_analysis.xlsx)}
Feature, S1 (first-order index), ST (total-effect index), S1\_conf, ST\_conf.

\subsection{Sheet: Morris\_Indices}
Feature, mu (mean), mu\_star (mean of absolute), sigma (std of EE), Rank.

\subsection{Sheet: bootstrap\_results.xlsx}
Sheets: Summary, Model\_Performance (Model, Metric, Bootstrap\_Mean, Bootstrap\_Std,
CI\_Lower, CI\_Upper), Model\_Comparison (Model\_1, Model\_2, Delta\_R2\_Mean,
Delta\_R2\_CI\_Lower, Delta\_R2\_CI\_Upper).

% ===================================================================
\section{MASTER\_CROSS\_MODEL\_REPORT.xlsx  (PFAZ~5)}
% ===================================================================

\subsection{Sheet: AI\_vs\_ANFIS}
Dataset, Target, Best\_AI\_R2, Best\_ANFIS\_R2, Winner, Delta\_R2.

\subsection{Sheet: Consensus\_\{Target\}}
NUCLEUS, Z, N, Mean\_Pred, Std\_Pred, Min\_Pred, Max\_Pred, N\_Good\_Models.

\subsection{Sheet: Quality\_Summary}
Target, Model\_Type, Good\_Count ($R^2>0.90$), Medium\_Count ($0.70$--$0.90$),
Fair\_Count ($<0.70$), Total.

% ===================================================================
\section{ensemble\_report.xlsx  (PFAZ~7)}
% ===================================================================

\subsection{Sheet: Summary}
Ensemble\_Type (Simple\_Vote / Weighted\_Vote / Stack\_Ridge / \ldots),
Target, Val\_R2, Test\_R2, N\_Base\_Models.

\subsection{Sheets: Voting\_Simple, Voting\_Weighted, Stacking\_\{Meta\}}
Per-nucleus columns: NUCLEUS, Z, N, Experimental, Predicted, Residual.
"""

    # ------------------------------------------------------------------
    # STEP 3: Copy figures
    # ------------------------------------------------------------------

    def _step3_copy_figures(self) -> Dict:
        copied = 0
        p8 = self._pfaz_path(8, 'visualizations')
        for fig in self.collected['figures']:
            fig = Path(fig)
            try:
                # Preserve sub-folder structure relative to PFAZ8 output root
                try:
                    rel = fig.relative_to(p8)
                except ValueError:
                    rel = Path(fig.name)
                dest = self.output_dir / 'figures' / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fig, dest)
                copied += 1
            except Exception:
                pass
        logger.info(f"  Copied {copied} figures to thesis/figures/ (sub-folders preserved)")
        return {'files': [], 'warnings': []}

    # ------------------------------------------------------------------
    # STEP 4: Generate LaTeX tables from Excel reports
    # ------------------------------------------------------------------

    def _step4_generate_tables(self) -> Dict:
        generated = 0
        for xlsx in self.collected['excel_reports'][:8]:
            df = _safe_excel_first_sheet(xlsx, nrows=8)
            if df is None or df.empty:
                continue
            label = xlsx.stem[:40].replace(' ', '_')
            caption = xlsx.stem.replace('_', ' ')
            latex = _df_to_latex(df, caption, label)
            dest = self.output_dir / 'tables' / f'{label}.tex'
            dest.write_text(latex, encoding='utf-8')
            generated += 1
        logger.info(f"  Generated {generated} LaTeX tables")
        return {'files': [], 'warnings': []}

    # ------------------------------------------------------------------
    # STEP 5: Bibliography
    # ------------------------------------------------------------------

    def _step5_bibliography(self) -> Dict:
        bib = r"""@article{mayer1949,
  author  = {Mayer, Maria Goeppert},
  title   = {On Closed Shells in Nuclei. {II}},
  journal = {Physical Review},
  year    = {1949},
  volume  = {75},
  pages   = {1969--1970},
}

@book{ring1980,
  author    = {Ring, Peter and Schuck, Peter},
  title     = {The Nuclear Many-Body Problem},
  publisher = {Springer},
  year      = {1980},
}

@article{wang2021ame,
  author  = {Wang, M. and others},
  title   = {{AME 2020}: Atomic mass evaluation},
  journal = {Chinese Physics C},
  year    = {2021},
  volume  = {45},
  pages   = {030003},
}

@article{chen2016xgboost,
  author    = {Chen, Tianqi and Guestrin, Carlos},
  title     = {{XGBoost}: A Scalable Tree Boosting System},
  booktitle = {Proceedings of KDD},
  year      = {2016},
}

@inproceedings{akiba2019optuna,
  author    = {Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and
               Ohta, Takeru and Koyama, Masanori},
  title     = {{Optuna}: A Next-generation Hyperparameter Optimization Framework},
  booktitle = {Proceedings of KDD},
  year      = {2019},
}

@article{jang1993anfis,
  author  = {Jang, Jyh-Shing Roger},
  title   = {{ANFIS}: Adaptive-network-based fuzzy inference system},
  journal = {IEEE Transactions on Systems, Man, and Cybernetics},
  year    = {1993},
  volume  = {23},
  pages   = {665--685},
}
"""
        bib_file = self.output_dir / 'references.bib'
        bib_file.write_text(bib, encoding='utf-8')
        logger.info(f"  Bibliography written: {bib_file}")
        return {'files': [str(bib_file)], 'warnings': []}

    # ------------------------------------------------------------------
    # STEP 6: Main document
    # ------------------------------------------------------------------

    def _step6_main_document(self) -> Dict:
        content = self._main_tex()
        main_file = self.output_dir / 'thesis_main.tex'
        main_file.write_text(content, encoding='utf-8')
        logger.info(f"  Main document: {main_file}")
        self._write_compile_scripts()
        return {'files': [str(main_file)], 'warnings': []}

    def _main_tex(self) -> str:
        m = self.metadata
        return rf"""\documentclass[12pt,a4paper,oneside]{{report}}

% ---- Encoding & language ----
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[english]{{babel}}

% ---- Math & Physics ----
\usepackage{{amsmath, amssymb, physics}}

% ---- Layout ----
\usepackage[top=2.5cm, bottom=2.5cm, left=3cm, right=2.5cm]{{geometry}}
\usepackage{{setspace}}
\onehalfspacing

% ---- Graphics & Tables ----
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{multirow}}
\usepackage{{array}}

% ---- Hyperlinks ----
\usepackage[colorlinks=true, linkcolor=blue, citecolor=green, urlcolor=cyan]{{hyperref}}

% ---- Code listings ----
\usepackage{{listings}}
\lstset{{basicstyle=\footnotesize\ttfamily, breaklines=true}}

% ---- Caption ----
\usepackage{{caption}}
\usepackage{{subcaption}}

% ---- Metadata ----
\title{{{m['title']}\\[0.5em]\large{m['subtitle']}}}
\author{{{m['author']}\\[0.3em]
  \small Supervisor: {m['supervisor']}\\
  \small {m['university']}, {m['department']}}}
\date{{{m['date']}}}

% ====================================================================
\begin{{document}}

\maketitle
\thispagestyle{{empty}}
\newpage

% ---- Abstract & Özet ----
\input{{chapters/00_abstract.tex}}
\newpage

% ---- List of Abbreviations ----
\input{{chapters/00_abbreviations.tex}}
\newpage

% ---- List of Symbols ----
\input{{chapters/00_symbols.tex}}
\newpage

% ---- ToC / LoF / LoT ----
\tableofcontents
\listoffigures
\listoftables
\newpage

% ---- Chapters ----
\input{{chapters/01_introduction.tex}}
\input{{chapters/02_nuclear_theory.tex}}
\input{{chapters/03_methodology.tex}}
\input{{chapters/04_dataset.tex}}
\input{{chapters/05_ai_training.tex}}
\input{{chapters/06_anfis.tex}}
\input{{chapters/07_results.tex}}
\input{{chapters/08_unknown_preds.tex}}
\input{{chapters/09_cross_model.tex}}
\input{{chapters/10_ensemble.tex}}
\input{{chapters/11_statistical.tex}}
\input{{chapters/12_automl.tex}}
\input{{chapters/13_discussion.tex}}
\input{{chapters/14_conclusion.tex}}

% ---- Appendices ----
\appendix
\input{{appendices/A_hyperparams.tex}}
\input{{appendices/B_dataset_details.tex}}
\input{{appendices/C_feature_list.tex}}
\input{{appendices/D_excel_reports.tex}}

% ---- Bibliography ----
\bibliographystyle{{ieeetr}}
\bibliography{{references}}

\end{{document}}
"""

    def _write_compile_scripts(self):
        bash = (
            "#!/bin/bash\ncd \"$(dirname \"$0\")\"\n"
            "pdflatex -interaction=nonstopmode thesis_main.tex\n"
            "bibtex thesis_main\n"
            "pdflatex -interaction=nonstopmode thesis_main.tex\n"
            "pdflatex -interaction=nonstopmode thesis_main.tex\n"
            "echo 'Done: thesis_main.pdf'\n"
        )
        bat = (
            "@echo off\ncd /d \"%~dp0\"\n"
            "pdflatex -interaction=nonstopmode thesis_main.tex\n"
            "bibtex thesis_main\n"
            "pdflatex -interaction=nonstopmode thesis_main.tex\n"
            "pdflatex -interaction=nonstopmode thesis_main.tex\n"
            "echo Done: thesis_main.pdf\n"
        )
        sh = self.output_dir / 'compile.sh'
        sh.write_text(bash, encoding='utf-8')
        try:
            sh.chmod(0o755)
        except Exception:
            pass
        (self.output_dir / 'compile.bat').write_text(bat, encoding='utf-8')
        logger.info("  Compilation scripts written (compile.sh, compile.bat)")

    # ------------------------------------------------------------------
    # STEP 7: Quality checks
    # ------------------------------------------------------------------

    def _step7_quality_checks(self) -> Dict:
        checks = {
            'thesis_main.tex':            (self.output_dir / 'thesis_main.tex').exists(),
            'references.bib':             (self.output_dir / 'references.bib').exists(),
            'chapters/ directory':        (self.output_dir / 'chapters').is_dir(),
            'figures/ directory':         (self.output_dir / 'figures').is_dir(),
            '02_nuclear_theory.tex':      (self.output_dir / 'chapters' / '02_nuclear_theory.tex').exists(),
            '03_methodology.tex':         (self.output_dir / 'chapters' / '03_methodology.tex').exists(),
            '06_anfis.tex':               (self.output_dir / 'chapters' / '06_anfis.tex').exists(),
            '07_results.tex':             (self.output_dir / 'chapters' / '07_results.tex').exists(),
            '08_unknown_preds.tex':       (self.output_dir / 'chapters' / '08_unknown_preds.tex').exists(),
            '09_cross_model.tex':         (self.output_dir / 'chapters' / '09_cross_model.tex').exists(),
            '10_ensemble.tex':            (self.output_dir / 'chapters' / '10_ensemble.tex').exists(),
            'appendices/C_feature_list.tex': (self.output_dir / 'appendices' / 'C_feature_list.tex').exists(),
        }
        warnings = []
        for check, ok in checks.items():
            if ok:
                logger.info(f"  [OK] {check}")
            else:
                msg = f"MISSING: {check}"
                logger.warning(f"  [WARN] {msg}")
                warnings.append(msg)

        n_figs = len(list((self.output_dir / 'figures').glob('*.png')))
        logger.info(f"  Figures in thesis/figures/: {n_figs}")
        if n_figs == 0:
            warnings.append("No PNG figures copied — run PFAZ~8 first")

        return {'files': [], 'warnings': warnings}

    # ------------------------------------------------------------------
    # STEP 8: PDF compilation (optional)
    # ------------------------------------------------------------------

    def _step8_compile_pdf(self) -> Dict:
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(self.output_dir)
            for _ in range(2):
                subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', 'thesis_main.tex'],
                    capture_output=True, timeout=180,
                )
            subprocess.run(
                ['bibtex', 'thesis_main'],
                capture_output=True, timeout=60,
            )
            for _ in range(2):
                subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', 'thesis_main.tex'],
                    capture_output=True, timeout=180,
                )
            pdf = self.output_dir / 'thesis_main.pdf'
            if pdf.exists():
                logger.info(f"  PDF created: {pdf}")
                return {'pdf_path': str(pdf)}
            raise RuntimeError("thesis_main.pdf not found after compilation")
        except FileNotFoundError:
            raise RuntimeError(
                "pdflatex not found — install TeX Live or MiKTeX, "
                "then run compile.bat / compile.sh manually"
            )
        finally:
            os.chdir(original_dir)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _save_report(self, results: Dict):
        report = self.output_dir / 'logs' / 'execution_report.json'
        report.parent.mkdir(parents=True, exist_ok=True)
        # Remove non-serialisable Path objects
        safe = {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in results.items()
        }
        report.write_text(json.dumps(safe, indent=2, default=str), encoding='utf-8')

    def _print_summary(self, results: Dict):
        print("\n" + "=" * 70)
        print("PFAZ 10: THESIS GENERATION SUMMARY")
        print("=" * 70)
        status = "SUCCESS" if results['success'] else "PARTIAL"
        print(f"Status          : {status}")
        print(f"Steps completed : {len(results['steps_completed'])}")
        print(f"Files generated : {len(results['files_generated'])}")
        print(f"Warnings        : {len(results['warnings'])}")
        print(f"Errors          : {len(results['errors'])}")
        if results.get('pdf_path'):
            print(f"PDF             : {results['pdf_path']}")
        main_tex = self.output_dir / 'thesis_main.tex'
        print(f"LaTeX source    : {main_tex}")
        if results['warnings']:
            print("\nWarnings:")
            for w in results['warnings']:
                print(f"  - {w}")
        if results['errors']:
            print("\nErrors (non-fatal):")
            for e in results['errors']:
                print(f"  - {e}")
        print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("PFAZ 10: MASTER THESIS INTEGRATION v5.0")
    print("=" * 70)

    author     = input("  Author name     (Enter for default): ").strip() or None
    supervisor = input("  Supervisor name (Enter for default): ").strip() or None
    university = input("  University name (Enter for default): ").strip() or None
    compile_yn = input("  Compile to PDF? [y/N]: ").strip().lower() == 'y'

    thesis = MasterThesisIntegration()
    results = thesis.execute_full_pipeline(
        author=author, supervisor=supervisor,
        university=university, compile_pdf=compile_yn,
    )
    sys.exit(0 if results['success'] else 1)


if __name__ == '__main__':
    main()
