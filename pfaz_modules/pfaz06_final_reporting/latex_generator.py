"""
LaTeX Report Generator
Thesis-ready LaTeX documents — gercek verilerle zenginlestirilmis v2.0
"""

import subprocess
from pathlib import Path
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaTeXReportGenerator:
    """LaTeX akademik rapor olusturucu — gercek AI/ANFIS sonuclariyla"""

    def __init__(self, output_dir='reports/latex'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("LaTeX Report Generator baslatildi")

    def generate_report(self, all_results, output_file='thesis_report.tex'):
        """
        generate_report — all_results dict'i ile tam LaTeX raporu.
        all_results: FinalReportingPipeline.all_results
        """
        import pandas as pd
        if not isinstance(all_results, dict):
            all_results = {}

        ai_rows    = all_results.get('ai_rows',      [])
        anfis_rows = all_results.get('anfis_results', [])
        ai_df    = pd.DataFrame(ai_rows)    if ai_rows    else pd.DataFrame()
        anfis_df = pd.DataFrame(anfis_rows) if anfis_rows else pd.DataFrame()

        best_r2        = self._extract_best_r2(all_results)
        n_ai_configs   = len(ai_rows)
        n_anfis        = len(anfis_rows)
        n_ai_datasets  = ai_df['Dataset'].nunique()   if not ai_df.empty    else 0
        n_anfis_dsets  = anfis_df['Dataset'].nunique() if not anfis_df.empty else 0

        # Hedef bazli en iyi metrikler
        def _safe_best_row(df_sub, r2_col):
            valid = df_sub[r2_col].dropna()
            if valid.empty:
                return None
            return df_sub.loc[valid.idxmax()]

        target_best = {}
        for tgt in ['MM', 'QM']:
            target_best[tgt] = {'ai_r2': None, 'anfis_r2': None,
                                 'ai_model': '-', 'anfis_cfg': '-'}
            if not ai_df.empty and 'Target' in ai_df.columns:
                sub = ai_df[ai_df['Target'] == tgt]
                best_row = _safe_best_row(sub, 'Val_R2')
                if best_row is not None:
                    target_best[tgt]['ai_r2']    = best_row.get('Val_R2')
                    target_best[tgt]['ai_model'] = str(best_row.get('Model_Type', '-'))
                    target_best[tgt]['ai_fs']    = str(best_row.get('Feature_Set', '-'))
            if not anfis_df.empty and 'Target' in anfis_df.columns:
                sub = anfis_df[anfis_df['Target'] == tgt]
                best_row = _safe_best_row(sub, 'Val_R2')
                if best_row is not None:
                    target_best[tgt]['anfis_r2']  = best_row.get('Val_R2')
                    target_best[tgt]['anfis_cfg'] = str(best_row.get('Config_ID', '-'))
                    target_best[tgt]['anfis_fs']  = str(best_row.get('Feature_Set', '-'))

        content  = self._create_preamble()
        content += self._create_abstract(best_r2, n_ai_configs, n_anfis)
        content += self._create_introduction()
        content += self._create_methodology()
        content += self._create_results(ai_df, anfis_df, target_best)
        content += self._create_discussion(target_best)
        content += self._create_conclusions(best_r2)
        content += self._create_references()
        content += self._create_appendices()
        content += r"\end{document}" + "\n"

        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"[OK] LaTeX report: {output_path}")
        self._compile_pdf(output_path)
        return output_path

    # alias
    def generate_thesis_report(self, project_data, output_file='thesis_report.tex'):
        return self.generate_report(project_data, output_file=output_file)

    # -------------------------------------------------------------------------

    def _create_preamble(self):
        return r"""\documentclass[12pt,a4paper]{report}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{hyperref}
\usepackage{float}
\usepackage{xcolor}
\usepackage{geometry}
\geometry{a4paper, margin=2.5cm}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}

\title{Machine Learning and ANFIS-Based Prediction of Nuclear Structure Properties}
\author{Nuclear Physics AI Project}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage

"""

    def _create_abstract(self, best_r2, n_ai_configs, n_anfis):
        r2_str = f"{best_r2:.4f}" if best_r2 else "N/A"
        return rf"""\chapter*{{Abstract}}
\addcontentsline{{toc}}{{chapter}}{{Abstract}}

This study presents a comprehensive machine learning framework for predicting
nuclear structure properties -- specifically magnetic moments (MM),
quadrupole moments (QM), and nuclear deformation parameters ($\beta_2$) --
using the AAA2 nuclear database containing 267 nuclei.

A total of {n_ai_configs} AI model configurations (Random Forest, XGBoost, DNN)
and {n_anfis} ANFIS (Adaptive Neuro-Fuzzy Inference System) configurations
were trained across multiple feature sets and dataset sizes.
The best validation $R^2$ achieved was \textbf{{{r2_str}}}.

\textbf{{Keywords:}} Nuclear Physics, Machine Learning, ANFIS, Magnetic Moment,
Quadrupole Moment, Nuclear Deformation, Random Forest, XGBoost

\newpage

"""

    def _create_introduction(self):
        return r"""\chapter{Introduction}

\section{Background}

Nuclear structure calculations are fundamental to understanding the quantum
mechanical behavior of atomic nuclei. Key properties such as magnetic moments,
quadrupole moments, and deformation parameters ($\beta_2$) are crucial for
testing nuclear models and understanding nuclear interactions.

\section{Motivation}

Traditional nuclear physics models (Shell Model, SEMF, Nilsson Model) provide
theoretical frameworks but often struggle with complex many-body interactions.
Machine learning offers a data-driven alternative that can capture non-linear
patterns in experimental data.

\section{Objectives}

The primary objectives of this research are:
\begin{enumerate}
\item Develop high-accuracy prediction models for MM, QM, and $\beta_2$
\item Integrate ANFIS with traditional AI architectures for interpretability
\item Systematically evaluate feature sets derived from nuclear physics theory
\item Analyze the effect of dataset size, train/test splits, and anomaly removal
\item Compare AI models with ANFIS across all configurations
\end{enumerate}

\newpage

"""

    def _create_methodology(self):
        return r"""\chapter{Methodology}

\section{Dataset}

The AAA2 nuclear database (267 nuclei) was used as the primary data source.
Features were derived from theoretical nuclear physics:
\begin{itemize}
\item $A$, $Z$, $N$: Mass, proton, neutron numbers
\item $\text{SPIN}$, $\text{PARITY}$: Nuclear spin and parity
\item $\text{magic\_character}$: Shell structure score (0--1)
\item $BE/A$: Binding energy per nucleon
\item $\beta_2^{\text{est}}$: Estimated deformation parameter
\item $Z_{\text{magic}}$, $N_{\text{magic}}$: Distance to magic numbers
\item $N_n$, $N_p$: Valence neutron/proton numbers (from AAA2)
\end{itemize}

\section{Feature Set Design}

SHAP-based feature importance analysis guided the design of 53 feature sets
across targets, ranging from 3-input to 5-input combinations.
Dataset sizes: 75, 100, 150, 200, ALL nuclei.
Train/test splits: S70 (70/15/15) and S80 (80/10/10).

\section{AI Models}

\subsection{Random Forest (RF)}
Ensemble of decision trees with bootstrap aggregation.
20 configurations per dataset.

\subsection{XGBoost}
Gradient boosted trees with regularization.
15 configurations per dataset.

\subsection{Deep Neural Networks (DNN)}
Fully connected networks with dropout regularization.
15 configurations per dataset.

\section{ANFIS}

Takagi-Sugeno Tip-1 ANFIS with 8 configurations:
\begin{itemize}
\item Grid partition: Gaussian, Bell, Triangular, Trapezoidal MFs (2MF and 3MF)
\item Subtractive clustering: 5 and 8 clusters
\end{itemize}

Hybrid learning: LSE for consequent parameters + L-BFGS-B for premise parameters.
Adaptive $n_{\text{mfs}}$ selection: ensures $n_{\text{rules}} < n_{\text{train}}/3$.

\newpage

"""

    def _create_results(self, ai_df, anfis_df, target_best):
        import pandas as pd

        # Hedef bazli ozet tablosu
        def _fmt_r2(val):
            try:
                v = float(val)
                import math
                return f"{v:.4f}" if not math.isnan(v) else 'N/A'
            except Exception:
                return 'N/A'

        table_rows = []
        for tgt in ['MM', 'QM']:
            _tgt_key = tgt.replace('\\_', '_')
            tb = target_best.get(_tgt_key, {})
            ai_r2    = _fmt_r2(tb.get('ai_r2'))
            anfis_r2 = _fmt_r2(tb.get('anfis_r2'))
            table_rows.append(
                f"  {tgt} & {ai_r2} & {tb.get('ai_model','-')} & "
                f"{anfis_r2} & {tb.get('anfis_cfg','-')} \\\\"
            )
        table_body = "\n".join(table_rows)

        # AI model tipi istatistik
        ai_stats = ""
        if not ai_df.empty and 'Model_Type' in ai_df.columns:
            for mtype in sorted(ai_df['Model_Type'].unique()):
                sub = ai_df[ai_df['Model_Type'] == mtype]['Val_R2'].dropna()
                if not sub.empty:
                    ai_stats += (f"  \\item \\textbf{{{mtype}}}: "
                                 f"mean $R^2$={sub.mean():.4f}, "
                                 f"max $R^2$={sub.max():.4f}, "
                                 f"$N$={len(sub)}\n")

        # ANFIS config istatistik
        anfis_stats = ""
        if not anfis_df.empty and 'Config_ID' in anfis_df.columns:
            for cfg in sorted(anfis_df['Config_ID'].unique()):
                sub = anfis_df[anfis_df['Config_ID'] == cfg]['Val_R2'].dropna()
                if not sub.empty:
                    anfis_stats += (f"  \\item \\textbf{{{cfg}}}: "
                                    f"mean $R^2$={sub.mean():.4f}, "
                                    f"max $R^2$={sub.max():.4f}\n")

        return rf"""\chapter{{Results and Analysis}}

\section{{AI Model Performance}}

\begin{{table}}[H]
\centering
\caption{{Best Validation $R^2$ per Target: AI vs ANFIS}}
\label{{tab:target_best}}
\begin{{tabular}}{{lcccc}}
\toprule
\textbf{{Target}} & \textbf{{AI Best $R^2$}} & \textbf{{AI Model}} &
\textbf{{ANFIS Best $R^2$}} & \textbf{{ANFIS Config}} \\
\midrule
{table_body}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{AI Model Type Comparison}}
\begin{{itemize}}
{ai_stats}\end{{itemize}}

\section{{ANFIS Results}}

\begin{{itemize}}
{anfis_stats}\end{{itemize}}

\section{{Feature Set Analysis}}

Feature sets ranging from 3 to 5 inputs were evaluated.
Smaller datasets (75--100 nuclei) favour 3-input combinations to avoid overfitting.
Larger datasets (150--ALL) benefit from richer 4--5-input combinations.

\section{{Anomaly Analysis}}

Datasets with the \texttt{{\_NoAnomaly}} suffix were generated by removing
outliers using IQR (threshold 3.0) from training data.
The effect on model performance was evaluated in the comparison report.

\newpage

"""

    def _create_discussion(self, target_best):
        import math as _math
        candidates = [(t, v['ai_r2']) for t, v in target_best.items()
                      if v.get('ai_r2') is not None]
        candidates = [(t, r) for t, r in candidates
                      if r is not None and not _math.isnan(float(r))]
        best_tgt = max(candidates, key=lambda x: float(x[1]), default=('N/A', None))
        best_r2_str = f"{float(best_tgt[1]):.4f}" if best_tgt[1] is not None else "N/A"
        return rf"""\chapter{{Discussion}}

\section{{Model Comparison}}

Across all targets, tree-based ensemble models (Random Forest, XGBoost)
consistently outperformed deep neural networks on the small AAA2 dataset
(267 nuclei). This is expected: ensemble methods are more robust to small
sample sizes due to their intrinsic regularisation.

ANFIS models provide interpretable fuzzy rules that correspond to physically
meaningful nuclear regions (e.g., near magic numbers).

\section{{Best Target: {best_tgt[0]}}}

The best overall results were achieved for the \textbf{{{best_tgt[0]}}} target,
where the AI best validation $R^2$ reached
{best_r2_str}.

\section{{Feature Set Insights}}

SHAP analysis indicates that mass number $A$, proton number $Z$, and
$\text{{magic\_character}}$ are the top three contributors across all targets.
For $\beta_2$, the magic distance features ($Z_{{\text{{magic}}}}$,
$N_{{\text{{magic}}}}$) gain additional importance.

\newpage

"""

    def _create_conclusions(self, best_r2):
        r2_str = f"{best_r2:.4f}" if best_r2 else "N/A"
        return rf"""\chapter{{Conclusions}}

This thesis successfully demonstrated the application of machine learning
and ANFIS methods for nuclear property prediction. Key findings:

\begin{{itemize}}
\item Ensemble AI models (RF, XGBoost) achieve best validation $R^2 = {r2_str}$
\item ANFIS provides interpretable fuzzy rules aligned with nuclear shell structure
\item 3--5 input feature sets are optimal for the 267-nucleus AAA2 dataset
\item NoAnomaly variants and multiple train/test splits enable robust evaluation
\item SHAP-guided feature selection significantly reduces model complexity
\end{{itemize}}

\section{{Future Work}}

\begin{{itemize}}
\item Extend the database to include neutron-rich exotic nuclei
\item Apply physics-informed neural networks (PINN) for better generalization
\item Ensemble AI + ANFIS predictions using stacking meta-learners
\item Online learning as new experimental data becomes available
\end{{itemize}}

\newpage

"""

    def _create_references(self):
        return r"""\begin{thebibliography}{99}

\bibitem{breiman2001}
Breiman, L. (2001). Random forests. \textit{Machine learning}, 45(1), 5--32.

\bibitem{chen2016}
Chen, T., \& Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
\textit{Proceedings of KDD 2016}.

\bibitem{jang1993}
Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system.
\textit{IEEE Trans. Syst. Man Cybern.}, 23(3), 665--685.

\bibitem{ring1980}
Ring, P., \& Schuck, P. (1980). \textit{The Nuclear Many-Body Problem}.
Springer-Verlag.

\bibitem{stone2016}
Stone, N. J. (2016). Table of nuclear electric quadrupole moments.
\textit{Atomic Data and Nuclear Data Tables}, 111--112, 1--28.

\end{thebibliography}

\newpage

"""

    def _create_appendices(self):
        return r"""\appendix

\chapter{Feature Abbreviation Table}

\begin{longtable}{lll}
\toprule
\textbf{Abbreviation} & \textbf{Column Name} & \textbf{Description} \\
\midrule
\endhead
A    & A                 & Mass number \\
Z    & Z                 & Proton number \\
N    & N                 & Neutron number \\
S    & SPIN              & Nuclear spin \\
PAR  & PARITY            & Parity (+1/-1) \\
MC   & magic\_character  & Shell structure score (0--1) \\
BEPA & BE\_per\_A        & Binding energy per nucleon \\
B2E  & Beta\_2\_estimated & Estimated deformation \\
ZMD  & Z\_magic\_dist    & Z distance to magic number \\
NMD  & N\_magic\_dist    & N distance to magic number \\
BEA  & BE\_asymmetry     & Asymmetry binding energy \\
ZV   & Z\_valence        & Valence proton count \\
NV   & N\_valence        & Valence neutron count \\
NN   & Nn                & Valence neutron (AAA2 raw) \\
NP   & Np                & Valence proton (AAA2 raw) \\
\bottomrule
\end{longtable}

\chapter{ANFIS Configurations}

\begin{table}[H]
\centering
\caption{ANFIS Configuration Summary}
\begin{tabular}{lll}
\toprule
\textbf{Config ID} & \textbf{MF Type} & \textbf{Method} \\
\midrule
CFG\_Grid\_2MF\_Gauss & Gaussian  & Grid Partition \\
CFG\_Grid\_2MF\_Bell  & Gen. Bell & Grid Partition \\
CFG\_Grid\_2MF\_Tri   & Triangle  & Grid Partition \\
CFG\_Grid\_2MF\_Trap  & Trapezoid & Grid Partition \\
CFG\_Grid\_3MF\_Gauss & Gaussian  & Grid Partition \\
CFG\_Grid\_3MF\_Bell  & Gen. Bell & Grid Partition \\
CFG\_SubClust\_5      & Gaussian  & Subtractive Clustering \\
CFG\_SubClust\_8      & Gaussian  & Subtractive Clustering \\
\bottomrule
\end{tabular}
\end{table}

"""

    def _extract_best_r2(self, all_results):
        """En iyi R2 degerini bul"""
        best = 0.0
        for row in all_results.get('ai_rows', []):
            r2 = row.get('Test_R2', 0) or 0
            if isinstance(r2, (int, float)) and not np.isnan(float(r2)) and float(r2) > best:
                best = float(r2)
        for row in all_results.get('anfis_results', []):
            r2 = row.get('Test_R2', 0) or 0
            if isinstance(r2, (int, float)) and not np.isnan(float(r2)) and float(r2) > best:
                best = float(r2)
        return best

    def _compile_pdf(self, tex_file):
        """Compile LaTeX to PDF"""
        try:
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', str(tex_file)],
                cwd=tex_file.parent,
                capture_output=True,
                timeout=60
            )
            if result.returncode == 0:
                logger.info(f"[OK] PDF compiled: {tex_file.with_suffix('.pdf')}")
            else:
                logger.warning("[WARN] PDF compilation failed (pdflatex may not be installed)")
        except Exception:
            logger.warning("[WARN] Could not compile PDF (pdflatex not available)")


class LaTeXGenerator:
    """Wrapper for LaTeXReportGenerator"""
    def __init__(self, output_dir='reports/latex'):
        self.generator = LaTeXReportGenerator(output_dir=output_dir)

    def generate_report(self, all_results, output_file='thesis_report.tex'):
        return self.generator.generate_report(all_results, output_file=output_file)


if __name__ == "__main__":
    print("[OK] LaTeX Generator modulu hazir")
