#!/usr/bin/env python3
"""
PFAZ 10 LaTeX Integration System
=================================

Advanced LaTeX integration for figures, tables, and bibliography.

Features:
- Single figure integration
- Multi-panel subfigures (2x2, 3x3, etc.)
- Excel to LaTeX table conversion (ADVANCED with booktabs)
- Smart caption generation
- BibTeX management

Author: PFAZ Team
Version: 1.0.0 (100% Complete)
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LaTeXIntegrator:
    """
    Advanced LaTeX integration system.

    Provides tools for professional figure and table integration.
    """

    def __init__(self):
        """Initialize LaTeX integrator."""
        self.figure_counter = 0
        self.table_counter = 0

    def create_single_figure(self,
                           image_path: str,
                           caption: str,
                           label: str,
                           width: float = 0.8) -> str:
        """
        Create LaTeX code for a single figure.

        Args:
            image_path: Path to image file
            caption: Figure caption
            label: Figure label (without 'fig:' prefix)
            width: Width as fraction of textwidth

        Returns:
            str: LaTeX code for figure
        """
        self.figure_counter += 1

        latex = f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width={width}\\textwidth]{{{image_path}}}
\\caption{{{caption}}}
\\label{{fig:{label}}}
\\end{{figure}}
"""
        return latex

    def create_subfigures(self,
                         image_paths: List[str],
                         captions: List[str],
                         main_caption: str,
                         main_label: str,
                         layout: Tuple[int, int] = (2, 2)) -> str:
        """
        Create multi-panel subfigures.

        Args:
            image_paths: List of image paths
            captions: List of subcaptions
            main_caption: Main figure caption
            main_label: Main figure label
            layout: (rows, cols) layout

        Returns:
            str: LaTeX code for subfigures
        """
        rows, cols = layout
        width = 0.95 / cols  # Calculate width per subfigure

        latex = "\\begin{figure}[H]\n\\centering\n"

        for i, (img_path, caption) in enumerate(zip(image_paths, captions)):
            if i > 0 and i % cols == 0:
                latex += "\\\\\n"  # New row

            latex += f"""\\begin{{subfigure}}{{{width}\\textwidth}}
\\centering
\\includegraphics[width=\\textwidth]{{{img_path}}}
\\caption{{{caption}}}
\\label{{fig:{main_label}_{i+1}}}
\\end{{subfigure}}
"""

            if (i + 1) % cols != 0 and i < len(image_paths) - 1:
                latex += "\\hfill\n"

        latex += f"""
\\caption{{{main_caption}}}
\\label{{fig:{main_label}}}
\\end{{figure}}
"""

        self.figure_counter += 1
        return latex

    def excel_to_latex_table_advanced(self,
                                      df: pd.DataFrame,
                                      caption: str = '',
                                      label: str = '',
                                      style: str = 'booktabs',
                                      highlight_best: bool = True,
                                      highlight_metric: str = 'r2') -> str:
        """
        Convert pandas DataFrame to professional LaTeX table.

        Features:
        - booktabs formatting
        - Automatic highlighting of best values
        - Smart numeric formatting
        - Scientific notation for small numbers
        - Bold formatting for best values

        Args:
            df: Pandas DataFrame
            caption: Table caption
            label: Table label
            style: 'booktabs' or 'standard'
            highlight_best: Whether to bold best values
            highlight_metric: Which metric to use for highlighting

        Returns:
            str: Professional LaTeX table
        """
        self.table_counter += 1

        # Clean DataFrame
        df = df.dropna(how='all')  # Remove empty rows
        df = df.fillna('-')  # Replace NaN with dash

        # Detect numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Find best values for highlighting
        best_values = {}
        if highlight_best:
            for col in numeric_cols:
                col_lower = col.lower()

                # Higher is better
                if any(metric in col_lower for metric in ['r2', 'r²', 'accuracy', 'precision', 'recall', 'f1']):
                    best_values[col] = df[col].max()

                # Lower is better
                elif any(metric in col_lower for metric in ['rmse', 'mae', 'mse', 'error', 'loss']):
                    best_values[col] = df[col].min()

        # Start LaTeX table
        latex = "\\begin{table}[H]\n\\centering\n"

        if caption:
            latex += f"\\caption{{{caption}}}\n"
        if label:
            latex += f"\\label{{tab:{label}}}\n"

        # Column alignment
        n_cols = len(df.columns)
        col_align = 'l' + 'c' * (n_cols - 1)  # First left, others centered

        latex += f"\\begin{{tabular}}{{{col_align}}}\n"

        # Header lines
        if style == 'booktabs':
            latex += "\\toprule\n"
        else:
            latex += "\\hline\n"

        # Column headers
        headers = [f"\\textbf{{{col}}}" for col in df.columns]
        latex += ' & '.join(headers) + " \\\\\n"

        if style == 'booktabs':
            latex += "\\midrule\n"
        else:
            latex += "\\hline\n"

        # Data rows
        for idx, row in df.iterrows():
            formatted_row = []

            for col in df.columns:
                val = row[col]

                # Handle missing values
                if val == '-' or pd.isna(val):
                    formatted_row.append('-')
                    continue

                # Format numeric values
                if col in numeric_cols and isinstance(val, (int, float)):
                    # Determine formatting
                    if abs(val) < 0.001:
                        formatted_val = f"{val:.2e}"  # Scientific notation
                    elif abs(val) < 0.01:
                        formatted_val = f"{val:.4f}"
                    elif abs(val) < 1:
                        formatted_val = f"{val:.4f}"
                    elif abs(val) < 100:
                        formatted_val = f"{val:.3f}"
                    else:
                        formatted_val = f"{val:.1f}"

                    # Highlight best values
                    if col in best_values:
                        if abs(val - best_values[col]) < 1e-6:  # Floating point comparison
                            formatted_val = f"\\textbf{{{formatted_val}}}"

                    formatted_row.append(formatted_val)
                else:
                    # String values
                    formatted_row.append(str(val))

            latex += ' & '.join(formatted_row) + " \\\\\n"

        # Footer
        if style == 'booktabs':
            latex += "\\bottomrule\n"
        else:
            latex += "\\hline\n"

        latex += "\\end{tabular}\n\\end{table}\n"

        return latex

    def generate_smart_caption(self,
                              fig_type: str,
                              model_name: Optional[str] = None,
                              target: Optional[str] = None) -> str:
        """
        Generate smart caption based on figure type.

        Args:
            fig_type: Type of figure (training, performance, comparison, etc.)
            model_name: Optional model name
            target: Optional target property

        Returns:
            str: Generated caption
        """
        captions = {
            'training': f"Training convergence for {model_name or 'model'}" +
                       (f" on {target} prediction" if target else ""),

            'performance': f"Performance metrics for {model_name or 'model'}" +
                          (f" predicting {target}" if target else ""),

            'comparison': f"Model comparison" +
                         (f" for {target} prediction" if target else ""),

            'ensemble': f"Ensemble performance" +
                       (f" on {target} property" if target else ""),

            'residuals': f"Residual plot for {model_name or 'model'}" +
                        (f" - {target} target" if target else ""),

            'feature_importance': f"Feature importance for {model_name or 'model'}" +
                                 (f" ({target})" if target else ""),

            'cross_model': "Cross-model performance analysis",

            'hyperparameter': f"Hyperparameter optimization for {model_name or 'model'}"
        }

        return captions.get(fig_type, "Figure caption")

    def create_table_from_dict(self,
                               data: Dict,
                               caption: str,
                               label: str) -> str:
        """
        Create LaTeX table from dictionary.

        Args:
            data: Dictionary with data
            caption: Table caption
            label: Table label

        Returns:
            str: LaTeX table
        """
        # Convert dict to DataFrame
        df = pd.DataFrame(list(data.items()), columns=['Parameter', 'Value'])

        return self.excel_to_latex_table_advanced(
            df,
            caption=caption,
            label=label,
            highlight_best=False
        )


class BibTeXManager:
    """
    BibTeX bibliography manager.

    Manages references and exports to .bib format.
    """

    def __init__(self):
        """Initialize BibTeX manager."""
        self.entries = []
        self.entry_counter = 0

    def add_article(self,
                   authors: str,
                   title: str,
                   journal: str,
                   year: int,
                   volume: Optional[str] = None,
                   pages: Optional[str] = None,
                   doi: Optional[str] = None) -> str:
        """
        Add journal article reference.

        Args:
            authors: Author names
            title: Article title
            journal: Journal name
            year: Publication year
            volume: Volume number
            pages: Page numbers
            doi: DOI

        Returns:
            str: Citation key
        """
        self.entry_counter += 1
        key = f"ref{self.entry_counter:03d}"

        entry = f"""@article{{{key},
  author = {{{authors}}},
  title = {{{title}}},
  journal = {{{journal}}},
  year = {{{year}}}"""

        if volume:
            entry += f",\n  volume = {{{volume}}}"
        if pages:
            entry += f",\n  pages = {{{pages}}}"
        if doi:
            entry += f",\n  doi = {{{doi}}}"

        entry += "\n}\n"

        self.entries.append(entry)
        return key

    def add_book(self,
                authors: str,
                title: str,
                publisher: str,
                year: int,
                edition: Optional[str] = None) -> str:
        """
        Add book reference.

        Args:
            authors: Author names
            title: Book title
            publisher: Publisher name
            year: Publication year
            edition: Edition

        Returns:
            str: Citation key
        """
        self.entry_counter += 1
        key = f"book{self.entry_counter:03d}"

        entry = f"""@book{{{key},
  author = {{{authors}}},
  title = {{{title}}},
  publisher = {{{publisher}}},
  year = {{{year}}}"""

        if edition:
            entry += f",\n  edition = {{{edition}}}"

        entry += "\n}\n"

        self.entries.append(entry)
        return key

    def add_inproceedings(self,
                         authors: str,
                         title: str,
                         booktitle: str,
                         year: int,
                         pages: Optional[str] = None) -> str:
        """
        Add conference proceedings reference.

        Args:
            authors: Author names
            title: Paper title
            booktitle: Conference/proceedings name
            year: Year
            pages: Pages

        Returns:
            str: Citation key
        """
        self.entry_counter += 1
        key = f"conf{self.entry_counter:03d}"

        entry = f"""@inproceedings{{{key},
  author = {{{authors}}},
  title = {{{title}}},
  booktitle = {{{booktitle}}},
  year = {{{year}}}"""

        if pages:
            entry += f",\n  pages = {{{pages}}}"

        entry += "\n}\n"

        self.entries.append(entry)
        return key

    def save(self, output_file: str = 'references.bib'):
        """
        Save bibliography to .bib file.

        Args:
            output_file: Output filename
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in self.entries:
                f.write(entry)
                f.write('\n')

        logger.info(f"Saved {len(self.entries)} references to {output_file}")


def create_sample_bibliography() -> BibTeXManager:
    """
    Create sample bibliography with key ML and nuclear physics references.

    Returns:
        BibTeXManager: Manager with sample references
    """
    bib = BibTeXManager()

    # Machine Learning references
    bib.add_article(
        "Breiman, Leo",
        "Random Forests",
        "Machine Learning",
        2001,
        volume="45",
        pages="5-32"
    )

    bib.add_inproceedings(
        "Chen, Tianqi and Guestrin, Carlos",
        "XGBoost: A Scalable Tree Boosting System",
        "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining",
        2016,
        pages="785-794"
    )

    bib.add_book(
        "Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron",
        "Deep Learning",
        "MIT Press",
        2016
    )

    bib.add_article(
        "Jang, Jyh-Shing Roger",
        "ANFIS: Adaptive-Network-Based Fuzzy Inference System",
        "IEEE Transactions on Systems, Man, and Cybernetics",
        1993,
        volume="23",
        pages="665-685"
    )

    # Nuclear Physics references
    bib.add_article(
        "Gazula, S. and Clark, J. W. and Bohr, H.",
        "Learning and prediction of nuclear stability by neural networks",
        "Nuclear Physics A",
        1992,
        volume="540",
        pages="1-26"
    )

    bib.add_article(
        "Utama, R. and Piekarewicz, J. and Prosper, H. B.",
        "Nuclear mass predictions for the crustal composition of neutron stars: A Bayesian neural network approach",
        "Physical Review C",
        2016,
        volume="93",
        pages="014311"
    )

    bib.add_article(
        "Niu, Z. M. and Liang, H. Z.",
        "Nuclear mass predictions based on Bayesian neural network approach with pairing and shell effects",
        "Physics Letters B",
        2018,
        volume="778",
        pages="48-53"
    )

    bib.add_article(
        "Neufcourt, L. and Cao, Y. and Nazarewicz, W. and Viens, F.",
        "Bayesian approach to model-based extrapolation of nuclear observables",
        "Physical Review C",
        2018,
        volume="98",
        pages="034318"
    )

    bib.add_article(
        "Wu, X. H. and Zhao, P. W.",
        "Machine learning for nuclear masses towards neutron drip line",
        "Physical Review C",
        2020,
        volume="101",
        pages="051301"
    )

    bib.add_article(
        "Ma, C. and Niu, Z. and Zhao, Y.",
        "Nuclear charge radii: machine learning approach",
        "Chinese Physics C",
        2015,
        volume="39",
        pages="104102"
    )

    logger.info(f"Created sample bibliography with {len(bib.entries)} entries")

    return bib


def main():
    """Test LaTeX integration."""
    print("Testing LaTeX Integration System\n")

    integrator = LaTeXIntegrator()

    # Test 1: Single figure
    print("Test 1: Single Figure")
    single_fig = integrator.create_single_figure(
        "figures/test.png",
        "Test figure caption",
        "test_figure"
    )
    print(f"Generated {len(single_fig)} chars of LaTeX\n")

    # Test 2: Table from DataFrame
    print("Test 2: Advanced Table")
    df = pd.DataFrame({
        'Model': ['RF', 'XGBoost', 'DNN', 'SVR', 'ANFIS'],
        'R²': [0.92, 0.95, 0.91, 0.89, 0.90],
        'RMSE': [0.085, 0.072, 0.089, 0.095, 0.092],
        'MAE': [0.065, 0.058, 0.070, 0.078, 0.073]
    })

    table = integrator.excel_to_latex_table_advanced(
        df,
        caption="Model Performance Comparison",
        label="model_performance",
        highlight_best=True
    )
    print(f"Generated table:\n{table}\n")

    # Test 3: Bibliography
    print("Test 3: Bibliography")
    bib = create_sample_bibliography()
    bib.save('test_references.bib')
    print(f"Created bibliography with {len(bib.entries)} entries")

    return 0


if __name__ == '__main__':
    exit(main())
