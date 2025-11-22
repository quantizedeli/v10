"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   PFAZ 10: LATEX FIGURE & TABLE INTEGRATION                  ║
║                                                                              ║
║  Advanced LaTeX integration for figures and tables                          ║
║  - Automatic figure caption generation                                      ║
║  - Multi-figure layouts (subfigures)                                        ║
║  - Excel to LaTeX table conversion                                          ║
║  - Citation and reference management                                        ║
║                                                                              ║
║  Author: Nuclear Physics AI Project                                         ║
║  Version: 2.0.0                                                             ║
║  Date: October 2025                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LaTeXIntegrator:
    """
    Advanced LaTeX Integration System
    
    Features:
    - Automatic figure integration with captions
    - Subfigure layouts (2x2, 3x2, etc.)
    - Excel to LaTeX table conversion
    - Smart caption generation
    - Reference management
    """
    
    def __init__(self, thesis_dir: str = 'output/thesis'):
        """Initialize LaTeX Integrator"""
        self.thesis_dir = Path(thesis_dir)
        self.figures_dir = self.thesis_dir / 'figures'
        self.tables_dir = self.thesis_dir / 'tables'
        
        # Ensure directories exist
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Figure and table counters
        self.figure_counter = 0
        self.table_counter = 0
        
        logger.info("[OK] LaTeX Integrator initialized")
    
    def create_single_figure(self, 
                            image_path: str,
                            caption: str,
                            label: str,
                            width: str = '0.8\\textwidth',
                            placement: str = 'htbp') -> str:
        """
        Create LaTeX code for a single figure
        
        Args:
            image_path: Path to image file
            caption: Figure caption
            label: Figure label for references
            width: Figure width
            placement: Figure placement (htbp)
            
        Returns:
            LaTeX code string
        """
        self.figure_counter += 1
        
        latex_code = f"""\\begin{{figure}}[{placement}]
    \\centering
    \\includegraphics[width={width}]{{{image_path}}}
    \\caption{{{caption}}}
    \\label{{fig:{label}}}
\\end{{figure}}
"""
        return latex_code
    
    def create_subfigures(self,
                         image_paths: List[str],
                         subcaptions: List[str],
                         main_caption: str,
                         label: str,
                         layout: Tuple[int, int] = (2, 2),
                         width: str = '0.45\\textwidth',
                         placement: str = 'htbp') -> str:
        """
        Create LaTeX code for subfigures
        
        Args:
            image_paths: List of image paths
            subcaptions: List of subcaptions
            main_caption: Main figure caption
            label: Main figure label
            layout: (rows, cols) layout
            width: Width of each subfigure
            placement: Figure placement
            
        Returns:
            LaTeX code string
        """
        self.figure_counter += 1
        rows, cols = layout
        
        latex_code = f"""\\begin{{figure}}[{placement}]
    \\centering
"""
        
        for idx, (img_path, subcap) in enumerate(zip(image_paths, subcaptions)):
            latex_code += f"""    \\begin{{subfigure}}{{{width}}}
        \\centering
        \\includegraphics[width=\\textwidth]{{{img_path}}}
        \\caption{{{subcap}}}
        \\label{{fig:{label}_sub{idx+1}}}
    \\end{{subfigure}}
"""
            # Add spacing
            if (idx + 1) % cols != 0:
                latex_code += "    \\hfill\n"
            elif idx < len(image_paths) - 1:
                latex_code += "    \\vspace{0.5cm}\n\n"
        
        latex_code += f"""    \\caption{{{main_caption}}}
    \\label{{fig:{label}}}
\\end{{figure}}
"""
        return latex_code
    
    def excel_to_latex_table(self,
                            excel_file: str,
                            sheet_name: str,
                            caption: str,
                            label: str,
                            placement: str = 'htbp',
                            header_rows: int = 1,
                            format_cols: Optional[List[str]] = None) -> str:
        """
        Convert Excel table to LaTeX
        
        Args:
            excel_file: Path to Excel file
            sheet_name: Sheet name to read
            caption: Table caption
            label: Table label
            placement: Table placement
            header_rows: Number of header rows
            format_cols: Column format (e.g., ['l', 'c', 'r'])
            
        Returns:
            LaTeX table code
        """
        self.table_counter += 1
        
        # Read Excel
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Determine column format
        if format_cols is None:
            format_cols = ['c'] * len(df.columns)
        
        col_format = '|'.join(format_cols)
        
        # Start LaTeX table
        latex_code = f"""\\begin{{table}}[{placement}]
    \\centering
    \\caption{{{caption}}}
    \\label{{tab:{label}}}
    \\begin{{tabular}}{{|{col_format}|}}
        \\hline
"""
        
        # Add header
        header = ' & '.join([self._sanitize_latex(str(col)) for col in df.columns])
        latex_code += f"        {header} \\\\\n        \\hline\\hline\n"
        
        # Add data rows
        for idx, row in df.iterrows():
            row_data = ' & '.join([self._format_cell(val) for val in row])
            latex_code += f"        {row_data} \\\\\n        \\hline\n"
        
        latex_code += """    \\end{tabular}
\\end{table}
"""
        return latex_code
    
    def create_results_table(self,
                           results_dict: Dict[str, Any],
                           caption: str,
                           label: str,
                           placement: str = 'htbp') -> str:
        """
        Create LaTeX table from results dictionary
        
        Args:
            results_dict: Dictionary with results
            caption: Table caption
            label: Table label
            placement: Table placement
            
        Returns:
            LaTeX table code
        """
        self.table_counter += 1
        
        latex_code = f"""\\begin{{table}}[{placement}]
    \\centering
    \\caption{{{caption}}}
    \\label{{tab:{label}}}
    \\begin{{tabular}}{{|l|c|}}
        \\hline
        \\textbf{{Metric}} & \\textbf{{Value}} \\\\
        \\hline\\hline
"""
        
        for key, value in results_dict.items():
            formatted_key = self._sanitize_latex(str(key))
            formatted_value = self._format_cell(value)
            latex_code += f"        {formatted_key} & {formatted_value} \\\\\n        \\hline\n"
        
        latex_code += """    \\end{tabular}
\\end{table}
"""
        return latex_code
    
    def create_comparison_table(self,
                              models: List[str],
                              metrics: Dict[str, List[float]],
                              caption: str,
                              label: str,
                              placement: str = 'htbp') -> str:
        """
        Create model comparison table
        
        Args:
            models: List of model names
            metrics: Dictionary of metric_name -> [values]
            caption: Table caption
            label: Table label
            placement: Table placement
            
        Returns:
            LaTeX table code
        """
        self.table_counter += 1
        
        # Create column format
        n_cols = len(models) + 1
        col_format = 'l' + 'c' * len(models)
        
        latex_code = f"""\\begin{{table}}[{placement}]
    \\centering
    \\caption{{{caption}}}
    \\label{{tab:{label}}}
    \\begin{{tabular}}{{|{col_format}|}}
        \\hline
        \\textbf{{Metric}}"""
        
        # Add model headers
        for model in models:
            latex_code += f" & \\textbf{{{self._sanitize_latex(model)}}}"
        latex_code += " \\\\\n        \\hline\\hline\n"
        
        # Add metric rows
        for metric_name, values in metrics.items():
            latex_code += f"        {self._sanitize_latex(metric_name)}"
            for val in values:
                latex_code += f" & {self._format_cell(val)}"
            latex_code += " \\\\\n        \\hline\n"
        
        latex_code += """    \\end{tabular}
\\end{table}
"""
        return latex_code
    
    def _sanitize_latex(self, text: str) -> str:
        """Sanitize text for LaTeX (escape special characters)"""
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
            '\\': r'\textbackslash{}'
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def _format_cell(self, value: Any) -> str:
        """Format cell value for LaTeX"""
        if isinstance(value, (int, np.integer)):
            return str(value)
        elif isinstance(value, (float, np.floating)):
            if abs(value) < 0.01 and value != 0:
                return f"{value:.2e}"
            else:
                return f"{value:.4f}"
        elif isinstance(value, str):
            return self._sanitize_latex(value)
        else:
            return self._sanitize_latex(str(value))
    
    def create_figure_gallery(self,
                            image_dir: str,
                            pattern: str = '*.png',
                            caption_prefix: str = 'Figure',
                            label_prefix: str = 'fig',
                            cols: int = 2) -> str:
        """
        Create a gallery of all figures in directory
        
        Args:
            image_dir: Directory containing images
            pattern: File pattern (e.g., '*.png')
            caption_prefix: Prefix for captions
            label_prefix: Prefix for labels
            cols: Number of columns
            
        Returns:
            LaTeX code for figure gallery
        """
        image_dir_path = Path(image_dir)
        image_files = sorted(image_dir_path.glob(pattern))
        
        if not image_files:
            logger.warning(f"No images found in {image_dir} with pattern {pattern}")
            return ""
        
        latex_code = ""
        width = f"{0.95/cols:.2f}\\textwidth"
        
        for idx, img_file in enumerate(image_files):
            # Create label and caption from filename
            label = f"{label_prefix}_{img_file.stem}"
            caption = f"{caption_prefix} {idx+1}: {self._generate_caption(img_file.stem)}"
            
            latex_code += self.create_single_figure(
                image_path=f"figures/{img_file.name}",
                caption=caption,
                label=label,
                width=width
            )
            latex_code += "\n"
        
        return latex_code
    
    def _generate_caption(self, filename: str) -> str:
        """Generate human-readable caption from filename"""
        # Remove common prefixes/suffixes
        caption = filename.replace('_', ' ').replace('-', ' ')
        
        # Capitalize words
        caption = ' '.join(word.capitalize() for word in caption.split())
        
        return caption
    
    def save_latex_snippet(self, content: str, filename: str):
        """Save LaTeX code snippet to file"""
        output_file = self.thesis_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"[OK] Saved LaTeX snippet: {filename}")


class BibTeXManager:
    """
    BibTeX Reference Management
    
    Features:
    - Generate BibTeX entries
    - Import from common formats
    - Citation key generation
    - Duplicate detection
    """
    
    def __init__(self, bib_file: str = 'output/thesis/references.bib'):
        """Initialize BibTeX Manager"""
        self.bib_file = Path(bib_file)
        self.bib_file.parent.mkdir(parents=True, exist_ok=True)
        self.entries = []
        
        # Load existing entries if file exists
        if self.bib_file.exists():
            self._load_entries()
        
        logger.info("[OK] BibTeX Manager initialized")
    
    def add_article(self,
                   authors: str,
                   title: str,
                   journal: str,
                   year: int,
                   volume: Optional[str] = None,
                   number: Optional[str] = None,
                   pages: Optional[str] = None,
                   doi: Optional[str] = None,
                   citation_key: Optional[str] = None) -> str:
        """
        Add journal article entry
        
        Returns:
            Generated citation key
        """
        if citation_key is None:
            citation_key = self._generate_citation_key(authors, year)
        
        entry = f"""@article{{{citation_key},
    author = {{{authors}}},
    title = {{{title}}},
    journal = {{{journal}}},
    year = {{{year}}}"""
        
        if volume:
            entry += f",\n    volume = {{{volume}}}"
        if number:
            entry += f",\n    number = {{{number}}}"
        if pages:
            entry += f",\n    pages = {{{pages}}}"
        if doi:
            entry += f",\n    doi = {{{doi}}}"
        
        entry += "\n}\n"
        
        self.entries.append(entry)
        logger.info(f"[OK] Added article: {citation_key}")
        
        return citation_key
    
    def add_book(self,
                authors: str,
                title: str,
                publisher: str,
                year: int,
                edition: Optional[str] = None,
                isbn: Optional[str] = None,
                citation_key: Optional[str] = None) -> str:
        """Add book entry"""
        if citation_key is None:
            citation_key = self._generate_citation_key(authors, year)
        
        entry = f"""@book{{{citation_key},
    author = {{{authors}}},
    title = {{{title}}},
    publisher = {{{publisher}}},
    year = {{{year}}}"""
        
        if edition:
            entry += f",\n    edition = {{{edition}}}"
        if isbn:
            entry += f",\n    isbn = {{{isbn}}}"
        
        entry += "\n}\n"
        
        self.entries.append(entry)
        logger.info(f"[OK] Added book: {citation_key}")
        
        return citation_key
    
    def add_online(self,
                  title: str,
                  url: str,
                  author: Optional[str] = None,
                  year: Optional[int] = None,
                  accessed: Optional[str] = None,
                  citation_key: Optional[str] = None) -> str:
        """Add online resource entry"""
        if citation_key is None:
            key_base = author if author else title.split()[0]
            citation_key = self._generate_citation_key(key_base, year or 2025)
        
        entry = f"""@online{{{citation_key},
    title = {{{title}}},
    url = {{{url}}}"""
        
        if author:
            entry += f",\n    author = {{{author}}}"
        if year:
            entry += f",\n    year = {{{year}}}"
        if accessed:
            entry += f",\n    note = {{Accessed: {accessed}}}"
        
        entry += "\n}\n"
        
        self.entries.append(entry)
        logger.info(f"[OK] Added online: {citation_key}")
        
        return citation_key
    
    def _generate_citation_key(self, author: str, year: int) -> str:
        """Generate citation key from author and year"""
        # Get first author's last name
        first_author = author.split(' and ')[0].split(',')[0].strip()
        # Remove special characters
        first_author = re.sub(r'[^a-zA-Z]', '', first_author)
        
        return f"{first_author.lower()}{year}"
    
    def _load_entries(self):
        """Load existing BibTeX entries"""
        with open(self.bib_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by @ signs
        entries = content.split('@')[1:]
        self.entries = ['@' + entry for entry in entries]
        
        logger.info(f"[OK] Loaded {len(self.entries)} existing entries")
    
    def save(self):
        """Save all entries to BibTeX file"""
        with open(self.bib_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.entries))
        
        logger.info(f"[OK] Saved {len(self.entries)} entries to {self.bib_file}")
    
    def add_nuclear_physics_references(self):
        """Add common nuclear physics references"""
        logger.info("Adding standard nuclear physics references...")
        
        # Key references
        self.add_book(
            authors="Ring, Peter and Schuck, Peter",
            title="The Nuclear Many-Body Problem",
            publisher="Springer",
            year=1980,
            citation_key="ring1980"
        )
        
        self.add_article(
            authors="Bethe, H. A. and Bacher, R. F.",
            title="Nuclear Physics A. Stationary States of Nuclei",
            journal="Reviews of Modern Physics",
            year=1936,
            volume="8",
            pages="82-229",
            citation_key="bethe1936"
        )
        
        self.add_article(
            authors="Weizsäcker, C. F. von",
            title="Zur Theorie der Kernmassen",
            journal="Zeitschrift für Physik",
            year=1935,
            volume="96",
            pages="431-458",
            citation_key="weizsacker1935"
        )
        
        self.add_article(
            authors="Mayer, Maria Goeppert",
            title="On Closed Shells in Nuclei. II",
            journal="Physical Review",
            year=1949,
            volume="75",
            number="12",
            pages="1969",
            citation_key="mayer1949"
        )
        
        logger.info("[OK] Added standard nuclear physics references")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution for testing"""
    print("\n" + "="*80)
    print("PFAZ 10: LATEX INTEGRATION TEST")
    print("="*80)
    
    # Initialize integrator
    integrator = LaTeXIntegrator()
    
    # Test single figure
    fig_code = integrator.create_single_figure(
        image_path="figures/model_performance.png",
        caption="Model performance comparison across different architectures",
        label="model_perf"
    )
    print("\nSingle Figure Code:")
    print(fig_code)
    
    # Test subfigures
    subfig_code = integrator.create_subfigures(
        image_paths=["figures/mae_plot.png", "figures/rmse_plot.png",
                     "figures/r2_plot.png", "figures="loss_plot.png"],
        subcaptions=["MAE", "RMSE", "R² Score", "Training Loss"],
        main_caption="Training metrics across 50 different configurations",
        label="training_metrics",
        layout=(2, 2)
    )
    print("\nSubfigures Code:")
    print(subfig_code)
    
    # Test results table
    results = {
        'Training MAE': 0.234,
        'Validation MAE': 0.289,
        'Test MAE': 0.301,
        'Training Time (hours)': 48,
        'Total Parameters': 15420
    }
    
    table_code = integrator.create_results_table(
        results_dict=results,
        caption="Final model performance metrics",
        label="final_results"
    )
    print("\nResults Table Code:")
    print(table_code)
    
    # Test BibTeX
    bib_manager = BibTeXManager()
    bib_manager.add_nuclear_physics_references()
    bib_manager.save()
    
    print("\n[OK] LaTeX Integration Test Complete")


if __name__ == "__main__":
    main()
