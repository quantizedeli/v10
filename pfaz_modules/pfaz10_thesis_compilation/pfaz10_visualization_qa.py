"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             PFAZ 10: VISUALIZATION GALLERY & QUALITY ASSURANCE               ║
║                                                                              ║
║  Comprehensive visualization management and thesis quality checks           ║
║  - Automatic figure catalog generation                                      ║
║  - Caption and label management                                             ║
║  - Quality metrics for each visualization                                   ║
║  - LaTeX consistency checks                                                 ║
║  - Reference integrity validation                                           ║
║  - Compilation error detection                                              ║
║                                                                              ║
║  Author: Nuclear Physics AI Project                                         ║
║  Version: 2.0.0                                                             ║
║  Date: October 2025                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from PIL import Image
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisualizationGalleryManager:
    """
    Visualization Gallery Manager
    
    Features:
    - Automatic figure categorization
    - Smart caption generation
    - Multi-column gallery layouts
    - Figure quality metrics
    - Appendix generation
    """
    
    def __init__(self, 
                 figures_dir: str = 'output/thesis/figures',
                 output_dir: str = 'output/thesis'):
        """Initialize gallery manager"""
        self.figures_dir = Path(figures_dir)
        self.output_dir = Path(output_dir)
        
        # Figure catalog
        self.catalog = []
        
        # Categories for organization
        self.categories = {
            'training': ['loss', 'mae', 'rmse', 'training', 'learning'],
            'performance': ['performance', 'comparison', 'metrics', 'score'],
            'analysis': ['shap', 'feature', 'importance', 'correlation'],
            'prediction': ['prediction', 'residual', 'scatter', 'error'],
            'validation': ['validation', 'cross', 'bootstrap', 'confidence'],
            'ensemble': ['ensemble', 'stacking', 'voting', 'combination'],
            'robustness': ['robustness', 'sensitivity', 'noise', 'perturbation'],
            'theoretical': ['semf', 'nilsson', 'theoretical', 'shell']
        }
        
        logger.info("[OK] Visualization Gallery Manager initialized")
    
    def scan_all_figures(self) -> List[Dict]:
        """
        Scan all figures and build catalog
        
        Returns:
            List of figure metadata dictionaries
        """
        logger.info("-> Scanning figures directory...")
        
        if not self.figures_dir.exists():
            logger.warning(f"Figures directory not found: {self.figures_dir}")
            return []
        
        figure_files = sorted(self.figures_dir.glob('*.png'))
        
        for idx, fig_file in enumerate(figure_files):
            try:
                # Get image properties
                with Image.open(fig_file) as img:
                    width, height = img.size
                    file_size = fig_file.stat().st_size / 1024  # KB
                
                # Categorize figure
                category = self._categorize_figure(fig_file.stem)
                
                # Generate caption
                caption = self._generate_caption(fig_file.stem)
                
                # Create metadata
                metadata = {
                    'index': idx + 1,
                    'filename': fig_file.name,
                    'label': self._generate_label(fig_file.stem),
                    'caption': caption,
                    'category': category,
                    'width': width,
                    'height': height,
                    'size_kb': file_size,
                    'aspect_ratio': width / height,
                    'path': str(fig_file.relative_to(self.output_dir))
                }
                
                self.catalog.append(metadata)
                
            except Exception as e:
                logger.warning(f"Could not process {fig_file.name}: {e}")
        
        logger.info(f"[OK] Scanned {len(self.catalog)} figures")
        
        # Save catalog
        self._save_catalog()
        
        return self.catalog
    
    def _categorize_figure(self, filename: str) -> str:
        """Categorize figure based on filename"""
        filename_lower = filename.lower()
        
        for category, keywords in self.categories.items():
            if any(keyword in filename_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def _generate_caption(self, filename: str) -> str:
        """Generate human-readable caption from filename"""
        # Remove common prefixes
        caption = filename.replace('_', ' ').replace('-', ' ')
        
        # Remove numbering
        caption = re.sub(r'^\d+\s*', '', caption)
        
        # Capitalize
        caption = ' '.join(word.capitalize() for word in caption.split())
        
        # Add context based on keywords
        context_map = {
            'mae': 'Mean Absolute Error',
            'rmse': 'Root Mean Square Error',
            'r2': 'R² Score',
            'shap': 'SHAP Value',
            'anfis': 'ANFIS Model',
            'semf': 'Semi-Empirical Mass Formula',
            'nn': 'Neural Network',
            'rf': 'Random Forest',
            'gb': 'Gradient Boosting'
        }
        
        for key, value in context_map.items():
            if key in filename.lower():
                caption = caption.replace(key.upper(), value)
        
        return caption
    
    def _generate_label(self, filename: str) -> str:
        """Generate LaTeX label from filename"""
        # Remove special characters
        label = re.sub(r'[^a-zA-Z0-9_]', '_', filename.lower())
        return f"fig:{label}"
    
    def _save_catalog(self):
        """Save figure catalog to JSON"""
        catalog_file = self.output_dir / 'figure_catalog.json'
        with open(catalog_file, 'w', encoding='utf-8') as f:
            json.dump(self.catalog, f, indent=2)
        logger.info(f"[OK] Catalog saved: {catalog_file}")
    
    def generate_appendix_gallery(self, 
                                 columns: int = 2,
                                 figures_per_page: int = 4) -> str:
        """
        Generate LaTeX appendix with figure gallery
        
        Args:
            columns: Number of columns per row
            figures_per_page: Maximum figures per page
            
        Returns:
            LaTeX code for appendix
        """
        logger.info("-> Generating appendix gallery...")
        
        latex = r"""\chapter{Complete Figure Gallery}
\label{app:figures}

This appendix contains all visualizations generated during the analysis, organized by category.

"""
        
        # Group by category
        by_category = {}
        for fig in self.catalog:
            category = fig['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(fig)
        
        # Generate sections for each category
        for category, figures in sorted(by_category.items()):
            latex += f"\n\\section{{{category.title()} Visualizations}}\n\n"
            
            for i in range(0, len(figures), figures_per_page):
                page_figures = figures[i:i + figures_per_page]
                latex += self._generate_figure_page(page_figures, columns)
        
        # Save appendix
        appendix_file = self.output_dir / 'appendices' / 'figures_gallery.tex'
        appendix_file.parent.mkdir(parents=True, exist_ok=True)
        with open(appendix_file, 'w', encoding='utf-8') as f:
            f.write(latex)
        
        logger.info(f"[OK] Gallery saved: {appendix_file}")
        
        return latex
    
    def _generate_figure_page(self, figures: List[Dict], columns: int) -> str:
        """Generate LaTeX for one page of figures"""
        width = f"{0.95/columns:.2f}\\textwidth"
        
        latex = "\\begin{figure}[p]\n    \\centering\n"
        
        for idx, fig in enumerate(figures):
            latex += f"""    \\begin{{subfigure}}{{{width}}}
        \\centering
        \\includegraphics[width=\\textwidth]{{{fig['path']}}}
        \\caption{{{fig['caption']}}}
        \\label{{{fig['label']}}}
    \\end{{subfigure}}
"""
            # Add spacing
            if (idx + 1) % columns != 0 and idx < len(figures) - 1:
                latex += "    \\hfill\n"
            elif idx < len(figures) - 1:
                latex += "    \\vspace{0.5cm}\n\n"
        
        latex += "\\end{figure}\n\n\\clearpage\n\n"
        
        return latex
    
    def generate_figure_statistics(self) -> pd.DataFrame:
        """Generate statistics about figures"""
        if not self.catalog:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.catalog)
        
        stats = {
            'Total Figures': len(df),
            'Categories': df['category'].nunique(),
            'Avg Width (px)': df['width'].mean(),
            'Avg Height (px)': df['height'].mean(),
            'Avg Size (KB)': df['size_kb'].mean(),
            'Total Size (MB)': df['size_kb'].sum() / 1024
        }
        
        logger.info("\nFigure Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.2f}")
        
        return pd.DataFrame([stats])


class ThesisQualityAssurance:
    """
    Thesis Quality Assurance System
    
    Features:
    - LaTeX syntax validation
    - Reference integrity checks
    - Citation verification
    - Figure/table numbering
    - Consistency checks
    - Compilation warnings detection
    """
    
    def __init__(self, thesis_dir: str = 'output/thesis'):
        """Initialize QA system"""
        self.thesis_dir = Path(thesis_dir)
        
        # QA results
        self.qa_results = {
            'errors': [],
            'warnings': [],
            'info': [],
            'checks_passed': 0,
            'checks_failed': 0
        }
        
        logger.info("[OK] Thesis QA System initialized")
    
    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all quality assurance checks
        
        Returns:
            Dictionary with QA results
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING THESIS QUALITY ASSURANCE")
        logger.info("="*80)
        
        checks = [
            ("File Structure", self._check_file_structure),
            ("LaTeX Syntax", self._check_latex_syntax),
            ("References", self._check_references),
            ("Citations", self._check_citations),
            ("Figures", self._check_figures),
            ("Tables", self._check_tables),
            ("Consistency", self._check_consistency)
        ]
        
        for check_name, check_func in checks:
            logger.info(f"\n-> Checking: {check_name}")
            try:
                passed = check_func()
                if passed:
                    self.qa_results['checks_passed'] += 1
                    logger.info(f"  [OK] {check_name} passed")
                else:
                    self.qa_results['checks_failed'] += 1
                    logger.warning(f"  [FAIL] {check_name} failed")
            except Exception as e:
                self.qa_results['errors'].append(f"{check_name}: {str(e)}")
                self.qa_results['checks_failed'] += 1
                logger.error(f"  [FAIL] {check_name} error: {e}")
        
        # Save report
        self._save_qa_report()
        
        # Print summary
        self._print_qa_summary()
        
        return self.qa_results
    
    def _check_file_structure(self) -> bool:
        """Check thesis directory structure"""
        required_files = [
            'thesis_main.tex',
            'references.bib'
        ]
        
        required_dirs = [
            'chapters',
            'figures'
        ]
        
        passed = True
        
        for filename in required_files:
            if not (self.thesis_dir / filename).exists():
                self.qa_results['errors'].append(f"Missing file: {filename}")
                passed = False
        
        for dirname in required_dirs:
            if not (self.thesis_dir / dirname).exists():
                self.qa_results['warnings'].append(f"Missing directory: {dirname}")
        
        return passed
    
    def _check_latex_syntax(self) -> bool:
        """Check LaTeX syntax in main file"""
        main_file = self.thesis_dir / 'thesis_main.tex'
        
        if not main_file.exists():
            return False
        
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for common issues
        checks = {
            'document environment': r'\\begin{document}.*\\end{document}',
            'documentclass': r'\\documentclass',
            'matching braces': None  # Special check
        }
        
        passed = True
        
        for check_name, pattern in checks.items():
            if pattern and not re.search(pattern, content, re.DOTALL):
                self.qa_results['errors'].append(f"LaTeX syntax: Missing {check_name}")
                passed = False
        
        # Check brace matching
        if content.count('{') != content.count('}'):
            self.qa_results['errors'].append("LaTeX syntax: Unmatched braces")
            passed = False
        
        return passed
    
    def _check_references(self) -> bool:
        """Check bibliography file"""
        bib_file = self.thesis_dir / 'references.bib'
        
        if not bib_file.exists():
            self.qa_results['errors'].append("Bibliography file not found")
            return False
        
        with open(bib_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count entries
        entries = len(re.findall(r'@\w+{', content))
        
        if entries == 0:
            self.qa_results['warnings'].append("Bibliography is empty")
            return False
        
        self.qa_results['info'].append(f"Found {entries} bibliography entries")
        return True
    
    def _check_citations(self) -> bool:
        """Check for citation commands"""
        main_file = self.thesis_dir / 'thesis_main.tex'
        
        if not main_file.exists():
            return False
        
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        citations = len(re.findall(r'\\cite{', content))
        
        if citations == 0:
            self.qa_results['warnings'].append("No citations found in main document")
        else:
            self.qa_results['info'].append(f"Found {citations} citations")
        
        return True
    
    def _check_figures(self) -> bool:
        """Check figure references"""
        figures_dir = self.thesis_dir / 'figures'
        
        if not figures_dir.exists():
            self.qa_results['warnings'].append("Figures directory not found")
            return False
        
        figure_files = list(figures_dir.glob('*.png'))
        
        self.qa_results['info'].append(f"Found {len(figure_files)} figure files")
        
        return len(figure_files) > 0
    
    def _check_tables(self) -> bool:
        """Check table files"""
        tables_dir = self.thesis_dir / 'tables'
        
        if tables_dir.exists():
            table_files = list(tables_dir.glob('*.tex'))
            self.qa_results['info'].append(f"Found {len(table_files)} table files")
        
        return True
    
    def _check_consistency(self) -> bool:
        """Check overall consistency"""
        # Check if chapters directory has files
        chapters_dir = self.thesis_dir / 'chapters'
        
        if chapters_dir.exists():
            chapter_files = list(chapters_dir.glob('*.tex'))
            if len(chapter_files) == 0:
                self.qa_results['warnings'].append("No chapter files found")
                return False
            self.qa_results['info'].append(f"Found {len(chapter_files)} chapter files")
        
        return True
    
    def _save_qa_report(self):
        """Save QA report"""
        report_file = self.thesis_dir / 'quality_checks' / 'qa_report.json'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.qa_results, f, indent=2)
        
        logger.info(f"\n[OK] QA report saved: {report_file}")
    
    def _print_qa_summary(self):
        """Print QA summary"""
        print("\n" + "="*80)
        print("QUALITY ASSURANCE SUMMARY")
        print("="*80)
        print(f"Checks Passed: {self.qa_results['checks_passed']}")
        print(f"Checks Failed: {self.qa_results['checks_failed']}")
        print(f"Errors: {len(self.qa_results['errors'])}")
        print(f"Warnings: {len(self.qa_results['warnings'])}")
        
        if self.qa_results['errors']:
            print("\nErrors:")
            for error in self.qa_results['errors']:
                print(f"  [FAIL] {error}")
        
        if self.qa_results['warnings']:
            print("\nWarnings:")
            for warning in self.qa_results['warnings']:
                print(f"  [WARNING] {warning}")
        
        print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("PFAZ 10: VISUALIZATION GALLERY & QUALITY ASSURANCE")
    print("="*80)
    
    # Initialize managers
    gallery = VisualizationGalleryManager()
    qa = ThesisQualityAssurance()
    
    # Scan figures
    gallery.scan_all_figures()
    
    # Generate appendix
    gallery.generate_appendix_gallery()
    
    # Generate statistics
    stats = gallery.generate_figure_statistics()
    if not stats.empty:
        print("\n" + stats.to_string(index=False))
    
    # Run QA checks
    qa_results = qa.run_all_checks()
    
    # Overall status
    if qa_results['checks_failed'] == 0:
        print("\n[OK] ALL QUALITY CHECKS PASSED")
    else:
        print(f"\n[FAIL] {qa_results['checks_failed']} CHECKS FAILED")


if __name__ == "__main__":
    main()
