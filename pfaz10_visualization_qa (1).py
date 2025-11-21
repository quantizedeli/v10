#!/usr/bin/env python3
"""
PFAZ 10 Visualization and Quality Assurance
============================================

Comprehensive quality assurance system for thesis LaTeX documents.

Features:
- LaTeX syntax validation
- Reference checking (labels, citations, figures)
- Figure integrity verification
- Table validation
- Consistency checks
- Completeness verification

Author: PFAZ Team
Version: 1.0.0 (100% Complete)
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class VisualizationGalleryManager:
    """
    Manages visualization gallery and categorization.
    """

    def __init__(self, vis_dir: Path):
        """
        Initialize gallery manager.

        Args:
            vis_dir: Visualizations directory
        """
        self.vis_dir = vis_dir
        self.figures = []
        self.categories = {
            'training': [],
            'performance': [],
            'comparison': [],
            'ensemble': [],
            'analysis': [],
            'visualization': [],
            'other': []
        }

    def scan_visualizations(self) -> List[Path]:
        """
        Scan all visualization files.

        Returns:
            List of figure paths
        """
        if not self.vis_dir.exists():
            logger.warning(f"Visualization directory not found: {self.vis_dir}")
            return []

        self.figures = list(self.vis_dir.glob('**/*.png'))
        logger.info(f"Found {len(self.figures)} visualization files")

        return self.figures

    def categorize_figures(self):
        """Categorize figures based on filename patterns."""
        for fig in self.figures:
            name = fig.stem.lower()
            categorized = False

            # Training-related
            if any(kw in name for kw in ['training', 'loss', 'epoch', 'convergence']):
                self.categories['training'].append(fig)
                categorized = True

            # Performance-related
            elif any(kw in name for kw in ['performance', 'r2', 'rmse', 'mae', 'metric']):
                self.categories['performance'].append(fig)
                categorized = True

            # Comparison
            elif any(kw in name for kw in ['comparison', 'compare', 'vs', 'versus']):
                self.categories['comparison'].append(fig)
                categorized = True

            # Ensemble
            elif 'ensemble' in name:
                self.categories['ensemble'].append(fig)
                categorized = True

            # Analysis
            elif any(kw in name for kw in ['analysis', 'correlation', 'importance']):
                self.categories['analysis'].append(fig)
                categorized = True

            # Visualization
            elif any(kw in name for kw in ['plot', 'graph', 'chart', 'visual']):
                self.categories['visualization'].append(fig)
                categorized = True

            # Other
            if not categorized:
                self.categories['other'].append(fig)

        # Log statistics
        for cat, figs in self.categories.items():
            if figs:
                logger.info(f"  {cat}: {len(figs)} figures")

    def generate_gallery_appendix(self) -> str:
        """
        Generate LaTeX appendix with figure gallery.

        Returns:
            str: LaTeX code for gallery appendix
        """
        latex = r"""\chapter{Figure Gallery}
\label{app:gallery}

This appendix presents all visualizations generated during the analysis.

"""

        for category, figures in self.categories.items():
            if not figures:
                continue

            latex += f"\\section{{{category.title()} Figures}}\n\n"

            # Create grid layout (2 columns)
            for i in range(0, len(figures), 2):
                batch = figures[i:i+2]

                latex += "\\begin{figure}[H]\n\\centering\n"

                for j, fig in enumerate(batch):
                    latex += f"""\\begin{{subfigure}}{{0.48\\textwidth}}
\\centering
\\includegraphics[width=\\textwidth]{{figures/{fig.name}}}
\\caption{{{fig.stem.replace('_', ' ').title()}}}
\\end{{subfigure}}
"""
                    if j == 0 and len(batch) > 1:
                        latex += "\\hfill\n"

                latex += f"\\caption{{{category.title()} figures batch {i//2 + 1}}}\n"
                latex += "\\end{figure}\n\n"

        return latex


class ThesisQualityAssurance:
    """
    Comprehensive quality assurance for thesis.

    Checks:
    - LaTeX syntax errors
    - Undefined/unused labels
    - Missing figures
    - Broken citations
    - Consistency issues
    - Completeness
    """

    def __init__(self, thesis_dir: Path):
        """
        Initialize QA system.

        Args:
            thesis_dir: Thesis directory containing .tex files
        """
        self.thesis_dir = thesis_dir
        self.errors = []
        self.warnings = []
        self.info = []

    def run_all_checks(self) -> Dict:
        """
        Run all QA checks.

        Returns:
            Dict: Results from all checks
        """
        logger.info("Running comprehensive QA checks...")

        results = {
            'latex_syntax': self.check_latex_syntax_comprehensive(),
            'references': self.check_references_comprehensive(),
            'citations': self.check_citations_comprehensive(),
            'figures': self.check_figure_integrity(),
            'tables': self.check_table_integrity(),
            'consistency': self.check_consistency(),
            'completeness': self.check_completeness()
        }

        # Generate report
        self.generate_qa_report(results)

        return results

    def check_latex_syntax_comprehensive(self) -> Dict:
        """
        Comprehensive LaTeX syntax checking.

        Returns:
            Dict: Syntax check results
        """
        logger.info("Checking LaTeX syntax...")

        errors = []
        warnings = []

        # Find all .tex files
        tex_files = list(self.thesis_dir.glob('**/*.tex'))

        if not tex_files:
            warnings.append("No .tex files found!")
            return {'errors': errors, 'warnings': warnings, 'files_checked': 0}

        for tex_file in tex_files:
            try:
                with open(tex_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                errors.append(f"{tex_file.name}: Failed to read - {e}")
                continue

            # Check 1: Matching braces
            if content.count('{') != content.count('}'):
                errors.append(f"{tex_file.name}: Unmatched braces "
                            f"({{ {content.count('{')} vs }} {content.count('}')}")

            # Check 2: Matching $ for math mode
            dollar_count = content.count('$')
            if dollar_count % 2 != 0:
                errors.append(f"{tex_file.name}: Unmatched $ symbols (count: {dollar_count})")

            # Check 3: Matching environments
            begin_envs = re.findall(r'\\begin\{(\w+)\}', content)
            end_envs = re.findall(r'\\end\{(\w+)\}', content)

            begin_counter = Counter(begin_envs)
            end_counter = Counter(end_envs)

            for env in set(begin_envs + end_envs):
                if begin_counter[env] != end_counter[env]:
                    errors.append(
                        f"{tex_file.name}: Unmatched environment '{env}' "
                        f"(\\begin: {begin_counter[env]}, \\end: {end_counter[env]})"
                    )

            # Check 4: Common LaTeX mistakes
            # Missing closing bracket for \textbf, \textit, etc.
            for cmd in ['textbf', 'textit', 'emph', 'cite', 'ref', 'label']:
                pattern = rf'\\{cmd}\{{[^}}]*$'  # \cmd{ without closing }
                if re.search(pattern, content, re.MULTILINE):
                    warnings.append(f"{tex_file.name}: Possible unclosed \\{cmd}{{...}}")

            # Check 5: TODO or FIXME comments
            if 'TODO' in content or 'FIXME' in content:
                warnings.append(f"{tex_file.name}: Contains TODO/FIXME comments")

        logger.info(f"  Checked {len(tex_files)} files")
        logger.info(f"  Errors: {len(errors)}, Warnings: {len(warnings)}")

        return {
            'errors': errors,
            'warnings': warnings,
            'files_checked': len(tex_files)
        }

    def check_references_comprehensive(self) -> Dict:
        """
        Comprehensive reference checking.

        Returns:
            Dict: Reference check results
        """
        logger.info("Checking references...")

        # Collect all labels and references
        labels_defined = set()
        labels_referenced = set()
        label_locations = {}

        tex_files = list(self.thesis_dir.glob('**/*.tex'))

        for tex_file in tex_files:
            try:
                with open(tex_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                continue

            # Find all \label{...}
            for match in re.finditer(r'\\label\{([^}]+)\}', content):
                label = match.group(1)
                labels_defined.add(label)
                label_locations[label] = tex_file.name

            # Find all \ref{...} and \eqref{...}
            for match in re.finditer(r'\\(?:eq)?ref\{([^}]+)\}', content):
                label = match.group(1)
                labels_referenced.add(label)

        # Check for undefined labels
        undefined = labels_referenced - labels_defined
        unused = labels_defined - labels_referenced

        errors = []
        warnings = []

        if undefined:
            errors.extend([f"Undefined label: {label}" for label in sorted(undefined)])

        if unused:
            warnings.extend([f"Unused label: {label} (defined in {label_locations.get(label, 'unknown')})"
                           for label in sorted(unused)])

        logger.info(f"  Defined labels: {len(labels_defined)}")
        logger.info(f"  Referenced labels: {len(labels_referenced)}")
        logger.info(f"  Undefined: {len(undefined)}, Unused: {len(unused)}")

        return {
            'errors': errors,
            'warnings': warnings,
            'labels_defined': len(labels_defined),
            'labels_referenced': len(labels_referenced),
            'undefined': list(undefined),
            'unused': list(unused)
        }

    def check_citations_comprehensive(self) -> Dict:
        """
        Check bibliography citations.

        Returns:
            Dict: Citation check results
        """
        logger.info("Checking citations...")

        # Find citations in text
        citations_used = set()

        tex_files = list(self.thesis_dir.glob('**/*.tex'))

        for tex_file in tex_files:
            try:
                with open(tex_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                continue

            # Find all \cite{...}
            for match in re.finditer(r'\\cite\{([^}]+)\}', content):
                # Can have multiple citations: \cite{ref1,ref2,ref3}
                refs = match.group(1).split(',')
                citations_used.update(ref.strip() for ref in refs)

        # Find citations defined in .bib file
        citations_defined = set()
        bib_files = list(self.thesis_dir.glob('**/*.bib'))

        for bib_file in bib_files:
            try:
                with open(bib_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                continue

            # Find all @article{key,...}, @book{key,...}, etc.
            for match in re.finditer(r'@\w+\{([^,\s]+)', content):
                citations_defined.add(match.group(1))

        # Check for undefined citations
        undefined = citations_used - citations_defined
        unused = citations_defined - citations_used

        warnings = []

        if undefined:
            warnings.extend([f"Undefined citation: {cite}" for cite in sorted(undefined)])

        if unused:
            warnings.extend([f"Unused bibliography entry: {cite}" for cite in sorted(unused)])

        logger.info(f"  Citations used: {len(citations_used)}")
        logger.info(f"  Citations defined: {len(citations_defined)}")
        logger.info(f"  Undefined: {len(undefined)}, Unused: {len(unused)}")

        return {
            'errors': [],
            'warnings': warnings,
            'citations_used': len(citations_used),
            'citations_defined': len(citations_defined),
            'undefined': list(undefined),
            'unused': list(unused)
        }

    def check_figure_integrity(self) -> Dict:
        """
        Check figure integrity.

        Returns:
            Dict: Figure check results
        """
        logger.info("Checking figures...")

        # Find referenced figures
        figures_referenced = set()

        tex_files = list(self.thesis_dir.glob('**/*.tex'))

        for tex_file in tex_files:
            try:
                with open(tex_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                continue

            # Find all \includegraphics{...}
            for match in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', content):
                fig_path = match.group(1)
                figures_referenced.add(fig_path)

        # Find actual figure files
        figures_dir = self.thesis_dir / 'figures'
        figures_available = set()

        if figures_dir.exists():
            for fig_file in figures_dir.glob('*.*'):
                # Store relative path
                figures_available.add(f"figures/{fig_file.name}")

        # Check for missing files
        missing = []
        for ref in figures_referenced:
            # Build absolute path
            if not ref.startswith('/'):
                fig_path = self.thesis_dir / ref
            else:
                fig_path = Path(ref)

            if not fig_path.exists():
                missing.append(ref)

        # Check for unreferenced figures
        unreferenced = figures_available - figures_referenced

        errors = []
        warnings = []

        if missing:
            errors.extend([f"Missing figure: {fig}" for fig in sorted(missing)])

        if unreferenced:
            warnings.extend([f"Unreferenced figure: {fig}" for fig in sorted(unreferenced)])

        logger.info(f"  Referenced: {len(figures_referenced)}")
        logger.info(f"  Available: {len(figures_available)}")
        logger.info(f"  Missing: {len(missing)}, Unreferenced: {len(unreferenced)}")

        return {
            'errors': errors,
            'warnings': warnings,
            'figures_referenced': len(figures_referenced),
            'figures_available': len(figures_available),
            'missing': missing,
            'unreferenced': list(unreferenced)
        }

    def check_table_integrity(self) -> Dict:
        """
        Check table integrity.

        Returns:
            Dict: Table check results
        """
        logger.info("Checking tables...")

        errors = []
        warnings = []

        tex_files = list(self.thesis_dir.glob('**/*.tex'))

        for tex_file in tex_files:
            try:
                with open(tex_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                continue

            # Find all table environments
            tables = re.findall(r'\\begin\{table\}.*?\\end\{table\}', content, re.DOTALL)

            for table in tables:
                # Check for caption
                if '\\caption{' not in table:
                    warnings.append(f"{tex_file.name}: Table without caption")

                # Check for label
                if '\\label{' not in table:
                    warnings.append(f"{tex_file.name}: Table without label")

                # Check for tabular environment
                if '\\begin{tabular}' not in table:
                    errors.append(f"{tex_file.name}: Table without tabular environment")

        logger.info(f"  Errors: {len(errors)}, Warnings: {len(warnings)}")

        return {
            'errors': errors,
            'warnings': warnings
        }

    def check_consistency(self) -> Dict:
        """
        Check consistency across thesis.

        Returns:
            Dict: Consistency check results
        """
        logger.info("Checking consistency...")

        warnings = []

        # Check naming conventions
        tex_files = list(self.thesis_dir.glob('**/*.tex'))

        # Check for consistent label prefixes
        label_prefixes = Counter()

        for tex_file in tex_files:
            try:
                with open(tex_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                continue

            labels = re.findall(r'\\label\{([^:}]+):', content)
            label_prefixes.update(labels)

        # Expected prefixes
        expected_prefixes = {'ch', 'sec', 'fig', 'tab', 'eq', 'app'}
        found_prefixes = set(label_prefixes.keys())

        unexpected = found_prefixes - expected_prefixes
        if unexpected:
            warnings.append(f"Unexpected label prefixes: {unexpected}")

        return {
            'errors': [],
            'warnings': warnings
        }

    def check_completeness(self) -> Dict:
        """
        Check thesis completeness.

        Returns:
            Dict: Completeness check results
        """
        logger.info("Checking completeness...")

        required_chapters = [
            '00_abstract_en.tex',
            '00_abstract_tr.tex',
            '01_introduction.tex',
            '02_literature.tex',
            '03_methodology.tex',
            '04_results.tex',
            '05_discussion.tex',
            '06_conclusion.tex'
        ]

        missing = []
        empty = []

        chapters_dir = self.thesis_dir / 'chapters'

        if not chapters_dir.exists():
            chapters_dir = self.thesis_dir

        for chapter in required_chapters:
            chapter_path = chapters_dir / chapter

            if not chapter_path.exists():
                missing.append(chapter)
            else:
                # Check if too small (likely empty)
                size = chapter_path.stat().st_size
                if size < 100:  # Less than 100 bytes
                    empty.append(chapter)

        status = 'complete' if not missing and not empty else 'incomplete'

        logger.info(f"  Status: {status}")
        if missing:
            logger.warning(f"  Missing chapters: {missing}")
        if empty:
            logger.warning(f"  Empty chapters: {empty}")

        return {
            'status': status,
            'missing_chapters': missing,
            'empty_chapters': empty
        }

    def generate_qa_report(self, results: Dict):
        """
        Generate comprehensive QA report.

        Args:
            results: Results from all checks
        """
        logger.info("\n" + "="*80)
        logger.info("QUALITY ASSURANCE REPORT")
        logger.info("="*80)

        total_errors = 0
        total_warnings = 0

        for check_name, check_results in results.items():
            logger.info(f"\n{check_name.upper().replace('_', ' ')}:")

            errors = check_results.get('errors', [])
            warnings = check_results.get('warnings', [])

            total_errors += len(errors)
            total_warnings += len(warnings)

            if errors:
                logger.error(f"  Errors: {len(errors)}")
                for error in errors[:5]:  # Show first 5
                    logger.error(f"    - {error}")
                if len(errors) > 5:
                    logger.error(f"    ... and {len(errors) - 5} more")

            if warnings:
                logger.warning(f"  Warnings: {len(warnings)}")
                for warning in warnings[:5]:  # Show first 5
                    logger.warning(f"    - {warning}")
                if len(warnings) > 5:
                    logger.warning(f"    ... and {len(warnings) - 5} more")

            if not errors and not warnings:
                logger.info("  ✓ No issues found")

        logger.info("\n" + "="*80)
        logger.info(f"TOTAL: {total_errors} errors, {total_warnings} warnings")
        logger.info("="*80 + "\n")


def main():
    """Test QA system."""
    thesis_dir = Path('output/thesis')

    print("Testing Quality Assurance System\n")

    qa = ThesisQualityAssurance(thesis_dir)
    results = qa.run_all_checks()

    return 0


if __name__ == '__main__':
    exit(main())
