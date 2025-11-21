#!/usr/bin/env python3
"""
PFAZ 10 Complete Package - CLI Interface
=========================================

User-friendly command-line interface for automatic thesis generation.

Usage:
    python pfaz10_complete_package.py --quick      # Quick mode (skip PDF)
    python pfaz10_complete_package.py --full       # Full compilation with PDF
    python pfaz10_complete_package.py --interactive # Interactive mode
    python pfaz10_complete_package.py --qa-only    # Run QA checks only

Author: PFAZ Team
Version: 1.0.0 (100% Complete)
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ThesisCompilationCLI:
    """
    Command-line interface for thesis compilation.
    """

    def __init__(self):
        """Initialize CLI."""
        self.project_dir = Path.cwd()
        self.start_time = None

    def print_banner(self):
        """Print welcome banner."""
        banner = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║           PFAZ 10: AUTOMATIC LATEX THESIS GENERATION SYSTEM               ║
║                                                                           ║
║  Complete ML Pipeline → Professional LaTeX Thesis → PDF                  ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
        print(banner)
        print(f"Working Directory: {self.project_dir}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")

    def quick_mode(self):
        """
        Quick mode: Generate LaTeX without PDF compilation.

        Fast execution for iterative development.
        """
        logger.info("Starting QUICK MODE (LaTeX only, no PDF)")
        logger.info("="*80)

        from pfaz10_master_integration import MasterThesisIntegration

        try:
            master = MasterThesisIntegration(self.project_dir)

            # Steps 1-7 (skip PDF compilation)
            logger.info("[1/7] Collecting data...")
            master._step1_collect_all_data()

            logger.info("[2/7] Generating chapters...")
            master._step2_generate_chapters()

            logger.info("[3/7] Integrating figures...")
            master._step3_integrate_figures()

            logger.info("[4/7] Generating tables...")
            master._step4_generate_tables()

            logger.info("[5/7] Creating bibliography...")
            master._step5_create_bibliography()

            logger.info("[6/7] Generating main document...")
            master._step6_generate_main_document()

            logger.info("[7/7] Running QA checks...")
            qa_results = master._step7_quality_assurance()

            print("\n" + "="*80)
            print("✓ QUICK MODE COMPLETED SUCCESSFULLY!")
            print(f"LaTeX source ready at: {master.output_dir}")
            print(f"QA Results: {qa_results['errors']} errors, {qa_results['warnings']} warnings")
            print("="*80)

            return True

        except Exception as e:
            logger.error(f"Quick mode failed: {e}", exc_info=True)
            return False

    def full_mode(self):
        """
        Full mode: Complete compilation including PDF.

        Executes all 8 steps including PDF compilation.
        """
        logger.info("Starting FULL MODE (Complete compilation with PDF)")
        logger.info("="*80)

        from pfaz10_master_integration import MasterThesisIntegration

        try:
            master = MasterThesisIntegration(self.project_dir)
            success = master.compile_complete_thesis()

            if success:
                print("\n" + "="*80)
                print("✓ FULL COMPILATION COMPLETED SUCCESSFULLY!")
                print(f"PDF available at: {master.output_dir / 'thesis_main.pdf'}")
                print("="*80)
            else:
                print("\n" + "="*80)
                print("⚠ Compilation completed with warnings")
                print(f"LaTeX source at: {master.output_dir / 'thesis_main.tex'}")
                print("="*80)

            return success

        except Exception as e:
            logger.error(f"Full mode failed: {e}", exc_info=True)
            return False

    def interactive_mode(self):
        """
        Interactive mode: Step-by-step with user confirmation.
        """
        print("\n" + "="*80)
        print("INTERACTIVE MODE")
        print("="*80)

        print("\nThis mode allows you to control each step of thesis generation.\n")

        from pfaz10_master_integration import MasterThesisIntegration

        try:
            master = MasterThesisIntegration(self.project_dir)

            steps = [
                ("Collect data from PFAZ 0-9", master._step1_collect_all_data),
                ("Generate chapter content", master._step2_generate_chapters),
                ("Integrate figures", master._step3_integrate_figures),
                ("Generate tables from Excel", master._step4_generate_tables),
                ("Create bibliography", master._step5_create_bibliography),
                ("Generate main document", master._step6_generate_main_document),
                ("Run QA checks", master._step7_quality_assurance),
                ("Compile to PDF", master._step8_compile_pdf)
            ]

            for i, (desc, func) in enumerate(steps, 1):
                print(f"\n[Step {i}/{len(steps)}] {desc}")
                response = input("Execute this step? [Y/n]: ").strip().lower()

                if response in ['', 'y', 'yes']:
                    logger.info(f"Executing step {i}...")
                    result = func()

                    if result is not None and isinstance(result, dict):
                        print(f"  Result: {result}")

                    print("  ✓ Step completed")
                else:
                    print("  ⊘ Step skipped")

            print("\n" + "="*80)
            print("✓ INTERACTIVE SESSION COMPLETED")
            print("="*80)

            return True

        except Exception as e:
            logger.error(f"Interactive mode failed: {e}", exc_info=True)
            return False

    def qa_only_mode(self):
        """
        QA only mode: Run quality assurance checks on existing thesis.
        """
        logger.info("Starting QA ONLY MODE")
        logger.info("="*80)

        from pfaz10_visualization_qa import ThesisQualityAssurance

        output_dir = self.project_dir / 'output' / 'thesis'

        if not output_dir.exists():
            logger.error(f"Thesis directory not found: {output_dir}")
            logger.error("Please run compilation first!")
            return False

        try:
            qa = ThesisQualityAssurance(output_dir)
            results = qa.run_all_checks()

            print("\n" + "="*80)
            print("✓ QA CHECKS COMPLETED")
            print("="*80)

            return True

        except Exception as e:
            logger.error(f"QA mode failed: {e}", exc_info=True)
            return False

    def show_status(self):
        """Show current project status."""
        print("\n" + "="*80)
        print("PROJECT STATUS")
        print("="*80 + "\n")

        # Check directories
        dirs_to_check = [
            ('Reports', self.project_dir / 'reports'),
            ('Visualizations', self.project_dir / 'output' / 'visualizations'),
            ('Thesis Output', self.project_dir / 'output' / 'thesis'),
            ('Data', self.project_dir / 'data')
        ]

        print("Directory Status:")
        for name, path in dirs_to_check:
            exists = "✓" if path.exists() else "✗"
            count = len(list(path.glob('**/*'))) if path.exists() else 0
            print(f"  {exists} {name:20s} ({count} files)")

        # Check for key files
        output_dir = self.project_dir / 'output' / 'thesis'
        key_files = [
            ('Main LaTeX', output_dir / 'thesis_main.tex'),
            ('Bibliography', output_dir / 'references.bib'),
            ('PDF Output', output_dir / 'thesis_main.pdf')
        ]

        print("\nKey Files:")
        for name, path in key_files:
            exists = "✓" if path.exists() else "✗"
            size = f"{path.stat().st_size / 1024:.1f} KB" if path.exists() else "N/A"
            print(f"  {exists} {name:20s} {size}")

        # Check chapters
        chapters_dir = output_dir / 'chapters'
        if chapters_dir.exists():
            chapters = list(chapters_dir.glob('*.tex'))
            print(f"\nChapters: {len(chapters)} files")
            for chapter in sorted(chapters)[:10]:
                print(f"  - {chapter.name}")

        print("\n" + "="*80)

    def run(self, args):
        """
        Run CLI with arguments.

        Args:
            args: Parsed command-line arguments
        """
        self.start_time = datetime.now()
        self.print_banner()

        success = False

        if args.status:
            self.show_status()
            return 0

        elif args.quick:
            success = self.quick_mode()

        elif args.full:
            success = self.full_mode()

        elif args.interactive:
            success = self.interactive_mode()

        elif args.qa_only:
            success = self.qa_only_mode()

        else:
            logger.error("No mode specified! Use --quick, --full, --interactive, or --qa-only")
            return 1

        # Print execution time
        if self.start_time:
            duration = datetime.now() - self.start_time
            print(f"\nExecution time: {duration}")

        return 0 if success else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='PFAZ 10: Automatic LaTeX Thesis Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick mode (no PDF)
  python pfaz10_complete_package.py --quick

  # Full compilation with PDF
  python pfaz10_complete_package.py --full

  # Interactive step-by-step
  python pfaz10_complete_package.py --interactive

  # Run QA checks only
  python pfaz10_complete_package.py --qa-only

  # Show project status
  python pfaz10_complete_package.py --status
        """
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: Generate LaTeX without PDF compilation'
    )

    mode_group.add_argument(
        '--full',
        action='store_true',
        help='Full mode: Complete compilation including PDF'
    )

    mode_group.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode: Step-by-step execution'
    )

    mode_group.add_argument(
        '--qa-only',
        action='store_true',
        help='QA only: Run quality assurance checks on existing thesis'
    )

    mode_group.add_argument(
        '--status',
        action='store_true',
        help='Show current project status'
    )

    # Parse arguments
    args = parser.parse_args()

    # Run CLI
    cli = ThesisCompilationCLI()
    return cli.run(args)


if __name__ == '__main__':
    sys.exit(main())
