"""
PFAZ Completeness Checker
=========================

Checklist'e göre her PFAZ modülünün varlığını kontrol eder
"""

import os
from pathlib import Path
import json

# PFAZ modül gereksinimleri (MASTER_PROJECT_CHECKLIST.md'den)
PFAZ_REQUIREMENTS = {
    "PFAZ 0": {
        "modules": ["constants_v1_1_0.py", "config.json"],
        "description": "Temel Hazırlık"
    },
    "PFAZ 1": {
        "modules": [
            "data_loader.py",
            "dataset_generator.py",
            "data_quality_modules.py",
            "qm_filter_manager.py",
            "semf_calculator.py",
            "woods_saxon.py",
            "nilsson_model.py",
            "theoretical_calculations_manager.py"
        ],
        "description": "Dataset Generation"
    },
    "PFAZ 2": {
        "modules": [
            "model_trainer.py",
            "hyperparameter_tuner.py",
            "model_validator.py",
            "parallel_ai_trainer.py"
        ],
        "description": "AI Model Training"
    },
    "PFAZ 3": {
        "modules": [
            "matlab_anfis_trainer.py",
            "anfis_parallel_trainer_v2.py",
            "anfis_config_manager.py",
            "anfis_adaptive_strategy.py",
            "anfis_performance_analyzer.py",
            "anfis_all_nuclei_predictor.py",
            "anfis_robustness_tester.py",
            "anfis_model_saver.py"
        ],
        "description": "ANFIS Training"
    },
    "PFAZ 4": {
        "modules": [
            "unknown_nuclei_predictor.py",
            "unknown_nuclei_splitter.py",
            "all_nuclei_predictor.py",
            "generalization_analyzer.py"
        ],
        "description": "Unknown Nuclei Predictions"
    },
    "PFAZ 5": {
        "modules": [
            "cross_model_evaluator.py",
            "faz5_complete_cross_model.py",
            "best_model_selector.py"
        ],
        "description": "Cross-Model Analysis"
    },
    "PFAZ 6": {
        "modules": [
            "pfaz6_final_reporting.py",
            "comprehensive_excel_reporter.py",
            "excel_formatter.py",
            "excel_charts.py",
            "latex_generator.py"
        ],
        "description": "Final Reporting"
    },
    "PFAZ 7": {
        "modules": [
            "ensemble_model_builder.py",
            "stacking_meta_learner.py",
            "ensemble_evaluator.py",
            "faz7_ensemble_pipeline.py"
        ],
        "outputs": [
            "PFAZ7_Ensemble_Results.xlsx"
        ],
        "description": "Ensemble & Meta-Learning"
    },
    "PFAZ 8": {
        "modules": [
            "visualization_system.py",
            "visualization_advanced_modules.py",
            "ai_visualizer.py",
            "interactive_html_visualizer.py"
        ],
        "description": "Visualization & Dashboard"
    },
    "PFAZ 9": {
        "modules": [
            "aaa2_control_group_complete_v4.py",
            "monte_carlo_simulation_system.py",
            "advanced_analytics_comprehensive.py"
        ],
        "description": "AAA2 & Monte Carlo"
    },
    "PFAZ 10": {
        "modules": [
            "pfaz10_complete_package.py",
            "pfaz10_master_integration.py",
            "pfaz10_content_generator.py",
            "pfaz10_latex_integration.py",
            "pfaz10_visualization_qa.py"
        ],
        "description": "Thesis Compilation"
    },
    "PFAZ 12": {
        "modules": [
            "advanced_analytics_comprehensive.py",
            "statistical_testing_suite.py",
            "bootstrap_confidence_intervals.py",
            "advanced_sensitivity_analysis.py"
        ],
        "description": "Advanced Analytics"
    },
    "PFAZ 13": {
        "modules": [
            "automl_anfis_optimizer.py",
            "automl_hyperparameter_optimizer.py",
            "automl_feature_engineer.py",
            "automl_visualizer.py",
            "automl_logging_reporting_system.py"
        ],
        "description": "AutoML Integration"
    }
}

def check_pfaz_completeness(project_dir="/home/user/nucdatav1"):
    """Check which PFAZ modules exist"""

    project_path = Path(project_dir)
    all_files = set(os.listdir(project_path))

    results = {}

    for pfaz_name, requirements in PFAZ_REQUIREMENTS.items():
        modules = requirements.get("modules", [])
        outputs = requirements.get("outputs", [])
        description = requirements.get("description", "")

        # Check modules
        found_modules = []
        missing_modules = []

        for module in modules:
            if module in all_files:
                found_modules.append(module)
            else:
                # Check for similar names (e.g., automl_optimizer.py vs automl_hyperparameter_optimizer.py)
                similar = [f for f in all_files if module.replace('.py', '') in f or f.replace('.py', '') in module.replace('.py', '')]
                if similar:
                    found_modules.append(f"{module} (similar: {', '.join(similar[:2])})")
                else:
                    missing_modules.append(module)

        # Check outputs
        found_outputs = []
        missing_outputs = []

        for output in outputs:
            if output in all_files:
                found_outputs.append(output)
            else:
                missing_outputs.append(output)

        # Calculate completion percentage
        total_items = len(modules) + len(outputs)
        found_items = len(found_modules) + len(found_outputs)
        completion = (found_items / total_items * 100) if total_items > 0 else 100

        results[pfaz_name] = {
            "description": description,
            "modules": {
                "required": modules,
                "found": found_modules,
                "missing": missing_modules
            },
            "outputs": {
                "required": outputs,
                "found": found_outputs,
                "missing": missing_outputs
            },
            "completion_percentage": completion,
            "status": "[SUCCESS] Complete" if completion == 100 else "[WARNING] Incomplete"
        }

    return results

def print_report(results):
    """Print formatted report"""

    print("=" * 80)
    print("PFAZ COMPLETENESS REPORT")
    print("=" * 80)
    print()

    for pfaz_name, data in results.items():
        status_icon = "[SUCCESS]" if data["completion_percentage"] == 100 else "[WARNING]"

        print(f"{status_icon} {pfaz_name}: {data['description']}")
        print(f"   Completion: {data['completion_percentage']:.1f}%")

        # Missing modules
        if data["modules"]["missing"]:
            print(f"   [ERROR] Missing modules ({len(data['modules']['missing'])}):")
            for module in data["modules"]["missing"]:
                print(f"      - {module}")

        # Missing outputs
        if data["outputs"]["missing"]:
            print(f"   [ERROR] Missing outputs ({len(data['outputs']['missing'])}):")
            for output in data["outputs"]["missing"]:
                print(f"      - {output}")

        # Found modules (if incomplete)
        if data["completion_percentage"] < 100 and data["modules"]["found"]:
            print(f"   [OK] Found modules ({len(data['modules']['found'])}):")
            for module in data["modules"]["found"][:3]:  # Show first 3
                print(f"      - {module}")
            if len(data["modules"]["found"]) > 3:
                print(f"      ... and {len(data['modules']['found']) - 3} more")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    complete_pfaz = [name for name, data in results.items() if data["completion_percentage"] == 100]
    incomplete_pfaz = [name for name, data in results.items() if data["completion_percentage"] < 100]

    print(f"[SUCCESS] Complete PFAZ: {len(complete_pfaz)}/{len(results)}")
    for pfaz in complete_pfaz:
        print(f"   - {pfaz}")

    print()
    print(f"[WARNING] Incomplete PFAZ: {len(incomplete_pfaz)}/{len(results)}")
    for pfaz in incomplete_pfaz:
        completion = results[pfaz]["completion_percentage"]
        print(f"   - {pfaz} ({completion:.1f}%)")

    print()
    print("=" * 80)

def save_report_json(results, output_path="/home/user/nucdatav1/pfaz_completeness_report.json"):
    """Save report to JSON"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Report saved to: {output_path}")

if __name__ == "__main__":
    results = check_pfaz_completeness()
    print_report(results)
    save_report_json(results)
