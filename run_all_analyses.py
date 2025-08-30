#!/usr/bin/env python3
"""
Master Script to Run All Analyses
Executes the complete pipeline for strategic cancellation detection
Based on capstone project requirements
"""

import os
import sys
import time
from datetime import datetime

def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title.center(80)}")
    print("=" * 80 + "\n")

def run_script(script_name, description):
    """Run a Python script and track execution."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('=' * 60)
    
    start_time = time.time()
    
    try:
        # Import and run the main function
        module_name = script_name.replace('.py', '')
        module = __import__(module_name)
        
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"Warning: No main() function found in {script_name}")
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Completed in {elapsed_time:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"\n✗ Error in {script_name}: {str(e)}")
        return False

def main():
    """Execute the complete analysis pipeline."""
    print_header("STRATEGIC CANCELLATION DETECTION - COMPLETE PIPELINE")
    print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if data file exists
    if not os.path.exists('shadowfax_processed-data-final.csv'):
        print("\n✗ ERROR: Data file 'shadowfax_processed-data-final.csv' not found!")
        print("Please ensure the data file is in the current directory.")
        sys.exit(1)
    
    # Define pipeline scripts in execution order
    pipeline = [
        ("data_processing.py", "Data Processing - Load and parse timestamps"),
        ("labeling_logic.py", "Labeling Logic - Apply behavioral proxy labels"),
        ("feature_engineering.py", "Feature Engineering - Create model features"),
        ("descriptive_statistics.py", "Descriptive Statistics - Generate summary stats"),
        ("threshold_optimization.py", "Threshold Optimization - Find optimal thresholds"),
        ("strategic_bike_issue_prediction_model.py", "Train Random Forest Model"),
        ("model_comparison.py", "Model Comparison - RF vs LR vs XGBoost"),
        ("evaluate_model.py", "Model Evaluation - Performance metrics"),
        ("shap_analysis.py", "SHAP Analysis - Model explainability"),
        ("hypothesis_tests.py", "Hypothesis Testing - Test H1-H5"),
        ("coldstart_model.py", "Cold-Start Model - New rider risk assessment"),
        ("simulate_policies.py", "Policy Simulation - Intervention strategies"),
        ("robustness_analysis.py", "Robustness Analysis - Model stability testing"),
        ("generate_pdf_tables.py", "Generate PDF Tables - Create all tables from paper"),
        ("generate_pdf_figures.py", "Generate PDF Figures - Create all figures from paper")
    ]
    
    # Track execution
    successful_scripts = []
    failed_scripts = []
    
    print("\nPipeline Overview:")
    print("-" * 60)
    for i, (script, desc) in enumerate(pipeline, 1):
        print(f"{i:2d}. {desc}")
    print("-" * 60)
    
    # Execute pipeline
    start_time = time.time()
    
    for script, description in pipeline:
        if os.path.exists(script):
            success = run_script(script, description)
            if success:
                successful_scripts.append(script)
            else:
                failed_scripts.append(script)
                print(f"\nContinuing with remaining scripts...")
        else:
            print(f"\n✗ Script not found: {script}")
            failed_scripts.append(script)
    
    # Final summary
    total_time = time.time() - start_time
    
    print_header("PIPELINE EXECUTION SUMMARY")
    
    print(f"Total execution time: {total_time/60:.1f} minutes")
    print(f"Successful scripts: {len(successful_scripts)}/{len(pipeline)}")
    
    if successful_scripts:
        print("\n✓ Successfully completed:")
        for script in successful_scripts:
            print(f"  - {script}")
    
    if failed_scripts:
        print("\n✗ Failed or missing:")
        for script in failed_scripts:
            print(f"  - {script}")
    
    # List key outputs
    print("\nKey Output Files Generated:")
    print("-" * 40)
    
    expected_outputs = [
        "processed_data.csv",
        "labeled_data.csv",
        "model_ready_features.csv",
        "descriptive_statistics_report.txt",
        "threshold_optimization_report.txt",
        "models/strategic_bike_issue_model.pkl",
        "model_comparison_results.csv",
        "model_evaluation_report.txt",
        "shap_interpretation_report.txt",
        "hypothesis_testing_report.txt",
        "coldstart_policy_recommendations.txt",
        "policy_recommendations.txt",
        "robustness_analysis_report.txt",
        '../tables/table_1_threshold_optimization.csv",
        '../tables/table_2_hypothesis_testing.csv",
        '../tables/table_3_full_model_performance.csv",
        '../tables/table_6_coldstart_performance.csv",
        '../figures/figure_1_venn_diagram.png",
        '../figures/figure_2_probability_curve.png",
        '../figures/figure_3_hourly_distribution.png",
        '../figures/figure_4_distance_effect.png",
        '../figures/figure_5_time_to_cancel.png",
        '../figures/figure_6_shap_visualization.png"
    ]
    
    for output_file in expected_outputs:
        if os.path.exists(output_file):
            size = os.path.getsize(output_file) / 1024  # KB
            print(f"  ✓ {output_file} ({size:.1f} KB)")
        else:
            print(f"  ✗ {output_file} (not found)")
    
    print("\n" + "=" * 80)
    if len(failed_scripts) == 0:
        print("ALL ANALYSES COMPLETED SUCCESSFULLY! 🎉")
        print("Review the generated reports and visualizations for insights.")
    else:
        print("PIPELINE COMPLETED WITH ERRORS")
        print("Please review failed scripts and run them individually if needed.")
    print("=" * 80)

if __name__ == "__main__":
    main()