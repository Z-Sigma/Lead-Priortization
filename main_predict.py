#!/usr/bin/env python3
"""
Main prediction script for Lead Prioritization System

Usage:
    python predict.py --test_data_path path/to/your/test_data.csv
    
    or 
    
    # Edit TEST_DATA_PATH variable below and run:
    python predict.py
"""

import argparse
import pandas as pd
import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our modules
from model_inference import LeadPrioritizationInference
from output_formatter import OutputFormatter
from config import get_config, validate_test_data

# ====================
# CONFIGURATION
# ====================
# You can edit this path or use command line argument
TEST_DATA_PATH = "test_data.csv"

def main():
    """Main function to run lead prioritization predictions"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Lead Prioritization Prediction System")
    parser.add_argument(
        '--test_data_path', 
        type=str, 
        default=TEST_DATA_PATH,
        help='Path to test data CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (optional)'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=None,
        help='Number of top leads to display/save (optional)'
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Override config with command line arguments if provided
    test_data_path = args.test_data_path
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.top_n:
        config['show_top_n'] = args.top_n
    
    print("🚀 Starting Lead Prioritization System")
    print("=" * 50)
    print(f"Test Data Path: {test_data_path}")
    print(f"Model Path: {config['model_path']}")
    print(f"Pipeline Path: {config['pipeline_path']}")
    print(f"Output Directory: {config['output_dir']}")
    print("=" * 50)
    
    try:
        # Step 1: Load and validate test data
        print("\n📂 Step 1: Loading test data...")
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data file not found: {test_data_path}")
        
        df_test = pd.read_csv(test_data_path)
        print(f"   Loaded {len(df_test)} records from {test_data_path}")
        
        # Validate data
        validate_test_data(df_test)
        
        # Step 2: Initialize inference system
        print("\n🤖 Step 2: Initializing inference system...")
        inference_system = LeadPrioritizationInference(
            model_path=config['model_path'],
            pipeline_path=config['pipeline_path']
        )
        
        # Load model and pipeline
        inference_system.load_model_and_pipeline()
        
        # Step 3: Preprocess data
        print("\n⚙️  Step 3: Preprocessing test data...")
        X_test, df_clean = inference_system.preprocess_data(df_test)
        print(f"   Preprocessed data shape: {X_test.shape}")
        
        # Step 4: Generate predictions
        print("\n🎯 Step 4: Generating priority predictions...")
        prioritized_leads = inference_system.predict_priorities(X_test, df_clean)
        print(f"   Generated predictions for {len(prioritized_leads)} leads")
        
        # Step 5: Get top leads and summary
        print("\n📊 Step 5: Analyzing results...")
        top_leads = inference_system.get_top_leads(prioritized_leads, config['show_top_n'])
        summary_report = inference_system.generate_summary_report(prioritized_leads)
        
        # Step 6: Format and save outputs
        print("\n💾 Step 6: Formatting and saving outputs...")
        output_formatter = OutputFormatter(output_dir=config['output_dir'])
        
        # Print results to console
        output_formatter.print_summary_report(summary_report)
        output_formatter.print_top_leads(top_leads, show_count=min(10, len(top_leads)))
        
        # Save all outputs
        saved_files = output_formatter.save_all_outputs(
            prioritized_leads, 
            summary_report, 
            top_leads
        )
        
        # Export CRM-friendly format
        crm_file = output_formatter.export_for_crm(top_leads)
        
        print("\n🎉 SUCCESS! Lead prioritization completed successfully!")
        print("\n📁 Generated Files:")
        for file in saved_files + [crm_file]:
            print(f"   ✓ {file}")
        
        # Quick stats
        print(f"\n📈 Quick Stats:")
        print(f"   • Total leads processed: {len(prioritized_leads)}")
        print(f"   • High priority leads: {summary_report['high_priority_count']}")
        print(f"   • Medium priority leads: {summary_report['medium_priority_count']}")
        print(f"   • Low priority leads: {summary_report['low_priority_count']}")
        print(f"   • Average priority score: {summary_report['average_priority_score']:.3f}")
        
        return prioritized_leads, summary_report, top_leads
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please check that the model files exist:")
        print(f"   - {config['model_path']}")
        print(f"   - {config['pipeline_path']}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("   Please check your input data and model files.")
        sys.exit(1)

def quick_predict(test_data_path, top_n=20):
    """
    Quick prediction function for programmatic use
    
    Args:
        test_data_path (str): Path to test CSV file
        top_n (int): Number of top leads to return
        
    Returns:
        dict: Contains prioritized_leads, summary_report, and top_leads
    """
    # Temporarily set arguments
    sys.argv = ['predict.py', '--test_data_path', test_data_path, '--top_n', str(top_n)]
    
    try:
        prioritized_leads, summary_report, top_leads = main()
        return {
            'prioritized_leads': prioritized_leads,
            'summary_report': summary_report,
            'top_leads': top_leads
        }
    except SystemExit:
        return None

if __name__ == "__main__":
    main()