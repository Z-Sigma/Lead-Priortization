"""
Configuration file for lead prioritization system
"""
import os

# Model and Pipeline Paths
MODEL_PATH = "sac_lead_prioritization_model_1"
PIPELINE_PATH = "lead_pipeline.joblib"

# Output Configuration
OUTPUT_DIR = "outputs"
SAVE_DETAILED_RESULTS = True
SAVE_TOP_LEADS = True
SAVE_SUMMARY_REPORT = True
EXPORT_CRM_FORMAT = True

# Display Configuration
SHOW_TOP_N_LEADS = 20
PRINT_SUMMARY = True
PRINT_TOP_LEADS = True

# Priority Thresholds
HIGH_PRIORITY_THRESHOLD = 0.8
MEDIUM_PRIORITY_THRESHOLD = 0.5

# Required Columns in Test Data
REQUIRED_COLUMNS = [
    'domain',
    'lead_site', 
    'summary',
    'relevance_score'
]

# Optional Columns (will be dropped if present)
OPTIONAL_COLUMNS_TO_DROP = [
    'response',
    'portfolio', 
    'sale_price',
    'reserve'
]

# Sentence Transformer Model
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# Environment Variables (can be overridden)
def get_config():
    """Get configuration with environment variable overrides"""
    config = {
        'model_path': os.getenv('LEAD_MODEL_PATH', MODEL_PATH),
        'pipeline_path': os.getenv('LEAD_PIPELINE_PATH', PIPELINE_PATH),
        'output_dir': os.getenv('LEAD_OUTPUT_DIR', OUTPUT_DIR),
        'show_top_n': int(os.getenv('SHOW_TOP_N_LEADS', SHOW_TOP_N_LEADS)),
        'high_priority_threshold': float(os.getenv('HIGH_PRIORITY_THRESHOLD', HIGH_PRIORITY_THRESHOLD)),
        'medium_priority_threshold': float(os.getenv('MEDIUM_PRIORITY_THRESHOLD', MEDIUM_PRIORITY_THRESHOLD)),
    }
    return config

# Validation function
def validate_test_data(df):
    """Validate that test data has required columns"""
    missing_columns = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing_columns.append(col)
    
    if missing_columns:
        raise ValueError(f"Missing required columns in test data: {missing_columns}")
    
    print(f"✅ Test data validation passed. Found all required columns.")
    return True