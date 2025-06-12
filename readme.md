# Lead Prioritization System

A machine learning system for prioritizing leads using reinforcement learning and deep learning techniques.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Run setup script to install dependencies and verify environment
python setup.py
```

### 2. Run Predictions

```bash
# Option 1: Using command line arguments
python predict.py --test_data_path path/to/your/test_data.csv

# Option 2: Edit TEST_DATA_PATH in predict.py and run
python predict.py

# Option 3: Specify custom output directory and top N leads
python predict.py --test_data_path test_data.csv --output_dir results --top_n 50
```

## 📁 File Structure

```
lead-prioritization-system/
├── predict.py              # 🎯 Main prediction script (START HERE)
├── setup.py               # 🔧 Setup and environment verification
├── requirements.txt       # 📦 Python dependencies
├── config.py             # ⚙️ Configuration settings
├── feature_extractors.py # 🔍 Feature extraction classes
├── data_pipeline.py      # ⚡ Data preprocessing pipeline
├── rl_environment.py     # 🎮 RL environment for prioritization
├── model_inference.py    # 🤖 Model loading and inference
├── output_formatter.py   # 📊 Output formatting and saving
├── outputs/              # 📁 Generated results directory
└── README.md            # 📖 This file
```

## 📋 Required Files

Before running predictions, ensure these model files are in the same directory:

- `lead_pipeline.joblib` - Fitted data preprocessing pipeline
- `sac_lead_prioritization_model_1.zip` - Trained SAC reinforcement learning model

## 📊 Input Data Format

Your test CSV file must contain these columns:

- `domain` - Company domain name
- `lead_site` - Lead source site/URL
- `summary` - Text description/summary of the lead
- `relevance_score` - Numerical relevance score

Optional columns (will be automatically removed):
- `response`, `portfolio`, `sale_price`, `reserve`

### Example CSV Format:
```csv
domain,lead_site,summary,relevance_score
example.com,leadgen.io,Great company with AI focus,0.85
techcorp.ai,website.com,Innovative tech startup,0.72
business.net,portal.biz,Traditional business model,0.34
```

## 🎯 Output Files

The system generates several output files in the `outputs/` directory:

### 1. Detailed Results (`lead_prioritization_detailed_TIMESTAMP.csv`)
Complete results with all leads ranked by priority score.

### 2. Top Leads (`top_prioritized_leads_TIMESTAMP.csv`)
Top N prioritized leads for immediate action.

### 3. Summary Report (`prioritization_summary_TIMESTAMP.json`)
Statistical summary and categorization of results.

### 4. CRM Export (`crm_export_TIMESTAMP.csv`)
CRM-friendly format ready for import into sales systems.

## 🎛️ Configuration

Edit `config.py` to customize:

```python
# Priority thresholds
HIGH_PRIORITY_THRESHOLD = 0.8  # Priority scores ≥ 0.8
MEDIUM_PRIORITY_THRESHOLD = 0.5  # Priority scores 0.5-0.8

# Display settings
SHOW_TOP_N_LEADS = 20  # Number of top leads to show/save

# File paths
MODEL_PATH = "sac_lead_prioritization_model_1"
PIPELINE_PATH = "lead_pipeline.joblib"
OUTPUT_DIR = "outputs"
```

## 🔧 System Components

### 1. Feature Extractors (`feature_extractors.py`)
- **LeadSiteFeatureExtractor**: Extracts 22 features from lead site URLs
- **DomainFeatureExtractor**: Extracts 5 features from domain names  
- **DomainLeadSiteAttention**: Neural attention mechanism for feature fusion

### 2. Data Pipeline (`data_pipeline.py`)
- **LeadDataPipeline**: Complete preprocessing pipeline
  - Text embedding generation using Sentence Transformers
  - Feature scaling and normalization
  - Attention-based feature combination
  - Save/load functionality for reuse

### 3. RL Environment (`rl_environment.py`)
- **LeadPrioritizationEnv**: Gymnasium environment for reinforcement learning
  - Reward function based on priority-relevance alignment
  - Continuous action space for priority scores
  - Support for batch inference

### 4. Model Inference (`model_inference.py`)
- **LeadPrioritizationInference**: Handles model loading and prediction
  - SAC model loading and inference
  - Priority score generation
  - Lead categorization and ranking
  - Summary statistics generation

### 5. Output Formatting (`output_formatter.py`)
- **OutputFormatter**: Multiple output format support
  - CSV exports for analysis
  - JSON reports for integration
  - CRM-ready formats
  - Console display formatting

## 📈 Priority Scoring

The system uses a Soft Actor-Critic (SAC) reinforcement learning model to generate priority scores:

- **High Priority (≥0.8)**: Immediate action recommended
- **Medium Priority (0.5-0.8)**: Schedule for follow-up
- **Low Priority (<0.5)**: Monitor or deprioritize

The scoring considers:
- Domain characteristics and tech indicators
- Lead site features and URL patterns  
- Text summary semantic content
- Historical relevance patterns

## 🛠️ Programmatic Usage

For integration into other systems:

```python
from predict import quick_predict

# Run prediction programmatically
results = quick_predict('test_data.csv', top_n=20)

if results:
    prioritized_leads = results['prioritized_leads']
    summary_report = results['summary_report'] 
    top_leads = results['top_leads']
```

## 🐛 Troubleshooting

### Common Issues:

1. **Missing model files**
   ```bash
   python setup.py  # Verify all files are present
   ```

2. **Import errors**
   ```bash
   pip install -r requirements.txt
   ```

3. **CSV format issues**
   - Ensure all required columns are present
   - Check for null values in key columns
   - Verify CSV encoding (UTF-8 recommended)

4. **Memory issues with large datasets**
   - Process data in smaller batches
   - Reduce sentence transformer batch size
   - Use lighter embedding models

### Environment Variables:

Set these to override default paths:
```bash
export LEAD_MODEL_PATH="/path/to/model"
export LEAD_PIPELINE_PATH="/path/to/pipeline.joblib"  
export LEAD_OUTPUT_DIR="/path/to/outputs"
```

## 📊 Performance Notes

- **Processing Speed**: ~100-500 leads per minute (depending on hardware)
- **Memory Usage**: ~2-4GB RAM for typical datasets
- **GPU Acceleration**: Automatic if CUDA available for sentence transformers

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify input data format matches requirements
3. Run `python setup.py` to diagnose environment issues
4. Check console output for detailed error messages

## 🔄 Additional Debugging

If you're still encountering issues, you can run the following command to check the numpy version:

```bash
python -c "import numpy; print(numpy.__version__)"
```

## 🔄 Additional Debugging

If you're still encountering issues, you can run the following command to check the Python executable:

```bash
python -c "import sys; print(sys.executable)"
```