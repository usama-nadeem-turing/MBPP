# Model Runs Directory

This directory contains the results from multiple model inference runs on the MBPP dataset. Each numbered subdirectory represents a separate model run with different configurations or parameters.

## 📁 Directory Structure

```
ModelRuns/
├── 01/                    # Model run 1
├── 02/                    # Model run 2
├── 03/                    # Model run 3
├── 04/                    # Model run 4
├── 05/                    # Model run 5
├── 06/                    # Model run 6
├── 07/                    # Model run 7
├── 08/                    # Model run 8
├── 09/                    # Model run 9
├── 10/                    # Model run 10
├── 11/                    # Model run 11
├── 12/                    # Model run 12
├── analysis/              # Aggregated analysis across all runs
├── datasets_for_FT/       # Fine-tuning datasets created from results
└── pass@k.csv            # Pass@k metrics summary
```

## 📊 Individual Run Structure

Each numbered directory (01-12) contains:

- **`mbpp_results_final_test.json`** - Model responses for test split
- **`mbpp_results_final_train.json`** - Model responses for train split  
- **`mbpp_results_final_validation.json`** - Model responses for validation split
- **`analysis_final_test.json`** - Analysis summary for test split
- **`analysis_final_train.json`** - Analysis summary for train split
- **`analysis_final_validation.json`** - Analysis summary for validation split
- **`run_metadata_*.json`** - Configuration and metadata for each split
- **`evaluations/`** - Directory containing evaluation results
  - `evaluation_results_all.json` - All evaluation results
  - `evaluation_results_all.xlsx` - Excel format of evaluation results

## 🔍 Analysis Directory

The `analysis/` directory contains aggregated results across all model runs:

- **`aggregated_evaluations.json`** - Combined evaluation results from all runs
- **`aggregated_evaluations.csv`** - CSV format of aggregated results
- **`aggregated_evaluations.xlsx`** - Excel format of aggregated results
- **`pivot_evaluations.json`** - Pivot table format for cross-run analysis
- **`pivot_evaluations.csv`** - CSV pivot table
- **`pivot_evaluations.xlsx`** - Excel pivot table
- **`analysis_notebook.ipynb`** - Jupyter notebook for detailed analysis
- **`create_pivot_table.py`** - Script to generate pivot tables

## 🎯 Fine-tuning Datasets

The `datasets_for_FT/` directory contains datasets created from the model run results for fine-tuning purposes:

- **`dataset_balanced.jsonl`** - Balanced dataset across difficulty levels
- **`dataset_easy_heavy.jsonl`** - Dataset with emphasis on easy problems
- **`dataset_medium_heavy.jsonl`** - Dataset with emphasis on medium problems
- **`dataset_hard_heavy.jsonl`** - Dataset with emphasis on hard problems
- **`datasets_summary.csv`** - Summary statistics of all datasets
- **`create_datasets.py`** - Script to create fine-tuning datasets
- **`create_jsonl_datasets.py`** - Script to convert datasets to JSONL format

## 📈 Pass@k Metrics

The `pass@k.csv` file contains Pass@k metrics (Pass@1, Pass@5, Pass@10) for each model run, providing a quick comparison of model performance across different runs.

## 🚀 Usage

### View Individual Run Results
```bash
# View results from run 01
ls ModelRuns/01/

# View evaluation results
ls ModelRuns/01/evaluations/
```

### Analyze Aggregated Results
```bash
# View aggregated analysis
ls ModelRuns/analysis/

# Open analysis notebook
jupyter notebook ModelRuns/analysis/analysis_notebook.ipynb
```

### Create Fine-tuning Datasets
```bash
# Navigate to datasets directory
cd ModelRuns/datasets_for_FT/

# Create datasets from results
python create_datasets.py

# Convert to JSONL format
python create_jsonl_datasets.py
```

## 📋 Run Configurations

Each run may have different configurations such as:
- Different temperature settings
- Different model parameters
- Different prompt templates
- Different evaluation criteria

Check the `run_metadata_*.json` files in each run directory for specific configuration details. 