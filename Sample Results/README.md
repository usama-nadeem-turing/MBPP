# Sample Results Directory

This directory contains sample outputs and example results from the MBPP inference and evaluation pipeline. These files serve as references and examples for understanding the expected output formats and data structures.

## 📁 Directory Structure

```
Sample Results/
├── Evaluation/                           # Sample evaluation results
│   └── evaluation_results_test_split.json
├── evaluation_results_all_20250626_142820.json
├── evaluation_results_all_20250626_142950.json
├── evaluation_results_all_20250626_142950.xlsx
├── mbpp_results_final_test.json
├── mbpp_results_final_train.json
└── mbpp_results_final_validation.json
```

## 🎯 Purpose

This directory provides:
- **Reference examples** - Sample outputs for understanding data formats
- **Testing data** - Files for testing scripts and tools
- **Documentation** - Examples of expected results
- **Validation** - Sample data for verifying pipeline functionality

## 📊 Files Description

### Model Results Files
- **`mbpp_results_final_test.json`** - Sample model responses for test split
- **`mbpp_results_final_train.json`** - Sample model responses for train split
- **`mbpp_results_final_validation.json`** - Sample model responses for validation split

### Evaluation Results Files
- **`evaluation_results_all_20250626_142820.json`** - Sample evaluation results (JSON format)
- **`evaluation_results_all_20250626_142950.json`** - Sample evaluation results (JSON format)
- **`evaluation_results_all_20250626_142950.xlsx`** - Sample evaluation results (Excel format)

### Evaluation Directory
- **`Evaluation/evaluation_results_test_split.json`** - Sample test split evaluation results

## 🔍 Data Structure Examples

### Model Results Format
```json
{
  "mbpp_id": "1",
  "prompt": "Write a function that...",
  "response": "def solution(...):\n    ...",
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "temperature": 0.2,
  "max_tokens": 512
}
```

### Evaluation Results Format
```json
{
  "mbpp_id": "1",
  "prompt": "Write a function that...",
  "response": "def solution(...):\n    ...",
  "passed": true,
  "error_message": null,
  "test_cases_passed": 3,
  "total_test_cases": 3
}
```

## 🚀 Usage

### Reference for Development
```bash
# View sample model results
head -n 50 "Sample Results/mbpp_results_final_test.json"

# View sample evaluation results
head -n 50 "Sample Results/evaluation_results_all_20250626_142950.json"
```

### Testing Scripts
```bash
# Test evaluation script with sample data
python evaluation_pass_xlsx.py --file-path "Sample Results/mbpp_results_final_test.json"

# Test conversion script
python eval_converter.py "Sample Results/Evaluation/evaluation_results_test_split.json"
```

### Understanding Data Formats
```bash
# Check JSON structure
python -m json.tool "Sample Results/mbpp_results_final_test.json" | head -n 100

# View Excel file structure
python -c "import pandas as pd; df = pd.read_excel('Sample Results/evaluation_results_all_20250626_142950.xlsx'); print(df.head())"
```

## 📋 File Sizes and Content

- **Model Results**: Large files containing full model responses
- **Evaluation Results**: Smaller files with pass/fail information
- **Excel Files**: Formatted results for easy analysis
- **JSON Files**: Raw data for programmatic processing

## 🔗 Related Directories

- **`ModelRuns/`** - Contains the actual model run results
- **`Scrap/`** - Contains experimental and discarded results
- **`FineTuning/`** - Uses results for creating training datasets

## ⚠️ Important Notes

- **Sample data only** - These are examples, not complete results
- **Reference purpose** - Use for understanding formats and structures
- **Not for analysis** - Use actual results from `ModelRuns/` for analysis
- **May be outdated** - Sample files may not reflect current data formats 