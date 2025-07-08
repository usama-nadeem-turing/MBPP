# Model Run 01

This directory contains the complete results from model run 01, including model responses, evaluations, and analysis for all dataset splits.

## 📁 Directory Structure

```
01/
├── evaluations/                           # Evaluation results directory
│   ├── evaluation_results_all.json       # All evaluation results
│   └── evaluation_results_all.xlsx       # Excel format of evaluation results
├── mbpp_results_final_test.json          # Model responses for test split
├── mbpp_results_final_train.json         # Model responses for train split
├── mbpp_results_final_validation.json    # Model responses for validation split
├── analysis_final_test.json              # Analysis summary for test split
├── analysis_final_train.json             # Analysis summary for train split
├── analysis_final_validation.json        # Analysis summary for validation split
├── run_metadata_test.json                # Configuration for test split
├── run_metadata_train.json               # Configuration for train split
└── run_metadata_validation.json          # Configuration for validation split
```

## 📊 File Descriptions

### Model Results Files
- **`mbpp_results_final_*.json`** - Raw model responses for each dataset split
  - Contains problem statements, model responses, and metadata
  - Large files with complete inference results
  - Used as input for evaluation scripts

### Analysis Files
- **`analysis_final_*.json`** - Summary analysis for each split
  - Performance metrics and statistics
  - Difficulty level breakdowns
  - Error analysis and patterns
  - Compact summary of results

### Evaluation Files
- **`evaluations/evaluation_results_all.json`** - Complete evaluation results
  - Pass/fail status for each problem
  - Error messages and test case results
  - Performance metrics and statistics
- **`evaluations/evaluation_results_all.xlsx`** - Excel format for easy analysis

### Metadata Files
- **`run_metadata_*.json`** - Configuration and metadata for each split
  - Model parameters and settings
  - Dataset information
  - Run timestamps and environment details

## 🔍 Data Structure

### Model Results Format
```json
{
  "mbpp_id": "1",
  "prompt": "Write a function that...",
  "response": "def solution(...):\n    ...",
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "temperature": 0.2,
  "max_tokens": 512,
  "split": "test"
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
  "total_test_cases": 3,
  "difficulty": "medium"
}
```

### Analysis Summary Format
```json
{
  "total_problems": 500,
  "passed_problems": 350,
  "pass_rate": 0.70,
  "difficulty_breakdown": {
    "easy": {"total": 167, "passed": 140, "pass_rate": 0.84},
    "medium": {"total": 167, "passed": 120, "pass_rate": 0.72},
    "hard": {"total": 166, "passed": 90, "pass_rate": 0.54}
  }
}
```

## 🚀 Usage

### View Results
```bash
# View model responses
python -c "import json; data = json.load(open('mbpp_results_final_test.json')); print(f'Total problems: {len(data)}')"

# View evaluation results
python -c "import json; data = json.load(open('evaluations/evaluation_results_all.json')); print(f'Pass rate: {sum(1 for x in data if x[\"passed\"])/len(data):.2%}')"

# View analysis summary
python -c "import json; data = json.load(open('analysis_final_test.json')); print(json.dumps(data, indent=2))"
```

### Compare with Other Runs
```bash
# Compare with run 02
python -c "import json; r1 = json.load(open('analysis_final_test.json')); r2 = json.load(open('../02/analysis_final_test.json')); print(f'Run 01: {r1[\"pass_rate\"]:.2%}, Run 02: {r2[\"pass_rate\"]:.2%}')"
```

### Export to Excel
```bash
# Convert evaluation results to Excel
python ../../eval_converter.py "evaluations/evaluation_results_all.json"
```

## 📈 Performance Metrics

### Overall Performance
- **Total Problems**: Varies by split (test: ~500, train: ~1000, validation: ~250)
- **Pass Rate**: Percentage of problems that pass all test cases
- **Average Response Time**: Time taken for model inference

### Difficulty Breakdown
- **Easy Problems**: Basic Python concepts and syntax
- **Medium Problems**: Intermediate algorithms and data structures
- **Hard Problems**: Complex algorithms and optimization challenges

### Error Analysis
- **Syntax Errors**: Python syntax and indentation issues
- **Logic Errors**: Incorrect algorithm implementation
- **Edge Case Failures**: Problems with boundary conditions
- **Timeout Errors**: Solutions that exceed execution limits

## 🔗 Related Files

- **`../analysis/`** - Aggregated analysis across all runs
- **`../datasets_for_FT/`** - Fine-tuning datasets created from results
- **`../../evaluation_pass_xlsx.py`** - Script to evaluate model responses
- **`../../aggregate_evaluations.py`** - Script to aggregate results across runs

## 📋 Run Configuration

Check the `run_metadata_*.json` files for specific configuration details including:
- Model name and version
- Temperature and sampling parameters
- Maximum token limits
- Dataset split information
- Run timestamps and environment details

## ⚠️ Notes

- **File sizes**: Model result files are large (1-2MB) due to complete responses
- **Evaluation dependency**: Evaluation results depend on model results
- **Cross-run comparison**: Use analysis files for comparing performance across runs
- **Data integrity**: All files are validated and consistent within the run 