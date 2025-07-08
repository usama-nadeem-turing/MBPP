# MBPP Dataset Inference and Evaluation Pipeline

This project provides a comprehensive solution to load the MBPP (Mostly Basic Python Problems) dataset, get model responses for each problem using a local model server, and evaluate the generated code against test cases.

## 🚀 Features

- **Dataset Loading**: Load MBPP dataset from Hugging Face with flexible split handling
- **Model Inference**: Send problems to local model server (vLLM) with configurable parameters
- **Code Evaluation**: Test generated code against original MBPP test cases
- **Excel Export**: Convert evaluation results to formatted Excel files for analysis
- **Results Aggregation**: Aggregate evaluation results across multiple model runs
- **Pivot Table Generation**: Create pivot tables for cross-run analysis
- **Incremental Saving**: Save results after each task to prevent data loss
- **Multi-split Processing**: Process train, validation, and test splits individually or all at once
- **Demo Mode**: Quick testing with 3-4 problems
- **Clean Output Structure**: Organized results with split-specific files and no individual task clutter
- **Comprehensive Logging**: Detailed logging with debug mode support
- **Robust Error Handling**: Handle connection errors, API failures, and dataset issues

## 📁 Project Structure

```
MBPP/
├── inference.py                 # Main inference script
├── evaluation_pass_xlsx.py      # Evaluation script with Excel export
├── aggregate_evaluations.py     # Aggregate results across model runs
├── create_pivot_table.py        # Create pivot tables for analysis
├── eval_converter.py           # JSON to Excel converter
├── run_multiple_inference.py   # Run inference 9 times with different configs
├── test_inference.py           # Model server connection test
├── host model.py               # Model hosting utilities
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── ModelRuns/                  # Directory containing multiple model run results
│   ├── README.md               # Documentation for model runs directory
│   ├── 01/                     # Model run 1
│   │   ├── README.md           # Documentation for run 01
│   │   ├── evaluations/
│   │   │   ├── evaluation_results_all.json
│   │   │   └── evaluation_results_all.xlsx
│   │   ├── mbpp_results_final_test.json
│   │   ├── mbpp_results_final_train.json
│   │   └── mbpp_results_final_validation.json
│   ├── 02/ through 12/         # Additional model runs (02-12)
│   ├── analysis/               # Aggregated analysis across all runs
│   │   ├── README.md           # Documentation for analysis directory
│   │   ├── aggregated_evaluations.json
│   │   ├── aggregated_evaluations.csv
│   │   ├── aggregated_evaluations.xlsx
│   │   ├── pivot_evaluations.json
│   │   ├── pivot_evaluations.csv
│   │   └── pivot_evaluations.xlsx
│   ├── datasets_for_FT/        # Fine-tuning datasets
│   │   ├── README.md           # Documentation for fine-tuning datasets
│   │   ├── dataset_balanced.jsonl
│   │   ├── dataset_easy_heavy.jsonl
│   │   ├── dataset_medium_heavy.jsonl
│   │   └── dataset_hard_heavy.jsonl
│   └── pass@k.csv             # Pass@k metrics summary
├── FineTuning/                 # Fine-tuning scripts and data
│   ├── README.md               # Documentation for fine-tuning directory
│   ├── code.py                 # Fine-tuning implementation
│   ├── create jsonl.py         # JSONL dataset creation
│   └── new_train.jsonl         # Training dataset
├── Sample Results/             # Sample outputs and reference data
│   ├── README.md               # Documentation for sample results
│   ├── Evaluation/
│   │   └── evaluation_results_test_split.json
│   ├── mbpp_results_final_test.json
│   ├── mbpp_results_final_train.json
│   └── mbpp_results_final_validation.json
└── Scrap/                      # Experimental and discarded results
    ├── Faulty Run/             # Results from runs with errors
    ├── results_YYYYMMDD_HHMMSS/ # Experimental runs
    └── Results_no_1_0/         # Early version results
```

## 📚 Documentation Structure

This project includes comprehensive documentation for each directory:

- **`README.md`** (this file) - Main project overview and usage guide
- **`ModelRuns/README.md`** - Documentation for model run results and structure
- **`ModelRuns/analysis/README.md`** - Guide to aggregated analysis tools and results
- **`ModelRuns/datasets_for_FT/README.md`** - Fine-tuning dataset creation and usage
- **`ModelRuns/01/README.md`** - Template for individual model run documentation
- **`FineTuning/README.md`** - Fine-tuning process and implementation guide
- **`Sample Results/README.md`** - Sample outputs and reference data documentation

Each README provides detailed information about file structures, usage examples, and specific workflows for that directory.

## 📋 Prerequisites

1. **Model Server**: Make sure your vLLM model server is running on `localhost:18000`
2. **Python Dependencies**: Install required packages
3. **Dataset Access**: Access to MBPP dataset via Hugging Face

## 🛠️ Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start your model server (example for Qwen2.5-1.5B-Instruct):
```bash
vllm serve "Qwen/Qwen2.5-1.5B-Instruct" \
    --dtype float16 \
    --port 18000 \
    --host 0.0.0.0 \
    --tokenizer-pool-type none \
    --swap-space 0
```

## 🚀 Usage

### 1. Test Model Connection

First, test if your model server is running:

```bash
python test_inference.py
```

### 2. Run MBPP Inference

#### Quick Demo (Recommended for first run)
Run with demo mode to test with just 4 problems:

```bash
python inference.py --demo
```

This will:
- Process only 4 problems
- Save results to timestamped folders
- Show quick analysis

#### Full Run
Run the complete inference:

```bash
python inference.py
```

#### Custom Configuration
```bash
python inference.py --split test --max-problems 20 --temperature 0.1 --max-tokens 256
```

### 3. Command Line Options for Inference

```bash
python inference.py [OPTIONS]

Options:
  --demo              Run in demo mode with only 3-4 problems
  --split TEXT        Dataset split to use (train, validation, test, or "all" for all splits) [default: test]
  --max-problems INT  Maximum number of problems to process
  --model-url TEXT    URL of the model server [default: http://localhost:18000/v1/chat/completions]
  --model-name TEXT   Name of the model to use [default: Qwen/Qwen2.5-1.5B-Instruct]
  --temperature FLOAT Sampling temperature [default: 0.2]
  --max-tokens INT    Maximum tokens to generate [default: 512]
  --debug             Enable debug logging to see dataset structure
  --help              Show this message and exit
```

### 4. Split Processing Examples

#### Process specific split:
```bash
python inference.py --split test --demo
python inference.py --split train --max-problems 50
python inference.py --split validation
```

#### Process all splits:
```bash
python inference.py --split all --demo
python inference.py --split all --max-problems 100
```

**Note:** When using `--split all`, the "prompt" split is automatically excluded as it's not typically used for evaluation.

### 5. Evaluate Generated Code

After running inference, evaluate if the generated code passes the test cases:

#### Basic Evaluation
```bash
# Evaluate most recent results (all splits)
python evaluation_pass_xlsx.py

# Evaluate specific file
python evaluation_pass_xlsx.py --file-path "Results_no_1/mbpp_results_final_test.json"

# Evaluate specific split
python evaluation_pass_xlsx.py --split test
python evaluation_pass_xlsx.py --split train
python evaluation_pass_xlsx.py --split validation

# Evaluate specific results directory
python evaluation_pass_xlsx.py --results-dir results_20241225_143052

# Evaluate specific MBPP ID
python evaluation_pass_xlsx.py --mbpp-id 11

# Enable debug logging
python evaluation_pass_xlsx.py --debug
```

#### New Features in Evaluation
- **Problem Statement Inclusion**: Evaluation results now include the problem statement for better context
- **Improved Error Handling**: Better handling of assertion errors and silent test failures
- **Enhanced Debug Logging**: More detailed logging for troubleshooting test execution
- **Organized Output**: Results are saved in `evaluations/` subdirectory within the results directory

#### Command Line Options for Evaluation
```bash
python evaluation_pass_xlsx.py [OPTIONS]

Options:
  --results-dir TEXT  Results directory to evaluate (default: most recent)
  --file-path TEXT    Specific file path to evaluate (overrides results-dir)
  --split TEXT        Specific split to evaluate (test, train, validation)
  --mbpp-id TEXT      Evaluate specific MBPP ID only
  --debug             Enable debug logging
  --help              Show this message and exit
```

### 6. Convert Results to Excel

Convert evaluation results to Excel format for easy analysis:

```bash
# Convert evaluation results to Excel
python eval_converter.py "Results_no_1/Evaluation/evaluation_results_test_split.json"

# Specify output file
python eval_converter.py "Results_no_1/Evaluation/evaluation_results_test_split.json" --output "my_results.xlsx"

# Enable debug logging
python eval_converter.py "Results_no_1/Evaluation/evaluation_results_test_split.json" --debug
```

#### Command Line Options for Excel Converter
```bash
python eval_converter.py [OPTIONS] FILE_PATH

Arguments:
  FILE_PATH            Path to the evaluation results JSON file

Options:
  --output TEXT        Output Excel file path (optional)
  --debug              Enable debug logging
  --help               Show this message and exit
```

### 7. Aggregate Results Across Model Runs

After running multiple model runs and evaluations, aggregate the results for cross-run analysis:

#### Basic Aggregation
```bash
# Aggregate all evaluation results from Model Runs directory
python aggregate_evaluations.py
```

This creates:
- `aggregated_evaluations.json`: All results in JSON format
- `aggregated_evaluations.csv`: CSV format for analysis
- `aggregated_evaluations.xlsx`: Excel format with formatting

#### Command Line Options for Aggregation
```bash
python aggregate_evaluations.py [OPTIONS]

Options:
  --model-runs-dir TEXT  Path to Model Runs directory (default: "Model Runs")
  --output-dir TEXT      Output directory for results (default: current directory)
  --help                 Show this message and exit
```

### 8. Create Pivot Tables for Analysis

Create pivot tables for easier cross-run analysis:

#### Basic Pivot Table Creation
```bash
# Create pivot table from all model runs
python create_pivot_table.py
```

This creates:
- `pivot_evaluations.json`: Pivot table in JSON format
- `pivot_evaluations.csv`: CSV pivot table
- `pivot_evaluations.xlsx`: Excel pivot table with color coding

#### Pivot Table Format
The pivot table has the following structure:
```csv
mbpp_id,run_01_pass,run_02_pass,run_03_pass,...,run_12_pass
11,False,False,True,False,False,False,False,False,False,False,False,True
12,False,False,True,True,True,True,False,False,True,True,True,False
13,False,False,False,False,False,False,False,False,False,False,False,False
```

#### Command Line Options for Pivot Table
```bash
python create_pivot_table.py [OPTIONS]

Options:
  --model-runs-dir TEXT  Path to Model Runs directory (default: "Model Runs")
  --output-dir TEXT      Output directory for results (default: current directory)
  --help                 Show this message and exit
```

### 9. Run Multiple Experiments (Optional)

Run the same inference configuration 9 times for consistency testing:

```bash
# Run the same configuration 9 times
python run_multiple_inference.py
```

#### What This Does

The script runs `inference.py` 9 times with the **same configuration**:

- **Split**: All splits (train, validation, test - excluding "prompt")
- **Full Mode**: Process all problems in each split (no demo mode)
- **Default Parameters**: Uses default temperature and max_tokens from inference.py

#### Features

- ✅ **Comprehensive Testing**: Runs the same configuration 9 times to test reproducibility across all dataset splits
- ✅ **Progress Tracking**: Shows which run is currently executing (1/9, 2/9, etc.)
- ✅ **Error Handling**: Handles timeouts, crashes, and failures gracefully
- ✅ **Comprehensive Logging**: Shows progress, success/failure, and timing for each run
- ✅ **Safe Execution**: 1-hour timeout per run, 5-second delays between runs
- ✅ **Detailed Results**: Saves all outputs, errors, and timing data

#### Output

Creates a summary directory: `multiple_inference_run_YYYYMMDD_HHMMSS/`

```
multiple_inference_run_20241225_143052/
└── run_summary.json          # Summary of all 9 runs
```

Each individual run creates its own timestamped folder: `results_YYYYMMDD_HHMMSS/`

#### Sample Output

```
============================================================
🔄 Running inference 1/9
============================================================
Running command: python inference.py --split all
✅ Run 1/9 completed successfully
✅ Run 1 completed in 245.23 seconds
⏳ Waiting 5 seconds before next run...

============================================================
🔄 Running inference 2/9
============================================================
Running command: python inference.py --split all
✅ Run 2/9 completed successfully
✅ Run 2 completed in 242.18 seconds
```

#### Use Cases

- **Reproducibility Testing**: Verify that the same configuration produces consistent results across all splits
- **Performance Analysis**: Compare execution times across multiple comprehensive runs
- **Stability Testing**: Check if the model server and inference pipeline are stable with full dataset processing
- **Data Collection**: Gather multiple samples for statistical analysis across all dataset splits

## 📊 Output Files

The scripts create timestamped results folders for each run:

### Inference Script (`inference.py`)
- **Folder**: `results_YYYYMMDD_HHMMSS/`
- **Contents**: All inference results and analysis files

### Evaluation Script (`evaluation_pass_xlsx.py`)
- **Folder**: `evaluation_results_YYYYMMDD_HHMMSS/`
- **Contents**: Evaluation results in JSON and Excel formats

### Excel Converter (`eval_converter.py`)
- **Folder**: `conversion_results_YYYYMMDD_HHMMSS/`
- **Contents**: Converted Excel files

### Multiple Inference Runner (`run_multiple_inference.py`)
- **Folder**: `multiple_inference_run_YYYYMMDD_HHMMSS/`
- **Contents**: Summary of all 9 runs

### Aggregated Results (`aggregate_evaluations.py`)
- **Location**: `ModelRuns/analysis/`
- **Files**: `aggregated_evaluations.json`, `aggregated_evaluations.csv`, `aggregated_evaluations.xlsx`

### Pivot Tables (`create_pivot_table.py`)
- **Location**: `ModelRuns/analysis/`
- **Files**: `pivot_evaluations.json`, `pivot_evaluations.csv`, `pivot_evaluations.xlsx`

### Fine-tuning Datasets (`create_datasets.py`)
- **Location**: `ModelRuns/datasets_for_FT/`
- **Files**: Various JSONL and CSV datasets for fine-tuning

### Inference Output Files

#### Demo Mode
- `mbpp_demo_results_{split}.json`: Complete results for each split
- `analysis_demo_{split}.json`: Analysis statistics
- `run_metadata_{split}.json`: Run configuration and metadata

#### Full Mode
- `mbpp_results_final_{split}.json`: Complete results for each split
- `analysis_final_{split}.json`: Analysis statistics
- `run_metadata_{split}.json`: Run configuration and metadata

### Evaluation Output Files

#### JSON Results
- `evaluation_results_{split}.json`: Evaluation results in JSON format
- `evaluation_results_all.json`: All splits evaluation

#### Excel Results
- `evaluation_results_{split}.xlsx`: Evaluation results in Excel format
- `evaluation_results_all.xlsx`: All splits evaluation in Excel

### Excel Format

The Excel files contain the following columns:
- **mbpp_id**: The MBPP problem ID
- **test_case_statement**: The assert statement from the test case
- **test_output**: PASS or FAIL status
- **reason**: The reason (e.g., "All Clear", "Result does not match expected output")

## 📈 Data Formats

### Inference Result Format
```json
{
  "mbpp_id": "problem_id",
  "problem": {
    "text": "problem_description",
    "test_list": ["assert function(1) == 2", ...],
    "task_id": "task_id"
  },
  "prompt": "formatted_prompt_sent_to_model",
  "model_response": {
    "choices": [...],
    "usage": {...}
  },
  "timestamp": 1234567890.123
}
```

### Evaluation Result Format
```json
{
  "mbpp_id": 11,
  "problem_statement": "Write a python function to remove first and last occurrence of a given character from the string.",
  "passed": true,
  "error": null,
  "test_outputs": [
    "Test 1: PASS - All Clear",
    "Test 2: PASS - All Clear",
    "Test 3: PASS - All Clear"
  ],
  "generated_code": "def remove_Occ(s, ch):\n    # ...",
  "test_cases": [
    "assert remove_Occ(\"hello\",\"l\") == \"heo\"",
    "assert remove_Occ(\"abcda\",\"a\") == \"bcd\"",
    "assert remove_Occ(\"PHP\",\"P\") == \"H\""
  ]
}
```

### Analysis Results
```json
{
  "total_problems": 4,
  "successful_responses": 4,
  "failed_responses": 0,
  "success_rate": 1.0,
  "average_tokens": 150.5,
  "total_tokens": 602,
  "mbpp_ids": ["1", "2", "3", "4"]
}
```

### Aggregated Results Format
```json
{
  "model_run_id": "01",
  "mbpp_id": 11,
  "pass": false
}
```

### Pivot Table Format
```json
{
  "mbpp_id": 11,
  "run_01_pass": false,
  "run_02_pass": false,
  "run_03_pass": true,
  "run_04_pass": false,
  "run_05_pass": false,
  "run_06_pass": false,
  "run_07_pass": false,
  "run_08_pass": false,
  "run_09_pass": false,
  "run_10_pass": false,
  "run_11_pass": false,
  "run_12_pass": true
}
```

## 🔧 Configuration

### Model Settings
- `model_url`: URL of your model server
- `model_name`: Name of the model
- `temperature`: Sampling temperature (default: 0.2)
- `max_tokens`: Maximum tokens to generate (default: 512)

### Dataset Settings
- `split`: Dataset split to use ('train', 'validation', 'test')
- `max_problems`: Limit number of problems to process
- `demo_mode`: Process only 4 problems for quick testing

## 🐛 Troubleshooting

### Common Issues

1. **Connection Error**: Make sure your model server is running on the correct port
2. **Dataset Loading Error**: Check your internet connection and Hugging Face access
3. **Memory Issues**: Use `--demo` mode or reduce `--max-problems` parameter
4. **Slow Processing**: The script includes delays to avoid overwhelming the server
5. **ModuleNotFoundError**: Install missing dependencies with `pip install -r requirements.txt`

### Debug Mode
Run with `--debug` to inspect dataset structure and troubleshoot issues:
```bash
python inference.py --demo --debug
python evaluation_pass_xlsx.py --debug
```

## 🖥️ Model Server Examples

### Qwen2.5-1.5B-Instruct
```bash
vllm serve "Qwen/Qwen2.5-1.5B-Instruct" \
    --dtype float16 \
    --port 18000 \
    --host 0.0.0.0 \
    --tokenizer-pool-type none \
    --swap-space 0
```

### Qwen2.5-0.5B
```bash
vllm serve "Qwen/Qwen2.5-0.5B" \
    --dtype float16 \
    --port 18000 \
    --host 0.0.0.0 \
    --tokenizer-pool-type none \
    --swap-space 0
```

### CodeLlama-7b-Python
```bash
vllm serve "meta-llama/CodeLlama-7b-Python-hf" \
    --dtype float16 \
    --port 18000 \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --tokenizer-pool-type none \
    --swap-space 0
```

## 📋 Dependencies

```
requests>=2.25.1
datasets>=2.0.0
huggingface-hub>=0.16.0
numpy>=1.21.0
tqdm>=4.62.0
pandas>=1.3.0
openpyxl>=3.0.0
```

## 🔄 Complete Workflow Example

1. **Start Model Server**:
```bash
vllm serve "Qwen/Qwen2.5-1.5B-Instruct" --port 18000
```

2. **Test Connection**:
```bash
python test_inference.py
```

3. **Run Inference**:
```bash
python inference.py --split all
```

4. **Evaluate Results**:
```bash
python evaluation_pass_xlsx.py --results-dir "results_YYYYMMDD_HHMMSS"
```

5. **Aggregate Multiple Runs** (after running multiple experiments):
```bash
python aggregate_evaluations.py
```

6. **Create Pivot Table**:
```bash
python create_pivot_table.py
```

7. **Analyze Results**:
- Open `ModelRuns/analysis/pivot_evaluations.xlsx` for cross-run analysis
- Use `ModelRuns/analysis/aggregated_evaluations.csv` for statistical analysis
- Check individual run results in `ModelRuns/XX/evaluations/`
- Review detailed analysis in `ModelRuns/analysis/analysis_notebook.ipynb`

8. **Create Fine-tuning Datasets** (optional):
```bash
cd ModelRuns/datasets_for_FT/
python create_datasets.py
python create_jsonl_datasets.py
```
python test_inference.py
```

3. **Run Demo Inference**:
```bash
python inference.py --demo --split test
```
*Creates: `results_20241225_143052/`*

4. **Evaluate Results**:
```bash
python evaluation_pass_xlsx.py --split test
```
*Creates: `evaluation_results_20241225_144530/`*

5. **Convert to Excel**:
```bash
python eval_converter.py "evaluation_results_20241225_144530/evaluation_results_test.json"
```
*Creates: `conversion_results_20241225_145000/`*

6. **Run Multiple Experiments** (Optional):
```bash
python run_multiple_inference.py
```
*Creates: `multiple_inference_run_20241225_150000/` with 9 runs processing all splits*

## 📝 License

This project is open source and available under the MIT License.

## 📖 Additional Documentation

For detailed information about specific components of this project, refer to the README files in each directory:

- **Model Runs**: See `ModelRuns/README.md` for complete documentation of model run results, analysis tools, and fine-tuning datasets
- **Individual Runs**: Each run directory (01-12) contains detailed results and can be documented using `ModelRuns/01/README.md` as a template
- **Analysis Tools**: See `ModelRuns/analysis/README.md` for comprehensive analysis capabilities and statistical tools
- **Fine-tuning**: See `FineTuning/README.md` and `ModelRuns/datasets_for_FT/README.md` for fine-tuning workflows
- **Sample Data**: See `Sample Results/README.md` for reference examples and data formats

## 🤝 Contributing

Feel free to submit issues and enhancement requests!