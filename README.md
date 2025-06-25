# MBPP Dataset Inference

This project provides a comprehensive solution to load the MBPP (Mostly Basic Python Problems) dataset and get model responses for each problem using a local model server.

## Features

- Load MBPP dataset from Hugging Face
- Send problems to local model server (vLLM)
- Save results to JSON files with MBPP IDs
- Analyze response statistics
- Robust error handling and logging
- Configurable parameters (temperature, max_tokens, etc.)
- **Demo mode** for quick testing with 3-4 problems
- Command-line interface with multiple options

## Prerequisites

1. **Model Server**: Make sure your vLLM model server is running on `localhost:18000`
2. **Python Dependencies**: Install required packages

## Installation

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

## Usage

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
- Save results to `mbpp_demo_results.json`
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

### 3. Command Line Options

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

### 4. Custom Usage

You can also use the `MBPPInference` class programmatically:

```python
from inference import MBPPInference

# Initialize with custom settings
inference = MBPPInference(
    model_url="http://localhost:18000/v1/chat/completions",
    model_name="Qwen/Qwen2.5-1.5B-Instruct"
)

# Load dataset
problems = inference.load_mbpp_dataset(split="test")

# Process problems in demo mode
results = inference.process_problems(problems, demo_mode=True)

# Analyze results
analysis = inference.analyze_results(results)
print(analysis)
```

## Configuration

### Model Settings

You can modify the model settings via command line or in the `MBPPInference` class:

- `model_url`: URL of your model server
- `model_name`: Name of the model
- `temperature`: Sampling temperature (default: 0.2)
- `max_tokens`: Maximum tokens to generate (default: 512)

### Dataset Settings

- `split`: Dataset split to use ('train', 'validation', 'test')
- `max_problems`: Limit number of problems to process
- `demo_mode`: Process only 4 problems for quick testing

## Output Files

The script creates a timestamped results folder for each run: `results_YYYYMMDD_HHMMSS/`

### Demo Mode
- `results_20241225_143052/mbpp_demo_results_test.json`: Complete results for test split (updated incrementally)
- `results_20241225_143052/mbpp_demo_results_train.json`: Complete results for train split (updated incrementally)
- `results_20241225_143052/mbpp_demo_results_validation.json`: Complete results for validation split (updated incrementally)
- `results_20241225_143052/mbpp_demo_task_X_timestamp.json`: Individual task results (one file per task)
- `results_20241225_143052/analysis_demo_test.json`: Analysis statistics for test split
- `results_20241225_143052/analysis_demo_train.json`: Analysis statistics for train split
- `results_20241225_143052/analysis_demo_validation.json`: Analysis statistics for validation split
- `results_20241225_143052/run_metadata_test.json`: Run configuration and metadata for test split
- `results_20241225_143052/run_metadata_train.json`: Run configuration and metadata for train split
- `results_20241225_143052/run_metadata_validation.json`: Run configuration and metadata for validation split

### Full Mode
- `results_20241225_143052/mbpp_results_final_test.json`: Complete results for test split (updated incrementally)
- `results_20241225_143052/mbpp_results_final_train.json`: Complete results for train split (updated incrementally)
- `results_20241225_143052/mbpp_results_final_validation.json`: Complete results for validation split (updated incrementally)
- `results_20241225_143052/mbpp_task_X_timestamp.json`: Individual task results (one file per task)
- `results_20241225_143052/analysis_final_test.json`: Analysis statistics for test split
- `results_20241225_143052/analysis_final_train.json`: Analysis statistics for train split
- `results_20241225_143052/analysis_final_validation.json`: Analysis statistics for validation split
- `results_20241225_143052/run_metadata_test.json`: Run configuration and metadata for test split
- `results_20241225_143052/run_metadata_train.json`: Run configuration and metadata for train split
- `results_20241225_143052/run_metadata_validation.json`: Run configuration and metadata for validation split

### Incremental Saving
The script saves results **after each task** to prevent data loss:
- If the process is interrupted, you can resume from where you left off
- Each task result is saved individually for easy access
- The main results file is updated incrementally with all completed tasks
- All files are organized in timestamped folders to prevent overwriting

## Output Format

Each result contains:

```json
{
  "mbpp_id": "problem_id",
  "problem": {
    "text": "problem_description",
    "prompt": "function_signature",
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

## Analysis Results

The analysis includes:

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

## Error Handling

The script includes comprehensive error handling:

- Connection errors to model server
- Dataset loading errors
- API response errors
- File saving errors

All errors are logged with timestamps and appropriate error messages.

## Troubleshooting

1. **Connection Error**: Make sure your model server is running on the correct port
2. **Dataset Loading Error**: Check your internet connection and Hugging Face access
3. **Memory Issues**: Use `--demo` mode or reduce `--max-problems` parameter
4. **Slow Processing**: The script includes delays to avoid overwhelming the server
5. **Dataset Structure Issues**: If you encounter KeyError for missing fields, run with `--debug` to inspect the dataset structure:
   ```bash
   python inference.py --demo --debug
   ```
   This will show you the actual structure of the MBPP dataset and help identify the correct field names.

## Model Server Examples

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

## License

This project is open source and available under the MIT License.