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
  --split TEXT        Dataset split to use (train, validation, test) [default: test]
  --max-problems INT  Maximum number of problems to process
  --model-url TEXT    URL of the model server [default: http://localhost:18000/v1/chat/completions]
  --model-name TEXT   Name of the model to use [default: Qwen/Qwen2.5-1.5B-Instruct]
  --temperature FLOAT Sampling temperature [default: 0.2]
  --max-tokens INT    Maximum tokens to generate [default: 512]
  --help              Show this message and exit
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

The script generates different output files based on the mode:

### Demo Mode
- `mbpp_demo_results.json`: Results for 4 demo problems

### Full Mode
- `mbpp_results_final.json`: Complete results for all processed problems
- `mbpp_results_intermediate_X.json`: Intermediate results every 10 problems

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