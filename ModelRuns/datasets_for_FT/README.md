# Fine-tuning Datasets Directory

This directory contains datasets created from model run results specifically designed for fine-tuning language models on Python programming problems.

## 📁 Directory Structure

```
datasets_for_FT/
├── create_datasets.py              # Main script to create datasets
├── create_jsonl_datasets.py        # Script to convert to JSONL format
├── datasets_summary.csv            # Summary statistics of all datasets
├── dataset_balanced.jsonl          # Balanced dataset across difficulty levels
├── dataset_balanced.csv            # CSV version of balanced dataset
├── dataset_easy_heavy.jsonl        # Dataset with emphasis on easy problems
├── dataset_easy_heavy.csv          # CSV version of easy-heavy dataset
├── dataset_medium_heavy.jsonl      # Dataset with emphasis on medium problems
├── dataset_medium_heavy.csv        # CSV version of medium-heavy dataset
├── dataset_hard_heavy.jsonl        # Dataset with emphasis on hard problems
└── dataset_hard_heavy.csv          # CSV version of hard-heavy dataset
```

## 🎯 Purpose

This directory serves as a data preparation pipeline for fine-tuning by:
- **Filtering successful examples** - Only including code that passes test cases
- **Creating balanced datasets** - Ensuring representation across difficulty levels
- **Formatting for training** - Converting to JSONL format for model training
- **Quality control** - Ensuring high-quality training examples

## 📊 Dataset Types

### Balanced Dataset (`dataset_balanced.jsonl`)
- **Purpose**: Equal representation across all difficulty levels
- **Use case**: General fine-tuning for overall improvement
- **Characteristics**: 
  - Equal number of easy, medium, and hard problems
  - Diverse problem types and solution approaches
  - Suitable for general-purpose fine-tuning

### Easy-Heavy Dataset (`dataset_easy_heavy.jsonl`)
- **Purpose**: Emphasis on easy problems for foundational learning
- **Use case**: Teaching basic Python concepts and syntax
- **Characteristics**:
  - Higher proportion of easy problems
  - Focus on fundamental programming concepts
  - Good for beginners or basic skill reinforcement

### Medium-Heavy Dataset (`dataset_medium_heavy.jsonl`)
- **Purpose**: Emphasis on medium-difficulty problems
- **Use case**: Improving intermediate programming skills
- **Characteristics**:
  - Higher proportion of medium problems
  - Balanced mix of easy and hard problems
  - Focus on practical programming scenarios

### Hard-Heavy Dataset (`dataset_hard_heavy.jsonl`)
- **Purpose**: Emphasis on challenging problems
- **Use case**: Advanced programming and problem-solving
- **Characteristics**:
  - Higher proportion of hard problems
  - Complex algorithms and data structures
  - Edge cases and optimization challenges

## 🔧 Creation Scripts

### `create_datasets.py`
Main script that:
- Loads evaluation results from all model runs
- Filters for successful code examples
- Creates balanced and weighted datasets
- Exports in both CSV and JSONL formats
- Generates summary statistics

### `create_jsonl_datasets.py`
Script that:
- Converts CSV datasets to JSONL format
- Ensures proper formatting for training
- Validates data integrity
- Creates training-ready files

## 📈 Dataset Statistics

The `datasets_summary.csv` file contains:
- **Total examples** per dataset
- **Difficulty distribution** (easy/medium/hard counts)
- **Average problem length** and complexity
- **Success rates** by difficulty level
- **Quality metrics** for each dataset

## 🚀 Usage

### Create All Datasets
```bash
# Generate all datasets from model results
python create_datasets.py
```

### Convert to JSONL Format
```bash
# Convert CSV datasets to JSONL for training
python create_jsonl_datasets.py
```

### View Dataset Statistics
```bash
# View summary of all datasets
python -c "import pandas as pd; df = pd.read_csv('datasets_summary.csv'); print(df)"
```

### Inspect Dataset Contents
```bash
# View first few examples from balanced dataset
python -c "import json; data = [json.loads(line) for line in open('dataset_balanced.jsonl')][:3]; print(json.dumps(data, indent=2))"
```

## 📋 Data Format

### JSONL Format (for training)
Each line contains a JSON object with:
```json
{
  "instruction": "Write a function that...",
  "input": "",
  "output": "def solution(...):\n    ...",
  "difficulty": "medium",
  "mbpp_id": "123"
}
```

### CSV Format (for analysis)
Columns include:
- `mbpp_id`: Original problem ID
- `prompt`: Problem statement
- `response`: Successful code solution
- `difficulty`: Problem difficulty level
- `passed`: Whether the solution passed tests
- `run_id`: Source model run

## 🎯 Fine-tuning Applications

### Model Improvement
- **Code generation accuracy** - Better syntax and logic
- **Problem understanding** - Improved comprehension of requirements
- **Solution quality** - More robust and efficient code
- **Error reduction** - Fewer syntax and runtime errors

### Specialized Training
- **Difficulty-specific models** - Models optimized for specific difficulty levels
- **Domain adaptation** - Models better suited for Python programming
- **Style consistency** - Consistent coding style and patterns

## 🔗 Related Directories

- **`../01/ through ../12/`** - Source model run results
- **`../analysis/`** - Aggregated evaluation results
- **`../../FineTuning/`** - Fine-tuning implementation scripts

## ⚠️ Important Notes

- **Quality filtering** - Only successful examples are included
- **Balanced sampling** - Ensures diverse representation
- **Format validation** - All examples are validated before inclusion
- **Size optimization** - Datasets are sized for efficient training 