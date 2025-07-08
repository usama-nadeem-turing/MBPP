# Fine-tuning Directory

This directory contains scripts and data for fine-tuning models on the MBPP dataset using results from model inference runs.

## 📁 Directory Structure

```
FineTuning/
├── code.py              # Fine-tuning code implementation
├── create jsonl.py      # Script to create JSONL format datasets
└── new_train.jsonl     # Training dataset in JSONL format
```

## 🎯 Purpose

The fine-tuning process aims to improve model performance on Python programming problems by training on:
- Successful code examples from model runs
- Problem-solution pairs from the MBPP dataset
- Curated datasets based on difficulty levels and performance

## 📊 Files Description

### `code.py`
Main fine-tuning implementation script that:
- Loads training data from JSONL files
- Configures model training parameters
- Implements the fine-tuning loop
- Saves the fine-tuned model

### `create jsonl.py`
Script to create JSONL format datasets for fine-tuning:
- Converts evaluation results to training format
- Filters successful code examples
- Creates problem-solution pairs
- Exports in JSONL format for training

### `new_train.jsonl`
Training dataset in JSONL format containing:
- Problem statements from MBPP
- Successful code solutions
- Metadata for each training example

## 🚀 Usage

### Create Training Dataset
```bash
# Create JSONL dataset from model results
python "create jsonl.py"
```

### Run Fine-tuning
```bash
# Execute fine-tuning process
python code.py
```

## 🔗 Related Directories

- **`ModelRuns/datasets_for_FT/`** - Contains pre-processed datasets for fine-tuning
- **`ModelRuns/analysis/`** - Contains evaluation results used to create training data

## 📋 Training Data Sources

The fine-tuning datasets are created from:
1. **Successful model responses** - Code that passes test cases
2. **Original MBPP problems** - High-quality problem statements
3. **Difficulty-balanced samples** - Equal representation across difficulty levels
4. **Performance-based filtering** - Focus on examples that improve model performance

## 🎯 Fine-tuning Objectives

- Improve code generation accuracy
- Enhance problem-solving capabilities
- Reduce syntax and logical errors
- Better understanding of Python programming concepts
- Improved handling of edge cases and complex problems 