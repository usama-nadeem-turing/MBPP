# Analysis Directory

This directory contains aggregated analysis results and tools for analyzing performance across all model runs in the MBPP evaluation pipeline.

## 📁 Directory Structure

```
analysis/
├── analysis_notebook.ipynb        # Jupyter notebook for detailed analysis
├── create_pivot_table.py          # Script to generate pivot tables
├── pivot_evaluations.xlsx         # Pivot table in Excel format
├── pivot_evaluations.csv          # Pivot table in CSV format
├── pivot_evaluations.json         # Pivot table in JSON format
├── aggregated_evaluations.xlsx    # Aggregated results in Excel format
├── aggregated_evaluations.csv     # Aggregated results in CSV format
└── aggregated_evaluations.json    # Aggregated results in JSON format
```

## 🎯 Purpose

This directory provides comprehensive analysis tools and results that:
- **Aggregate data** from all 12 model runs
- **Compare performance** across different configurations
- **Generate insights** about model behavior
- **Create visualizations** for presentation and reporting
- **Enable statistical analysis** of results

## 📊 Files Description

### Analysis Scripts
- **`analysis_notebook.ipynb`** - Interactive Jupyter notebook with detailed analysis, visualizations, and statistical comparisons
- **`create_pivot_table.py`** - Python script to generate pivot tables for cross-run analysis

### Aggregated Results
- **`aggregated_evaluations.json`** - Complete aggregated results from all model runs (875KB)
- **`aggregated_evaluations.csv`** - CSV format for easy spreadsheet analysis (152KB)
- **`aggregated_evaluations.xlsx`** - Excel format with formatting and charts (168KB)

### Pivot Tables
- **`pivot_evaluations.json`** - Pivot table data in JSON format (329KB)
- **`pivot_evaluations.csv`** - Pivot table in CSV format (68KB)
- **`pivot_evaluations.xlsx`** - Pivot table in Excel format with formatting (83KB)

## 🔍 Analysis Capabilities

### Cross-Run Comparison
- Performance comparison across all 12 model runs
- Statistical significance testing
- Trend analysis over different configurations
- Outlier detection and analysis

### Difficulty Level Analysis
- Performance breakdown by problem difficulty (easy, medium, hard)
- Difficulty-specific insights and recommendations
- Comparative analysis across difficulty levels

### Error Analysis
- Common error patterns across runs
- Failure mode analysis
- Error correlation with problem characteristics

### Statistical Insights
- Confidence intervals for performance metrics
- Correlation analysis between variables
- Regression analysis for performance prediction

## 🚀 Usage

### Interactive Analysis
```bash
# Start Jupyter notebook for detailed analysis
jupyter notebook analysis_notebook.ipynb
```

### Generate Pivot Tables
```bash
# Create new pivot tables from aggregated data
python create_pivot_table.py
```

### View Aggregated Results
```bash
# View JSON results
python -c "import json; data = json.load(open('aggregated_evaluations.json')); print(len(data))"

# View CSV results
python -c "import pandas as pd; df = pd.read_csv('aggregated_evaluations.csv'); print(df.head())"
```

### Excel Analysis
```bash
# Open Excel files for manual analysis
# aggregated_evaluations.xlsx - Complete results
# pivot_evaluations.xlsx - Cross-run comparison
```

## 📈 Key Metrics Available

### Performance Metrics
- **Pass@1, Pass@5, Pass@10** - Standard code generation metrics
- **Accuracy by difficulty** - Performance breakdown by problem difficulty
- **Error rates** - Failure analysis and patterns
- **Response quality** - Code quality assessment

### Comparative Metrics
- **Run-to-run comparison** - Performance variation across runs
- **Configuration impact** - Effect of different parameters
- **Temporal trends** - Performance evolution over time
- **Statistical significance** - Confidence in performance differences

## 🔗 Data Sources

The analysis uses data from:
- **ModelRuns/01/ through ModelRuns/12/** - Individual run results
- **evaluations/evaluation_results_all.json** - Evaluation results from each run
- **run_metadata_*.json** - Configuration and metadata from each run

## 📋 Analysis Workflow

1. **Data Collection** - Gather results from all model runs
2. **Data Cleaning** - Remove outliers and invalid entries
3. **Aggregation** - Combine results across runs
4. **Statistical Analysis** - Perform statistical tests and comparisons
5. **Visualization** - Create charts and graphs
6. **Insight Generation** - Extract actionable insights
7. **Report Generation** - Create formatted reports and presentations

## 🎯 Use Cases

- **Research Analysis** - Academic research and paper writing
- **Model Comparison** - Comparing different model configurations
- **Performance Optimization** - Identifying best practices and parameters
- **Error Analysis** - Understanding failure modes and patterns
- **Reporting** - Creating presentations and reports for stakeholders 