import pandas as pd
import json
import os
from collections import defaultdict

def load_evaluation_results(run_dir):
    """Load evaluation results from a run directory"""
    eval_file = os.path.join(run_dir, 'evaluations', 'evaluation_results_all.json')
    if os.path.exists(eval_file):
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except UnicodeDecodeError:
            try:
                with open(eval_file, 'r', encoding='latin-1') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {eval_file}: {e}")
                return []
    return []

def count_passed_tests(test_outputs):
    """Count how many tests passed from test_outputs"""
    if not test_outputs:
        return 0
    return sum(1 for output in test_outputs if "PASS" in output)

def find_best_run_for_mbpp_id(mbpp_id, all_runs_data):
    """Find the best performing run for a given mbpp_id"""
    best_run = None
    best_score = -1
    best_run_id = None
    
    for run_id, run_data in all_runs_data.items():
        for result in run_data:
            if result['mbpp_id'] == mbpp_id:
                # Calculate score: passed = 100, otherwise count passed tests
                if result['passed']:
                    score = 100
                else:
                    score = count_passed_tests(result['test_outputs'])
                
                if score > best_score:
                    best_score = score
                    best_run = result
                    best_run_id = run_id
    
    return best_run, best_run_id, best_score

def create_jsonl_for_dataset(dataset_csv_path, all_runs_data, output_jsonl_path):
    """Create JSONL file for a specific dataset"""
    # Read the dataset CSV
    df = pd.read_csv(dataset_csv_path)
    
    jsonl_data = []
    
    for _, row in df.iterrows():
        mbpp_id = row['mbpp_id']
        
        # Find the best run for this mbpp_id
        best_result, run_id, score = find_best_run_for_mbpp_id(mbpp_id, all_runs_data)
        
        if best_result:
            # Create JSONL entry
            entry = {
                "instruction": best_result['problem_statement'],
                "output": best_result['generated_code']
            }
            jsonl_data.append(entry)
            print(f"MBPP {mbpp_id}: Run {run_id}, Score: {score}")
        else:
            print(f"Warning: No results found for MBPP {mbpp_id}")
    
    # Write JSONL file
    with open(output_jsonl_path, 'w') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created {output_jsonl_path} with {len(jsonl_data)} entries")
    return len(jsonl_data)

def main():
    # Define run directories
    base_dir = ".."
    run_dirs = [f"{base_dir}/{i:02d}" for i in range(1, 13)]
    
    # Load all evaluation results
    print("Loading evaluation results from all runs...")
    all_runs_data = {}
    
    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir)
        print(f"Loading {run_id}...")
        results = load_evaluation_results(run_dir)
        if results:
            all_runs_data[run_id] = results
            print(f"  Loaded {len(results)} results")
        else:
            print(f"  No results found")
    
    # Dataset files to process
    datasets = [
        'dataset_easy_heavy.csv',
        'dataset_hard_heavy.csv', 
        'dataset_medium_heavy.csv',
        'dataset_balanced.csv'
    ]
    
    # Create JSONL files for each dataset
    for dataset_csv in datasets:
        if os.path.exists(dataset_csv):
            output_jsonl = dataset_csv.replace('.csv', '.jsonl')
            print(f"\nProcessing {dataset_csv}...")
            count = create_jsonl_for_dataset(dataset_csv, all_runs_data, output_jsonl)
            print(f"Created {output_jsonl} with {count} entries")
        else:
            print(f"Warning: {dataset_csv} not found")

if __name__ == "__main__":
    main() 