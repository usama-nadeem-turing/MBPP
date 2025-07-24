import pandas as pd
import random
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def classify_difficulty(pass_at_1):
    """Classify difficulty based on pass@1 value"""
    if pass_at_1 >= 0.75:
        return "Easy"
    elif 0.30 <= pass_at_1 < 0.75:
        return "Medium"
    else:
        return "Hard"

def create_dataset(data, easy_pct, medium_pct, hard_pct, dataset_name):
    """Create a dataset with specified difficulty distribution"""
    # Calculate target counts
    total_samples = len(data)
    target_easy = int(total_samples * easy_pct)
    target_medium = int(total_samples * medium_pct)
    target_hard = int(total_samples * hard_pct)
    
    # Get available samples for each difficulty
    easy_samples = data[data['difficulty'] == 'Easy'].copy()
    medium_samples = data[data['difficulty'] == 'Medium'].copy()
    hard_samples = data[data['difficulty'] == 'Hard'].copy()
    
    print(f"\n{dataset_name} Dataset:")
    print(f"Available samples - Easy: {len(easy_samples)}, Medium: {len(medium_samples)}, Hard: {len(hard_samples)}")
    print(f"Target samples - Easy: {target_easy}, Medium: {target_medium}, Hard: {target_hard}")
    
    # Sample from each difficulty level
    selected_easy = easy_samples.sample(n=min(target_easy, len(easy_samples)), random_state=42)
    selected_medium = medium_samples.sample(n=min(target_medium, len(medium_samples)), random_state=42)
    selected_hard = hard_samples.sample(n=min(target_hard, len(hard_samples)), random_state=42)
    
    # Combine all selected samples
    dataset = pd.concat([selected_easy, selected_medium, selected_hard], ignore_index=True)
    
    # Shuffle the dataset
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate actual distribution
    actual_dist = dataset['difficulty'].value_counts(normalize=True)
    print(f"Actual distribution - Easy: {actual_dist.get('Easy', 0):.1%}, Medium: {actual_dist.get('Medium', 0):.1%}, Hard: {actual_dist.get('Hard', 0):.1%}")
    
    return dataset

def main():
    # Read the CSV file
    df = pd.read_csv('../pass@k.csv')
    
    # Specific problem IDs to consider
    target_ids = [602,603,608,609,611,612,617,619,626,629,630,636,638,640,641,642,644,646,647,651,652,656,657,659,661,663,665,668,671,677,684,685,689,690,695,696,699,702,706,708,709,712,714,717,721,722,734,735,738,739,743,745,746,751,752,754,757,758,761,763,765,768,769,773,776,777,780,782,784,790,794,797,799,805,810,815,819,831,832,834,836,838,840,844,849,851,853,856,859,863,867,872,873,874,878,880,881,882,890,893,898,900,901,903,906,907,908,909,910,912,914,915,916,917,918,920,922,923,925,926,927,934,936,938,940,943,944,949,950,951,952,953,956,957,959,963,967,968,969,971]
    
    # Filter for specific mbpp_ids
    df = df[df['mbpp_id'].isin(target_ids)].copy()
    
    # Add difficulty classification
    df['difficulty'] = df['pass@1'].apply(classify_difficulty)
    
    # Print overall statistics
    print("Overall Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    difficulty_counts = df['difficulty'].value_counts()
    difficulty_pcts = df['difficulty'].value_counts(normalize=True)
    
    for difficulty in ['Easy', 'Medium', 'Hard']:
        count = difficulty_counts.get(difficulty, 0)
        pct = difficulty_pcts.get(difficulty, 0)
        print(f"{difficulty}: {count} samples ({pct:.1%})")
    
    # Create the 4 datasets
    datasets = {
        'Easy-heavy': create_dataset(df, 0.70, 0.20, 0.10, 'Easy-heavy'),
        'Hard-heavy': create_dataset(df, 0.10, 0.20, 0.70, 'Hard-heavy'),
        'Medium-heavy': create_dataset(df, 0.10, 0.70, 0.20, 'Medium-heavy'),
        'Balanced': create_dataset(df, 0.33, 0.33, 0.34, 'Balanced')  # 34% for hard to account for rounding
    }
    
    # Save datasets
    for name, dataset in datasets.items():
        filename = f'dataset_{name.lower().replace("-", "_")}.csv'
        dataset.to_csv(filename, index=False)
        print(f"\nSaved {filename} with {len(dataset)} samples")
        
        # Print sample statistics for each dataset
        print(f"{name} Dataset Statistics:")
        print(f"Pass@1 mean: {dataset['pass@1'].mean():.3f}")
        print(f"Pass@5 mean: {dataset['pass@5'].mean():.3f}")
        print(f"Pass@10 mean: {dataset['pass@10'].mean():.3f}")
    
    # Create a summary file
    summary_data = []
    for name, dataset in datasets.items():
        summary_data.append({
            'Dataset': name,
            'Total_Samples': len(dataset),
            'Easy_Count': len(dataset[dataset['difficulty'] == 'Easy']),
            'Medium_Count': len(dataset[dataset['difficulty'] == 'Medium']),
            'Hard_Count': len(dataset[dataset['difficulty'] == 'Hard']),
            'Easy_Pct': len(dataset[dataset['difficulty'] == 'Easy']) / len(dataset),
            'Medium_Pct': len(dataset[dataset['difficulty'] == 'Medium']) / len(dataset),
            'Hard_Pct': len(dataset[dataset['difficulty'] == 'Hard']) / len(dataset),
            'Avg_Pass@1': dataset['pass@1'].mean(),
            'Avg_Pass@5': dataset['pass@5'].mean(),
            'Avg_Pass@10': dataset['pass@10'].mean()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('datasets_summary.csv', index=False)
    print(f"\nSaved datasets_summary.csv with overview of all datasets")

if __name__ == "__main__":
    main() 