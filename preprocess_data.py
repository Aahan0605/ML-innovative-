import pandas as pd
import glob
import os

def preprocess_and_aggregate_data(data_dir: str, output_file: str):
    """
    Reads 50 subreddit CSV files, filters bots, aggregates engagement 
    stats by month, and outputs a single preprocessed time-series dataset.
    """
    print("Starting Data Preprocessing Pipeline...")
    
    # Get all CSVs in the data folder, excluding the master list
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    all_files = [f for f in all_files if '50_subreddits_list' not in f]
    
    monthly_stats = []
    
    # Process file by file to preserve memory (important for large datasets)
    for file in all_files:
        try:
            print(f"Ingesting & Processing: {os.path.basename(file)}...")
            
            # 1. READ CSV (Parse timestamps immediately)
            # Use chunks if memory becomes an issue, but for now try whole file
            df = pd.read_csv(file, parse_dates=['created_utc'], low_memory=False)
            
            # 2. FILTER NOISE
            if 'is_bot' in df.columns:
                df = df[df['is_bot'] == False]
                
            # If subreddit column is completely missing or wrong, make sure we have it
            subreddit_name = os.path.basename(file).replace('.csv', '')
            if 'subreddit' not in df.columns or df['subreddit'].isnull().all():
                df['subreddit'] = subreddit_name
            
            # 3. TEMPORAL EXTRACTION (Year-Month Level)
            # This turns granular times into "2024-03" for grouping
            df['year_month'] = df['created_utc'].dt.to_period('M')
            
            # 4. AGGREGATE CORE METRICS
            grouped = df.groupby(['subreddit', 'year_month']).agg(
                post_count=('id', 'count'),                        # Volume Features
                total_comments=('num_comments', 'sum'),            # Engagement Volume
                avg_comments_per_post=('num_comments', 'mean'),    # Engagement Quality
                total_score=('score', 'sum'),                      # Consensus
                avg_score=('score', 'mean'),                       # Average Quality
                avg_upvote_ratio=('upvote_ratio', 'mean'),         # Controversy proxy
                nsfw_count=('is_nsfw', 'sum')                      # Specific feature
            ).reset_index()
            
            monthly_stats.append(grouped)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")

    # 5. COMBINE ALL SUBREDDITS
    print("\nConcatenating all processed subreddits into master dataset...")
    final_df = pd.concat(monthly_stats, ignore_index=True)
    
    # Convert period back to timestamp for easier saving/loading/plotting later
    final_df['year_month'] = final_df['year_month'].dt.to_timestamp()
    
    # Sort chronologically by subreddit and time
    final_df = final_df.sort_values(by=['subreddit', 'year_month']).reset_index(drop=True)
    
    # Save to disk
    final_df.to_csv(output_file, index=False)
    print(f"\n==========================================")
    print(f"SUCCESS: Preprocessing Complete!")
    print(f"Total Rows generated (Community-Months): {len(final_df)}")
    print(f"Saved to: {output_file}")
    print(final_df.head(10))

if __name__ == "__main__":
    DATA_DIRECTORY = 'data'
    OUTPUT_FILE = 'processed_subreddit_time_series.csv'
    preprocess_and_aggregate_data(DATA_DIRECTORY, OUTPUT_FILE)
