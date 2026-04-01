import pandas as pd
import glob
import os

def preprocess_weekly_enriched(data_dir: str, output_file: str):
    """
    Reads 50 subreddit CSV files, filters bots, aggregates engagement 
    stats by WEEK instead of MONTH to increase dataset size, and includes 
    new columns like awards, crossposts, and subscribers.
    """
    print("Starting Weekly Enriched Preprocessing Pipeline...")
    
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    all_files = [f for f in all_files if '50_subreddits_list' not in f]
    
    weekly_stats = []
    
    for file in all_files:
        try:
            # Use chunks if memory becomes an issue, but for now try whole file
            df = pd.read_csv(file, parse_dates=['created_utc'], low_memory=False)
            
            # FILTER NOISE
            if 'is_bot' in df.columns:
                df = df[df['is_bot'] == False]
                
            subreddit_name = os.path.basename(file).replace('.csv', '')
            if 'subreddit' not in df.columns or df['subreddit'].isnull().all():
                df['subreddit'] = subreddit_name
            
            # TEMPORAL EXTRACTION (Year-Week Level) to increase data rows (approx 4x)
            # D = Daily, W = Weekly
            df['year_week'] = df['created_utc'].dt.to_period('W')
            
            # Ensure columns exist before aggregation to avoid errors
            cols_to_check = ['score', 'num_comments', 'upvote_ratio', 'is_nsfw', 'num_awards', 'num_crossposts', 'subscribers']
            for col in cols_to_check:
                if col not in df.columns:
                    df[col] = 0
            
            # AGGREGATE CORE & NEW METRICS
            grouped = df.groupby(['subreddit', 'year_week']).agg(
                post_count=('id', 'count'),                        
                total_comments=('num_comments', 'sum'),            
                avg_comments_per_post=('num_comments', 'mean'),    
                total_score=('score', 'sum'),                      
                avg_score=('score', 'mean'),                       
                avg_upvote_ratio=('upvote_ratio', 'mean'),         
                total_awards=('num_awards', 'sum'),                # NEW: Community appreciation
                total_crossposts=('num_crossposts', 'sum'),        # NEW: Network sharing
                avg_subscribers=('subscribers', 'mean'),           # NEW: Community size limit
                nsfw_count=('is_nsfw', 'sum')                      
            ).reset_index()
            
            weekly_stats.append(grouped)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")

    # COMBINE ALL SUBREDDITS
    print("\nConcatenating all weekly processed datasets...")
    final_df = pd.concat(weekly_stats, ignore_index=True)
    
    # Convert period back to timestamp for ML tools
    final_df['year_week'] = final_df['year_week'].dt.to_timestamp()
    
    # Sort chronologically
    final_df = final_df.sort_values(by=['subreddit', 'year_week']).reset_index(drop=True)
    
    # MERGE NLP FEATURES
    nlp_path = 'nlp_weekly_features.csv'
    if os.path.exists(nlp_path):
        print("\nMerging newly extracted Advanced NLP features...")
        nlp_df = pd.read_csv(nlp_path)
        nlp_df['year_week'] = pd.to_datetime(nlp_df['year_week'])
        
        final_df = final_df.merge(nlp_df, on=['subreddit', 'year_week'], how='left')
        final_df['avg_toxicity'] = final_df['avg_toxicity'].fillna(0.0)
        final_df['topic_drift'] = final_df['topic_drift'].fillna(0.0)
        print("NLP features successfully merged into dataset.")

    # Save to disk
    final_df.to_csv(output_file, index=False)
    print(f"\n==========================================")
    print(f"SUCCESS: Weekly Enriched Preprocessing Complete!")
    print(f"Total Rows generated (Community-Weeks): {len(final_df)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    DATA_DIRECTORY = 'data'
    OUTPUT_FILE = 'processed_subreddit_weekly_enriched.csv'
    preprocess_weekly_enriched(DATA_DIRECTORY, OUTPUT_FILE)
