import pandas as pd
import numpy as np
import os
import glob
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import warnings
warnings.filterwarnings('ignore')

def extract_nlp_features(data_dir: str, output_file: str, sample_size: int = 100):
    print("==========================================")
    print("Starting Advanced NLP Feature Extraction")
    print(f"Sampling up to {sample_size} posts per week per subreddit")
    print("Running strictly on CPU as requested...")
    print("==========================================")

    device = -1 # CPU
    
    print("Loading Hugging Face Models...")
    # Toxicity pipeline
    try:
        # toxic-bert outputs probabilities for toxic, severe_toxic, obscene, threat, insult, identity_hate
        toxicity_pipe = pipeline("text-classification", model="unitary/toxic-bert", device=device, top_k=None)
    except Exception as e:
        print(f"Could not load unitary/toxic-bert, falling back to basic sentiment: {e}")
        toxicity_pipe = pipeline("sentiment-analysis", device=device)
        
    # Topic Embeddings
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    print("Models Loaded.")

    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    all_files = [f for f in all_files if '50_subreddits_list' not in f]
    
    nlp_features_list = []

    for file in all_files:
        subreddit_name = os.path.basename(file).replace('.csv', '')
        print(f"\\nProcessing NLP for r/{subreddit_name}...")
        
        try:
            df = pd.read_csv(file, parse_dates=['created_utc'], low_memory=False)
            
            # Use 'body' or 'title' if body missing
            if 'body' not in df.columns:
                if 'title' in df.columns:
                    df['body'] = df['title']
                else:
                    continue
            
            # Filter non-strings and keep reasonably sized bodies
            df['body'] = df['body'].astype(str)
            df = df[df['body'].str.len() > 10].copy()
            
            df['year_week'] = df['created_utc'].dt.to_period('W')
            
            # Group by week
            weeks = df['year_week'].unique()
            weeks = sorted(weeks)
            
            prev_week_embedding = None
            
            for week in weeks:
                week_data = df[df['year_week'] == week]
                # Sample for performance
                if len(week_data) > sample_size:
                    week_data = week_data.sample(n=sample_size, random_state=42)
                
                texts = week_data['body'].tolist()
                
                # Truncate texts to avoid tokenizer errors (max 512 tokens -> roughly 400 words)
                texts = [" ".join(t.split()[:400]) for t in texts]
                
                if len(texts) == 0:
                    continue

                # 1. Toxicity
                try:
                    results = toxicity_pipe(texts, truncation=True, max_length=512)
                    week_toxic_scores = []
                    for res in results:
                        # If using toxic-bert (returns list of dicts with score per label)
                        if isinstance(res, list) and isinstance(res[0], dict) and 'label' in res[0]:
                            # Sum probabilities of toxicity
                            toxic_prob = sum([r['score'] for r in res if r['label'] in ['toxic', 'severe_toxic', 'insult']])
                            week_toxic_scores.append(toxic_prob)
                        elif isinstance(res, dict) and res.get('label') == 'NEGATIVE':
                            week_toxic_scores.append(res['score'])
                        else:
                            week_toxic_scores.append(0.0)
                            
                    avg_toxicity = np.mean(week_toxic_scores) if week_toxic_scores else 0.0
                except Exception as e:
                    avg_toxicity = 0.0

                # 2. Topic Drift (Embeddings)
                try:
                    embeddings = embed_model.encode(texts, show_progress_bar=False)
                    avg_embedding = np.mean(embeddings, axis=0) # Vector representing the average "topic" this week
                    
                    if prev_week_embedding is not None:
                        # Cosine similarity
                        sim = np.dot(avg_embedding, prev_week_embedding) / (np.linalg.norm(avg_embedding) * np.linalg.norm(prev_week_embedding) + 1e-10)
                        topic_drift = 1.0 - sim
                    else:
                        topic_drift = 0.0
                        
                    prev_week_embedding = avg_embedding
                except Exception:
                    topic_drift = 0.0
                
                nlp_features_list.append({
                    'subreddit': subreddit_name,
                    'year_week': week.to_timestamp(),
                    'avg_toxicity': avg_toxicity,
                    'topic_drift': topic_drift
                })
                
        except Exception as e:
            print(f"Error processing {subreddit_name}: {e}")

    final_df = pd.DataFrame(nlp_features_list)
    final_df.to_csv(output_file, index=False)
    print(f"\\nSUCCESS: NLP features computed and saved to {output_file}!")

if __name__ == "__main__":
    DATA_DIRECTORY = 'data'
    OUTPUT_FILE = 'data/nlp_weekly_features.csv'
    # For testing, we can limit to just a couple subreddits to see it work.
    # The script will process all if run entirely.
    extract_nlp_features(DATA_DIRECTORY, OUTPUT_FILE, sample_size=100)
