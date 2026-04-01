# Full Execution Guide: Advanced Pipeline

Because the new features rely on advanced Machine Learning packages (Hugging Face Transformers, PyTorch, SHAP), I created a safe, isolated virtual environment (`nlp_env`) in your folder. 

To run the full end-to-end pipeline, open your Terminal, make sure you are in your project folder (`~/Desktop/for_aahan`), and copy-paste these commands in this exact order:

> [!CAUTION]
> **Important:** Always activate the virtual environment first, otherwise your Mac will give you `ModuleNotFoundError` or permission errors!

## Step 0: Activate the Environment (Mandatory)
Activate the safe environment where all the heavy ML libraries are installed. You must do this every time you open a new terminal window to run these scripts:
```bash
source nlp_env/bin/activate
```
*(Your terminal prompt should now have `(nlp_env)` at the beginning).*

## Step 1: Extract Advanced NLP Features
Launch the script that downloads the Hugging Face models and processes your subreddit CSVs for Toxicity and Topic Drift. This will output a file called `nlp_weekly_features.csv`.
```bash
python nlp_feature_extraction.py
```
> [!NOTE]
> Since this script is running locally on your CPU instead of a GPU, it will take the longest. Let it run completely until it prints "SUCCESS".

## Step 2: Merge the NLP Features into the Dataset
Now that you have advanced metrics, run your pre-processor to merge the NLP text analysis with the behavioral metrics into `processed_subreddit_weekly_enriched.csv`.
```bash
python preprocess_data_weekly_enriched.py
```

## Step 3: Train the Dynamic/Temporal Graph Neural Network
Execute the new Temporal GNN sequence. This mathematically models the "splintering" of echo chambers dynamically over an advanced rolling time window.
```bash
python gnn_dynamic_training.py
```

## Step 4: Evaluate Predictive Models & SHAP Explanations
Run the notebook generator. This script trains SOTA Gradient Boosting algorithms and calculates **SHAP** feature intelligence.
```bash
python generate_notebook.py
```
*(You can then open `Community_Collapse_Modeling.ipynb` in your IDE or Jupyter to see the interactive charts and explanations).*

---

### Troubleshooting
If you want to clear the environment and restart from scratch:
```bash
rm -rf nlp_env
python3 -m venv nlp_env
source nlp_env/bin/activate
pip install transformers sentence-transformers shap setuptools torch pandas numpy scikit-learn
```
