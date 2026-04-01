import json
import os

with open('generate_notebook.py', 'r') as f:
    content = f.read()

# We need to insert the new cells right before `  ],\n "metadata": {`
search_str = '   }\n  ],\n  "metadata": {'

new_cells = '''   },
   {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
     "### VISUALIZATION 1: The Anatomy of a Collapse (The Real-World View)\\n",
     "Let's look at the lifecycle of a specific subreddit to visualize when the math flags a 'Collapse'."
    ]
   },
   {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
     "# Find a subreddit that has a collapse\\n",
     "df_model = df.dropna().sort_values(by='year_week').copy()\\n",
     "collapsed_subs = df_model[df_model['is_collapsed'] == 1]['subreddit'].unique()\\n",
     "target_sub = collapsed_subs[0] if len(collapsed_subs) > 0 else df_model['subreddit'].unique()[0]\\n",
     "sub_data = df_model[df_model['subreddit'] == target_sub].copy()\\n",
     "\\n",
     "plt.figure(figsize=(14, 6))\\n",
     "plt.plot(sub_data['year_week'], sub_data['engagement'], label='Actual Engagement (Weekly)', color='blue', alpha=0.6)\\n",
     "\\n",
     "historical_peak = sub_data['engagement'].expanding().max()\\n",
     "plt.plot(sub_data['year_week'], historical_peak, label='Historical Peak Normal', color='green', linestyle='--', alpha=0.7)\\n",
     "\\n",
     "collapse_points = sub_data[sub_data['is_collapsed'] == 1]\\n",
     "plt.scatter(collapse_points['year_week'], collapse_points['engagement'], color='red', label='Flagged as Collapsed', zorder=5)\\n",
     "\\n",
     "plt.title(f'Anatomy of a Collapse: r/{target_sub}')\\n",
     "plt.xlabel('Date')\\n",
     "plt.ylabel('Engagement (Comments + Posts)')\\n",
     "plt.legend()\\n",
     "plt.grid(True, alpha=0.3)\\n",
     "plt.show()"
    ]
   },
   {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
     "### VISUALIZATION 2: Predictive Early Warning Radar\\n",
     "Plotting the model's *Probability of Imminent Collapse* over time compared to the actual collapse event."
    ]
   },
   {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
     "gb_model = models['SOTA: Gradient Boosting']\\n",
     "sub_features = sub_data[[col for col in df_model.columns if 'lag' in col]]\\n",
     "sub_features_scaled = scaler.transform(sub_features)\\n",
     "sub_probs = gb_model.predict_proba(sub_features_scaled)[:, 1]\\n",
     "\\n",
     "plt.figure(figsize=(14, 6))\\n",
     "plt.plot(sub_data['year_week'], sub_probs, label='Predicted Probability of Collapse', color='purple', linewidth=2)\\n",
     "plt.axhline(y=0.5, color='orange', linestyle='--', label='50% Warning Threshold')\\n",
     "\\n",
     "for _, row in collapse_points.iterrows():\\n",
     "    plt.axvline(x=row['year_week'], color='red', alpha=0.1)\\n",
     "if len(collapse_points) > 0:\\n",
     "    plt.axvspan(collapse_points['year_week'].min(), collapse_points['year_week'].max(), color='red', alpha=0.2, label='Actual Collapse Period')\\n",
     "\\n",
     "plt.title(f'Predictive Early Warning Radar: r/{target_sub}')\\n",
     "plt.xlabel('Date')\\n",
     "plt.ylabel('Probability of Collapse')\\n",
     "plt.legend()\\n",
     "plt.grid(True, alpha=0.3)\\n",
     "plt.show()"
    ]
   },
   {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
     "### VISUALIZATION 3: Behavioral Degradation (What Dies First?)\\n",
     "Comparing the quality of human interaction between Healthy and Collapsed states."
    ]
   },
   {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
     "behavioral_cols = ['total_awards', 'total_crossposts', 'avg_score']\\n",
     "grouped = df.groupby('is_collapsed')[behavioral_cols].mean().reset_index()\\n",
     "\\n",
     "from sklearn.preprocessing import MinMaxScaler\\n",
     "scaler_minmax = MinMaxScaler()\\n",
     "grouped_scaled = grouped.copy()\\n",
     "grouped_scaled[behavioral_cols] = scaler_minmax.fit_transform(grouped[behavioral_cols].T).T # Normalize for visual comparison\\n",
     "melted = grouped_scaled.melt(id_vars='is_collapsed', value_vars=behavioral_cols)\\n",
     "melted['is_collapsed'] = melted['is_collapsed'].map({0: 'Healthy', 1: 'Collapsed'})\\n",
     "\\n",
     "plt.figure(figsize=(10, 6))\\n",
     "sns.barplot(data=melted, x='variable', y='value', hue='is_collapsed', palette=['#2ecc71', '#e74c3c'])\\n",
     "plt.title('Behavioral Degradation: Healthy vs Collapsed States (Normalized)')\\n",
     "plt.ylabel('Relative Frequency / Quality')\\n",
     "plt.xlabel('Behavioral Proxy')\\n",
     "\\n",
     "plt.show()"
    ]
   },
   {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
     "### VISUALIZATION 4: Production ML Metrics (Confusion Matrix & ROC)\\n",
     "Evaluating the Gradient Boosting classifier's performance mathematically."
    ]
   },
   {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
     "from sklearn.metrics import roc_curve, auc, RocCurveDisplay\\n",
     "\\n",
     "# We use gb_model trained earlier\\n",
     "gb_pred = gb_model.predict(X_test_scaled)\\n",
     "gb_probs = gb_model.predict_proba(X_test_scaled)[:, 1]\\n",
     "\\n",
     "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\\n",
     "\\n",
     "# Confusion Matrix Heatmap\\n",
     "cm = confusion_matrix(y_test, gb_pred)\\n",
     "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], \\n",
     "            xticklabels=['Predicted Healthy', 'Predicted Collapsed'],\\n",
     "            yticklabels=['Actual Healthy', 'Actual Collapsed'])\\n",
     "axes[0].set_title('Confusion Matrix (Gradient Boosting)')\\n",
     "\\n",
     "# ROC-AUC Curve\\n",
     "fpr, tpr, _ = roc_curve(y_test, gb_probs)\\n",
     "roc_auc = auc(fpr, tpr)\\n",
     "axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')\\n",
     "axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\\n",
     "axes[1].set_xlim([0.0, 1.0])\\n",
     "axes[1].set_ylim([0.0, 1.05])\\n",
     "axes[1].set_xlabel('False Positive Rate')\\n",
     "axes[1].set_ylabel('True Positive Rate')\\n",
     "axes[1].set_title('Receiver Operating Characteristic (ROC)')\\n",
     "axes[1].legend(loc=\"lower right\")\\n",
     "\\n",
     "plt.show()"
    ]
   }\n  ],\n  "metadata": {'

new_content = content.replace(search_str, new_cells)

with open('generate_notebook.py', 'w') as f:
    f.write(new_content)

print(f"Successfully modified: {search_str in content}")
