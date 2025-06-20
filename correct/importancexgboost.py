import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# === Setup ===
sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)

base_dir = r"C:\Users\sroes\PycharmProjects\PythonProject\correct\results\XGboost"
output_dir = os.path.join(base_dir, "PLOTS", "IMPORTANCE")
os.makedirs(output_dir, exist_ok=True)

top_n = 10  # Number of top features to show
model_prefix = "XGBoost"

all_files = glob(os.path.join(base_dir, f"{model_prefix}_*_fold*_importance.csv"))

# group by scenario
scenario_groups = {}
for file_path in all_files:
    filename = os.path.basename(file_path)
    scenario_name = filename.replace(f"{model_prefix}_", "").split("_fold")[0]
    scenario_groups.setdefault(scenario_name, []).append(file_path)

# process each scenario
for scenario, paths in scenario_groups.items():
    importance_dfs = []
    for path in paths:
        df = pd.read_csv(path)
        importance_dfs.append(df)

    # Combine and average
    combined_df = pd.concat(importance_dfs)
    mean_importance = combined_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False).reset_index()
    top_features = mean_importance.head(top_n)

    # === Plot ===
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_features, y="Feature", x="Importance", palette="Blues_d")
    plt.xlabel("Mean Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    filename = f"feature_importance_{scenario.replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()