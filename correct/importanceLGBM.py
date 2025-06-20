import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# set theme
sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)

# define directories
base_dir = r"C:\Users\sroes\PycharmProjects\PythonProject\correct\results\LGBM"
output_dir = os.path.join(base_dir, "PLOTS", "IMPORTANCE")
os.makedirs(output_dir, exist_ok=True)

# parameters
top_n = 10
model_prefix = "LightGBM"

# gather all importance files
all_files = glob(os.path.join(base_dir, f"{model_prefix}_*_fold*_importance.csv"))

# group by scenario
scenario_groups = {} # dictionary to hold scenario groups
for file_path in all_files: # iterate through all files
    filename = os.path.basename(file_path) # get the filename from the path
    scenario_name = filename.replace(f"{model_prefix}_", "").split("_fold")[0] # extract scenario name
    scenario_groups.setdefault(scenario_name, []).append(file_path) # group files by scenario name

# process each scenario
for scenario, paths in scenario_groups.items(): # iterate through each scenario and its file paths
    importance_dfs = [] # list to hold importances
    for path in paths: # iterate through each filepath in the scenario
        df = pd.read_csv(path) # read the CSV
        importance_dfs.append(df) # append each fold's importance DataFrame to the list

    # Combine and average
    combined_df = pd.concat(importance_dfs) # concatenate to combine importances in one table
    mean_importance = combined_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False).reset_index() # group by feature and calculate mean importance
    top_features = mean_importance.head(top_n) # get the top N features

    # plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_features, y="Feature", x="Importance", palette="Blues_d")
    plt.xlabel("Mean Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    filename = f"feature_importance_{scenario.replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300) # save the plot
    plt.close()