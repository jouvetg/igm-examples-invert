import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor 

# Load the data
df = pd.read_csv("summary_metrics.csv")

# df = df[df["+experiment"].str.contains("params_arh", na=False)]

del df["core.hardware.visible_gpus"]

df[df.select_dtypes(include='bool').columns] = df.select_dtypes(include='bool').astype(int)

# Define your custom objective function
def objective(row):
    return row["stdthk"] + row["stdvel"]

# Apply it to the dataframe
df["objective"] = df.apply(objective, axis=1)

# Sort to find best configurations
best_configs = df.sort_values("objective")

best_configs.to_csv("summary_metrics_sorted.csv", index=False)

# Display the best configurations
print(best_configs[["run_dir", "objective"]])  # example: includes some parameter columns

#############################################

# Optional: visualize correlations
#metrics_cols = ["stdthk", "stdvel", "objective", "rmsvel", "rmsdiv"]
#sns.heatmap(df[metrics_cols].corr(), annot=True)
#plt.title("Metric Correlations")
#plt.show()

#########################################
 
# Step 2: Split into X (parameters) and y (objective)
print(df.columns)
dot_param_cols = [col for col in df.columns if '.' in col]
print("dot_param_cols", dot_param_cols)
X = df[dot_param_cols]
y = df["objective"]

print("X.shape", X.shape)
print("y.shape", y.shape)

# Step 3: Fit Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
 
###########################################

# Step 4: Compute and normalize feature importances
importances = pd.Series(model.feature_importances_, index=X.columns)
importances_percent = 100 * importances / importances.sum()
sorted_params = importances_percent.sort_values(ascending=False)

# Step 5: Get recommended values from the best config (lowest objective)
best_config = df.loc[df["objective"].idxmin()]
recommended_values = [best_config[param] for param in sorted_params.index]

# Step 6: Combine into a DataFrame
importance_with_values = pd.DataFrame({
    "Parameter": sorted_params.index,
    "Importance (%)": sorted_params.values,
    "Recommended Value": recommended_values
})

# Step 7: Display or print
print(importance_with_values)

########################################################

# Step 5: Format and sort for readability
importance_df = importances_percent.sort_values(ascending=False).reset_index()
importance_df.columns = ["Parameter", "Importance (%)"]
 
# Optional: Visualize
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Parameter"], importance_df["Importance (%)"])
plt.xlabel("Importance (%)")
plt.title("Parameter Importance for Objective Function")
plt.gca().invert_yaxis()  # Highest on top
plt.tight_layout() 
plt.savefig("feature_importance.png", dpi=300)
