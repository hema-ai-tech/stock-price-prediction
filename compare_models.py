
import pandas as pd
import matplotlib.pyplot as plt

# Model evaluation results
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [20.15, 21.68],
    "RMSE": [26.02, 27.35],
    "R2 Score": [0.9339, 0.9270]
})

print("=" * 55)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 55)
print(results.to_string(index=False))

# Save comparison results
results.to_csv("data/model_comparison.csv", index=False)
print("\nComparison saved to data/model_comparison.csv")

# Plot comparison metrics
metrics = ["MAE", "RMSE", "R2 Score"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, metric in zip(axes, metrics):
    ax.bar(results["Model"], results[metric])
    ax.set_title(metric)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("data/model_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("Comparison chart saved to data/model_comparison.png")
print("\nSTEP 9 COMPLETED SUCCESSFULLY!")
