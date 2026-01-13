import pandas as pd

from utils.banking_model_training import train_banking_models

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
# You can change this to any of the 10 banking datasets
df = pd.read_csv("data/banking_sample_1.csv")

# -------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------
results, best_model = train_banking_models(df)

# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------
print("\n📊 Banking Fraud Model Performance:\n")

for model_name, metrics in results.items():
    print(f"{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("-" * 30)

print(f"\n🏆 Best Model Selected: {best_model}")
