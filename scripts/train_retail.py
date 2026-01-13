import pandas as pd
from utils.retail_model_training import train_retail_models

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
df = pd.read_csv("data/retail_dataset_1.csv")

# -------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------
results, best_model = train_retail_models(df)

# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------
print("\n📊 Retail Model Performance:\n")

for model, scores in results.items():
    print(f"{model}:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")
    print("-" * 30)

print(f"\n🏆 Best Model Selected: {best_model}")
