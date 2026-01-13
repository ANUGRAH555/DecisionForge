import pandas as pd
from utils.supply_chain_model_training import train_supply_chain_models

# Load dataset
df = pd.read_csv("data/supply_chain_dataset_1.csv")

results, best_model = train_supply_chain_models(df)

print("\n📦 Supply Chain Model Performance:\n")

for model, metrics in results.items():
    print(f"{model}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("-" * 30)

print(f"\n🏆 Best Model Selected: {best_model}")
