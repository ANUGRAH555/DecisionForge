import pandas as pd
from utils.customer_model_training import train_customer_models

# Load sample churn dataset
df = pd.read_csv("data/customer_churn_dataset_1.csv")

results, best_model = train_customer_models(df)

print("\n Customer Churn Model Performance:\n")

for model, metrics in results.items():
    print(model)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("-" * 30)

print(f"\n Best Model Selected: {best_model}")
