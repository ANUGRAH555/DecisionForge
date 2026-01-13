import pandas as pd
from utils.hr_model_training import train_hr_models

df = pd.read_csv("data/hr_dataset_100rows_1.csv")

results, best_model = train_hr_models(df)

print("Model Performance:")
for model, scores in results.items():
    print(model, scores)

print("Best Model:", best_model)
