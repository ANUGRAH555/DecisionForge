import sys
import os
import pandas as pd

# -------------------------------------------------
# Fix path so utils/ is found
# -------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.insurance_model_training import train_insurance_models

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("data/insurance_dataset_1.csv")

# -------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------
results, best_model = train_insurance_models(df)

# -------------------------------------------------
# OUTPUT
# -------------------------------------------------
print("\nInsurance Model Performance:\n")
for model, scores in results.items():
    print(f"{model}: {scores}")

print("\nBest Model:", best_model)
