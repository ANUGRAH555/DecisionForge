import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/hr_dataset_100rows_1.csv")

# Add Attrition column (simple realistic logic)
df["Attrition"] = np.where(
    (df["OverTime"] == "Yes") &
    (df["JobSatisfaction"] <= 2) &
    (df["MonthlyIncome"] < 40000),
    "Yes",
    "No"
)

# Save back
df.to_csv("data/hr_dataset_100rows_1.csv", index=False)

print("✅ Attrition column added successfully")
