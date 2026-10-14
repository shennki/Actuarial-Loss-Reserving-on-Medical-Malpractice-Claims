import numpy as np
import pandas as pd

df = pd.read_csv("medmal_pos_98-07.csv")
df.columns = df.columns.str.strip()

# Isolate a single company code
company_code = 683
company_data = df[df["GRCODE"] == company_code]

# Create the loss triangle using a pivot table
# Rows: Accident Year, Columns: Development Lag, Values: Cumulative Paid Losses
full_triangle = company_data.pivot(
    index="AccidentYear", columns="DevelopmentLag", values="CumPaidLoss"
)

# Choose an Evaluation Cutoff Lag
eval_lag = 5
truncated_triangle = full_triangle.iloc[:, :eval_lag]

print(f"--- Backtesting Model for Company {company_code} ---")
print(f"After cutting off data up to Development Lag {eval_lag}:\n")
print(truncated_triangle)

# Run chain ladder on the TRUNCATED triangle
factors = []
for i in range(truncated_triangle.shape[1] - 1):
  col_current = truncated_triangle.iloc[:, i]
  col_next = truncated_triangle.iloc[:, i + 1]
  valid_data = pd.concat([col_current, col_next], axis=1).dropna()
  factor = valid_data.iloc[:, 1].sum() / valid_data.iloc[:, 0].sum()
  factors.append(factor)

# Calculate age-to-ultimate factors for the truncated data
rev_factors = list(reversed(factors))
cum_rev = np.cumprod(rev_factors)
ultimate_factors = list(reversed(cum_rev))
while len(ultimate_factors) < truncated_triangle.shape[1]:
  ultimate_factors.append(1.0)
ultimate_factors.append(1.0)

# Project ultimate losses using ONLY the truncated data
latest_actuals_trunc = truncated_triangle.ffill(axis=1).iloc[:, -1]
n_actuals = len(latest_actuals_trunc)
if len(ultimate_factors) >= n_actuals:
  row_ultimate_factors = ultimate_factors[:n_actuals]
else:
  row_ultimate_factors = ultimate_factors + [1.0] * (
      n_actuals - len(ultimate_factors)
  )
projected_ultimates = latest_actuals_trunc.values * np.array(
    row_ultimate_factors
)

# Get the REAL actuals from the final column of the FULL triangle
actual_ultimates = full_triangle.iloc[:, -1].values

# Build a validation scorecard
validation_df = pd.DataFrame({
    "Accident Year": full_triangle.index,
    f"Paid at Lag {eval_lag}": latest_actuals_trunc.values,
    "Model Projected Ultimate": projected_ultimates,
    "Actual Ultimate (Lag 10)": actual_ultimates,
})
validation_df["Error ($)"] = (
    validation_df["Model Projected Ultimate"]
    - validation_df["Actual Ultimate (Lag 10)"]
)
validation_df["Error (%)"] = (
    validation_df["Error ($)"] / validation_df["Actual Ultimate (Lag 10)"]
) * 100

# Format dollar columns with commas and 2 decimal places
dollar_cols = [
    f"Paid at Lag {eval_lag}",
    "Model Projected Ultimate",
    "Actual Ultimate (Lag 10)",
    "Error ($)",
]
for col in dollar_cols:
  validation_df[col] = validation_df[col].map("${:,.2f}".format)
validation_df["Error (%)"] = validation_df["Error (%)"].map("{:,.2f}%".format)

# Row-by-row print formatting
print("\n--- Model Validation Scorecard ---")
for _, row in validation_df.iterrows():
  print(f"\nAccident Year          : {int(row['Accident Year'])}")
  print(
      f"Paid at Lag {eval_lag}         : {row[f'Paid at Lag {eval_lag}']}"
  )
  print(f"Model Projected Ultimate : {row['Model Projected Ultimate']}")
  print(f"Actual Ultimate (Lag 10) : {row['Actual Ultimate (Lag 10)']}")
  print(f"Error ($)                : {row['Error ($)']}")
  print(f"Error (%)                : {row['Error (%)']}")