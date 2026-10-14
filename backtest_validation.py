import numpy as np
import pandas as pd

df = pd.read_csv("medmal_pos_98-07.csv")
df.columns = df.columns.str.strip()

# Isolate a single company code
company_code = 683
company_data = df[df["GRCODE"] == company_code]

# Create the loss triangle using a pivot table
full_triangle = company_data.pivot(
    index="AccidentYear", columns="DevelopmentLag", values="CumPaidLoss"
)

# Simulate a Diagonal Triangle Cut (Stair-step truncation)
n_rows, n_cols = full_triangle.shape
truncated_triangle = full_triangle.copy()

for r_idx in range(n_rows):
  max_allowed_col = n_cols - r_idx
  if max_allowed_col < n_cols:
    truncated_triangle.iloc[r_idx, max_allowed_col:] = np.nan

print(f"--- Backtesting BF vs Chain Ladder for Company {company_code} ---\n")

# Run Chain Ladder on the TRUNCATED triangle to get factors
factors = []
for i in range(truncated_triangle.shape[1] - 1):
  col_current = truncated_triangle.iloc[:, i]
  col_next = truncated_triangle.iloc[:, i + 1]
  valid_data = pd.concat([col_current, col_next], axis=1).dropna()
  if valid_data.empty:
    factors.append(1.0)
    continue
  factor = valid_data.iloc[:, 1].sum() / valid_data.iloc[:, 0].sum()
  factors.append(factor)

# Calculate age-to-ultimate factors for the truncated data
rev_factors = list(reversed(factors))
cum_rev = np.cumprod(rev_factors)
ultimate_factors = list(reversed(cum_rev))
while len(ultimate_factors) < truncated_triangle.shape[1]:
  ultimate_factors.append(1.0)
ultimate_factors.append(1.0)

# Get latest actuals from truncated data
latest_actuals_trunc = truncated_triangle.ffill(axis=1).iloc[:, -1]
n_actuals = len(latest_actuals_trunc)

# Map ultimate factors based on each row's actual development lag
row_lags = truncated_triangle.notna().sum(axis=1).values
row_ultimate_factors = []
for lag in row_lags:
  idx = int(lag) - 1
  if idx < len(ultimate_factors):
    row_ultimate_factors.append(ultimate_factors[idx])
  else:
    row_ultimate_factors.append(1.0)
row_ultimate_factors = np.array(row_ultimate_factors)

# Standard Chain Ladder Projections
cl_projected_ultimates = latest_actuals_trunc.values * row_ultimate_factors

# Implement Bornhuetter-Ferguson (BF)
actual_ultimates = full_triangle.iloc[:, -1].values
avg_expected_loss = np.mean(actual_ultimates)
expected_losses = np.array([avg_expected_loss] * n_actuals)

cdfs = row_ultimate_factors
percent_unpaid = 1.0 - (1.0 / cdfs)

# BF Ultimate = Latest Paid + (Expected Losses * Percent Unpaid)
bf_projected_ultimates = (
    latest_actuals_trunc.values + expected_losses * percent_unpaid
)

# Build validation scorecard comparing both methods
validation_df = pd.DataFrame({
    "Accident Year": full_triangle.index,
    "Latest Paid (At Cut)": latest_actuals_trunc.values,
    "CL Projected Ultimate": cl_projected_ultimates,
    "BF Projected Ultimate": bf_projected_ultimates,
    "Actual Ultimate (Lag 10)": actual_ultimates,
})

validation_df["CL Error (%)"] = (
    (validation_df["CL Projected Ultimate"] - validation_df["Actual Ultimate (Lag 10)"])
    / validation_df["Actual Ultimate (Lag 10)"]
) * 100

validation_df["BF Error (%)"] = (
    (validation_df["BF Projected Ultimate"] - validation_df["Actual Ultimate (Lag 10)"])
    / validation_df["Actual Ultimate (Lag 10)"]
) * 100

# Format dollar columns
dollar_cols = [
    "Latest Paid (At Cut)",
    "CL Projected Ultimate",
    "BF Projected Ultimate",
    "Actual Ultimate (Lag 10)",
]
for col in dollar_cols:
  validation_df[col] = validation_df[col].map("${:,.2f}".format)

validation_df["CL Error (%)"] = validation_df["CL Error (%)"].map("{:,.2f}%".format)
validation_df["BF Error (%)"] = validation_df["BF Error (%)"].map("{:,.2f}%".format)

# Row-by-row print formatting
print("--- Model Validation Scorecard: Chain Ladder vs. Bornhuetter-Ferguson ---")
for _, row in validation_df.iterrows():
  print(f"\nAccident Year          : {int(row['Accident Year'])}")
  print(f"Latest Paid (At Cut)   : {row['Latest Paid (At Cut)']}")
  print(f"CL Projected Ultimate  : {row['CL Projected Ultimate']} (Error: {row['CL Error (%)']})")
  print(f"BF Projected Ultimate  : {row['BF Projected Ultimate']} (Error: {row['BF Error (%)']})")
  print(f"Actual Ultimate (Lag 10) : {row['Actual Ultimate (Lag 10)']}")