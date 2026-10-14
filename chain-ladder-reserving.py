import pandas as pd
import numpy as np

df = pd.read_csv("medmal_pos_98-07.csv")

# Iolsate a single company code
company_code = 683
company_data = df[df["GRCODE"] == company_code]

# Create the loss triangle using a pivot table
# Rows: Accident Year, Columns: Development Lag, Values: Cumulative Paid Losses
paid_triangle = company_data.pivot(
    index="AccidentYear", columns="DevelopmentLag", values="CumPaidLoss"
)

# Ensures output is printed in its entirety
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print(f"Full Loss Triangle for Company {company_code}:")
print(paid_triangle)

# Calculate age-to-age factors
factors=[]
for i in range(paid_triangle.shape[1] - 1):
    col_current = paid_triangle.iloc[:, i]
    col_next = paid_triangle.iloc[:, i + 1]

    # Only look at rows with complete data
    valid_data = pd.concat([col_current, col_next], axis=1).dropna()
    sum_current = valid_data.iloc[:, 0].sum()
    sum_next = valid_data.iloc[:, 1].sum()
    factor = sum_next / sum_current
    factors.append(factor)

print("\nAge-to-Age Development Factors (Link Ratios):")
for idx, f in enumerate(factors, start=1):
    print(f"Lag {idx} to Lag {idx+1}: {f:.4f}")

# Calculate age-to-ultimate (cumulitive) factors
# This is calculated from right to left
rev_factors = list(reversed(factors))
cum_rev = np.cumprod(rev_factors)
ultimate_factors = list(reversed(cum_rev))

# Add the final ultimate factor of 1.0 for the last lag (Lag 10 needs no further growth)
ultimate_factors.append(1.0)

print("\nAge-to-Ultimate (Cumulative) Factors:")
for idx, uf in enumerate(ultimate_factors, start=1):
    print(f"Lag {idx} to Ultimate: {uf:.4f}")

# Calculate total losses and IBNR
# Find the latest available (diagonal) paid loss for each accident year
latest_actuals = paid_triangle.ffill(axis=1).iloc[:, -1]

# Map the ultimate factors to each accident year
# (Each row corresponds to how many periods of development it has undergone)
n_rows = len(paid_triangle)
row_ultimate_factors = ultimate_factors[:n_rows]

# Calculate ultimate losses
ultimate_losses = latest_actuals.values * row_ultimate_factors

# Create a summary dataframe
summary_df = pd.DataFrame(
    {
        "Accident Year": paid_triangle.index,
        "Latest Paid": latest_actuals.values,
        "Ult. Factor": row_ultimate_factors,
        "Ultimate Loss": ultimate_losses,
        "IBNR Reserve": ultimate_losses - latest_actuals.values,
    }
)

print("\n--- Final Reserving Summary ---")
print(summary_df.to_string(index=False))

# Total IBNR required across all years
total_ibnr = summary_df["IBNR Reserve"].sum()

print(f"\nTotal Indicated IBNR Reserve needed: ${total_ibnr:,.2f}")