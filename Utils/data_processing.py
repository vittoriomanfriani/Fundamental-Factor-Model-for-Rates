import pandas as pd
import numpy as np

def process_data(df):
    df = df[['coupon', 'maturity', 'yield', 'price','time to maturity', 'issue_date']]
    df = df.rename(columns = {
    "maturity": "maturity_date"
    })
    return df

def filter_on_the_run_bonds(df):
    """Filter for on-the-run bonds."""
    df = df.loc[df["time to maturity"] > 0.02].copy()
    df_reset = df.reset_index()
    df_reset["issue_date"] = pd.to_datetime(df_reset["issue_date"])
    df_reset["maturity_date"] = pd.to_datetime(df_reset["maturity_date"])

    on_the_run_bonds = df_reset.loc[
        df_reset.groupby(["timestamp", "maturity_date"])["issue_date"].idxmax()
        ]
    return on_the_run_bonds.set_index(["timestamp", "id"])

def get_most_liquid_bond_by_interval(df, max_years=30):
    """gets the most liquid bond for each year of time to maturity"""

    df = df.loc[df["time to maturity"] > 0.02].copy() # Discard smallest maturities
    filtered_bonds = []  # List to store the selected bonds
    first_bond = df.sort_values(by='time to maturity', ascending=True).iloc[0]
    filtered_bonds.append(first_bond) # Add bond with smallest maturity - will be useful for interpolation

    # Iterate over each yearly interval
    for year in range(max_years):
        lower_bound = year
        upper_bound = year + 1

        # Filter bonds in the current time-to-maturity interval
        interval_bonds = df[(df['time to maturity'] > lower_bound) & (df['time to maturity'] <= upper_bound)]

        if not interval_bonds.empty:
            # Sort bonds by issue_date in descending order and select the latest issued bond
            latest_bond = interval_bonds.sort_values(by='issue_date', ascending=False).iloc[0]
            filtered_bonds.append(latest_bond)

    # Combine the selected bonds into a single DataFrame
    result_df = pd.DataFrame(filtered_bonds)

    return result_df



