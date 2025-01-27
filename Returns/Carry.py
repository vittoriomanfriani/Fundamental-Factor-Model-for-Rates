import pandas as pd

def compute_carry(df):
    df['carry'] = df['daily_coupons'] / df['prev_price']
    return df