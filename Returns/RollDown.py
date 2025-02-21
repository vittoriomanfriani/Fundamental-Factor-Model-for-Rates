import QuantLib as ql
from Utils.conversions import pydatetime_to_quantlib_date
import SpotCurve.Spot_Curve_Calculator as scc
from tqdm import tqdm
import numpy as np
from Utils.data_processing import get_most_liquid_bond_by_interval

def price_bond(spot_curve_handle, coupon, issue_date, maturity_date, current_date):
    """
    Computes the clean price of a fixed-rate bond using a given spot yield curve.

    Parameters:
    ----------
    spot_curve_handle : QuantLib.YieldTermStructureHandle
        A handle to the spot yield curve used for discounting and pricing the bond.
    coupon : float
        The annual coupon rate of the bond (as a percentage, e.g., 5.0 for 5%).
    issue_date : datetime or string
        The bond's issue date in a format convertible to QuantLib's date format.
    maturity_date : datetime or string
        The bond's maturity date in a format convertible to QuantLib's date format.

    Returns:
    -------
    float
        The clean price of the bond based on the provided spot yield curve.

    Methodology:
    -----------
    1. Converts the `issue_date` and `maturity_date` to QuantLib date objects.
    2. Constructs a fixed-rate bond with the following parameters:
        - Semi-annual coupon payments.
        - Face value of 100.
        - Day count convention: Actual/Actual (ISDA).
        - Calendar: United States Government Bond market.
    3. Uses QuantLib's `DiscountingBondEngine` to price the bond based on the provided
       spot curve handle.
    4. Returns the computed clean price of the bond.

    Notes:
    ------
    - The bond's cash flow schedule is generated with backward date generation and
      does not adjust for month-end.
    - The spot curve must be a valid QuantLib `YieldTermStructureHandle`. If invalid,
      the function will raise an exception.
    """
    # Bond params
    calendar = ql.UnitedStates(m=ql.UnitedStates.GovernmentBond)
    current_date = calendar.adjust(pydatetime_to_quantlib_date(current_date))
    ql.Settings.instance().evaluationDate = current_date
    day_count = ql.ActualActual(ql.ActualActual.ISDA)
    issue_date = pydatetime_to_quantlib_date(issue_date)
    maturity_date = pydatetime_to_quantlib_date(maturity_date)
    tenor = ql.Period(ql.Semiannual)
    business_convention = ql.Unadjusted
    date_generation = ql.DateGeneration.Backward
    month_end = False

    schedule = ql.Schedule(issue_date,
                           maturity_date,
                           tenor,
                           calendar,
                           business_convention,
                           business_convention,
                           date_generation,
                           month_end)

    coupon_rate = [coupon/100]
    settlement_days = 1
    face_value = 100

    fixed_rate_bond = ql.FixedRateBond(settlement_days,
                                       face_value,
                                       schedule,
                                       coupon_rate,
                                       day_count)

    bond_engine = ql.DiscountingBondEngine(spot_curve_handle)
    fixed_rate_bond.setPricingEngine(bond_engine)

    return fixed_rate_bond.cleanPrice()

def compute_rolldown(df):
    """
    Computes the roll-down for bonds in a given DataFrame based on their prices and a
    bootstrapped yield curve.

    Roll-down measures the price change of a bond over a one-day horizon under the
    assumption of a constant term structure.

    Parameters:
    ----------
    df : pandas.DataFrame
        A multi-index DataFrame containing bond data with the following structure:
        - The first index level corresponds to the dates.
        - The second index level corresponds to bond identifiers.
        - Columns include 'coupon', 'issue_date', 'maturity_date', and 'price', among others.

    Returns:
    -------
    pandas.DataFrame
        The input DataFrame with an additional column named 'rolldown', containing the
        computed roll-down values for each bond on each date. If the roll-down cannot
        be computed (e.g., due to missing data), the value will be NaN.

    Methodology:
    -----------
    1. Iterates over each unique date in the DataFrame.
    2. Bootstraps a spot yield curve for the bonds available on that date.
    3. Advances to the next day and calculates the roll-down for each bond using:
        roll_down = (price_t / price_t1) - 1
        where:
        - `price_t1` is the bond price on the previous date.
        - `price_t` is the bond price on the current date derived using the spot curve.
    4. Handles potential errors gracefully (e.g., missing data or pricing errors).

    Notes:
    ------
    - The `spot_rates_calculator.curve_bootstrapper` function is used to construct the
      spot yield curve, and `price_bond` is used to compute bond prices from the curve.
    - The function assumes that the DataFrame is sorted by date in the index.
    """
    # Create an empty series to store roll-down results
    df.loc[:, 'rolldown'] = np.nan
    spot_rates_calculator = scc.SpotRatesCalculator()

    for i, past_date in enumerate(tqdm(df.index.get_level_values(0).unique())):

        curve_set_df = df.loc[past_date]
        otr = get_most_liquid_bond_by_interval(curve_set_df)

        # Spot Curve
        yc = spot_rates_calculator.curve_bootstrapper(otr, past_date, rolldown=True)
        spot_curve_handle = ql.YieldTermStructureHandle(yc)

        # Move froward by one day to compute roll-down if term structure is constant
        if i + 1 < len(df.index.get_level_values(0).unique()):
            current_date = df.index.get_level_values(0).unique()[i + 1]

        else:
            break

        for id in df.loc[current_date].index:
            bond_id = id
            coupon = df.loc[current_date, id]['coupon']
            issue_date = df.loc[current_date, id]['issue_date']
            maturity_date = df.loc[current_date, id]['maturity_date']

            # price at t-1
            try:
                #attempt to get price at t - 1
                price_t1 = price_bond(spot_curve_handle, coupon, issue_date, maturity_date, past_date)
                # Price at t
                price_t = price_bond(spot_curve_handle, coupon, issue_date, maturity_date, current_date)

                # Compute roll-down
                roll_down = price_t / price_t1 - 1

            except (KeyError, IndexError, ValueError, ZeroDivisionError) as e:
                # Handle errors from .loc or other computations
                roll_down = np.nan

            # Store the result in the roll_down_series
            df.at[(current_date, bond_id), 'rolldown'] = roll_down

    return df