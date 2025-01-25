import pandas as pd
import QuantLib as ql

def get_spot_rates_on_tenors(yieldcurve, day_count):
    """Generate spot rates on specific tenors for given yield curve."""
    spots = []
    tenors = []
    ref_date = yieldcurve.referenceDate()
    dates = yieldcurve.dates()

    for i, d in enumerate(dates):
        yrs = day_count.yearFraction(ref_date, d)
        compounding = ql.Compounded
        freq = ql.Semiannual

        zero_rate = yieldcurve.zeroRate(yrs, compounding, freq)
        eq_rate = zero_rate.equivalentRate(day_count, compounding, freq, ref_date, d).rate()
        tenors.append(yrs)
        spots.append(100 * eq_rate)

    return pd.DataFrame(list(zip(tenors, spots)), columns=["Maturities", "Curve"], index=[''] * len(tenors))[1:]


def get_monthly_spot_rates(yieldcurve, day_count,
                   calendar=ql.UnitedStates(ql.UnitedStates.GovernmentBond), months=361):
    """Generate monthly spot rates for given yield curve."""
    spots = []
    tenors = []
    ref_date = yieldcurve.referenceDate()
    for month in range(0, months):
        yrs = month / 12.0
        d = calendar.advance(ref_date, ql.Period(month, ql.Months))
        compounding = ql.Compounded
        freq = ql.Semiannual

        zero_rate = yieldcurve.zeroRate(yrs, compounding, freq)
        eq_rate = zero_rate.equivalentRate(
                day_count, compounding, freq, ref_date, d
            ).rate()
        tenors.append(yrs)
        spots.append(100 * eq_rate)

    return pd.DataFrame(list(zip(tenors, spots)),
                        columns=["Maturities", "Curve"],
                        index=[''] * len(tenors))[1:]

def get_daily_spot_rates(yieldcurve, day_count,
                   calendar=ql.UnitedStates(ql.UnitedStates.GovernmentBond), days=361*30):
    """Generate daily spot rates for given yield curve."""
    spots = []
    tenors = []
    ref_date = yieldcurve.referenceDate()
    for day in range(0, days):
        yrs = day / 12.0
        d = calendar.advance(ref_date, ql.Period(day, ql.Days))
        compounding = ql.Compounded
        freq = ql.Semiannual

        zero_rate = yieldcurve.zeroRate(yrs, compounding, freq)
        eq_rate = zero_rate.equivalentRate(
                day_count, compounding, freq, ref_date, d
            ).rate()
        tenors.append(yrs)
        spots.append(100 * eq_rate)

    return pd.DataFrame(list(zip(tenors, spots)),
                        columns=["Maturities", "Curve"],
                        index=[''] * len(tenors))[1:]