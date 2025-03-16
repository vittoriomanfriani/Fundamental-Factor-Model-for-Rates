import QuantLib as ql
import pandas as pd
from tqdm import tqdm
from Utils.conversions import *
from Utils.data_processing import *
from Utils.get_spot_rates import *
from Models.Nelson_Siegel import *
from Models.Nelson_Siegel_Svensonn import *


class SpotRatesCalculator:
    """
    A class for yield curve bootstrapping and fitting models such as Nelson-Siegel and Nelson-Siegel-Svensson.

    This class provides tools to calculate and fit yield curves from bond market data. It includes methods
    to bootstrap yield curves using QuantLib and to fit yield curve models such as Nelson-Siegel and
    Nelson-Siegel-Svensson for given datasets.

    Methods:
        curve_bootstrapper:
            Bootstraps a yield curve using bond data and calculates spot rates at a specified frequency.

        apply_bootstrapper:
            Applies yield curve bootstrapping to a dataset of bond data and returns spot rates for each date.

        apply_nelson_siegel:
            Fits the Nelson-Siegel model to a dataset to estimate yield curve parameters for each date.

        apply_nelson_siegel_svensonn:
            Fits the Nelson-Siegel-Svensson model to a dataset to estimate extended yield curve parameters.

        interpolate_nelson_siegel:
            Use Nelson-Siegel parametric model as interpolation method.

        interpolate_nelson_siegel_svensonn:
            Use Nelson-Siegel-Svensonn parametric model as interpolation method.

    Attributes:
        calendar (QuantLib.Calendar):
            A QuantLib calendar object used for date adjustments, defaulting to U.S. Government Bond conventions.

    """
    def __init__(self):
        self.calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)

    def curve_bootstrapper(self, curve_set_df, current_date, freq = 'monthly', rolldown = False):
        """
        Bootstrap the spot curve using bond data and interpolate spot rates.

        Parameters:
            curve_set_df (pd.DataFrame):
                A DataFrame containing bond data, including "maturity_date", "price", and "coupon".
            current_date (datetime.datetime):
                The reference date for curve bootstrapping.
            freq (str, optional):
                Frequency for the output spot rates. Options are:
                    - 'monthly': Spot rates are calculated monthly.
                    - 'tenors': Spot rates are calculated at the specific bond tenors.
                    - 'daily': Spot rates are calculated daily.
                Default is 'monthly'.
            rolldown (bool, optional):
                use only to compute rolldown

        Returns:
            pd.DataFrame:
                A DataFrame containing spot rates for the chosen frequency, indexed by maturity.

        Notes:
            - Bonds are filtered to avoid duplicate maturities.
            - The QuantLib `PiecewiseCubicZero` method is used to construct the yield curve.
        """
        current_date = self.calendar.adjust(pydatetime_to_quantlib_date(current_date))
        ql.Settings.instance().evaluationDate = current_date

        t_plus = 1
        bond_settlement_date = self.calendar.advance(current_date, ql.Period(t_plus, ql.Days))
        frequency = ql.Semiannual
        day_count = ql.ActualActual(ql.ActualActual.ISDA)
        par = 100.0

        bond_helpers = []
        seen_maturities = set()

        for _, row in curve_set_df.iterrows():
            maturity = pydatetime_to_quantlib_date(row["maturity_date"])
            if maturity in seen_maturities:
                continue
            seen_maturities.add(maturity)

            schedule = ql.Schedule(
                bond_settlement_date,
                maturity,
                ql.Period(frequency),
                self.calendar,
                ql.ModifiedFollowing,
                ql.ModifiedFollowing,
                ql.DateGeneration.Backward,
                False,
            )
            helper = ql.FixedRateBondHelper(
                ql.QuoteHandle(ql.SimpleQuote(row["price"])),
                t_plus,
                par,
                schedule,
                [row["coupon"] / 100],
                day_count,
                ql.ModifiedFollowing,
                par,
            )
            bond_helpers.append(helper)

        yc = ql.PiecewiseLogLinearDiscount(current_date, bond_helpers, day_count)

        if rolldown == True:
            return yc

        if freq == 'monthly':
            splcd = get_monthly_spot_rates(yc, day_count)
        elif freq == 'tenors':
            splcd = get_spot_rates_on_tenors(yc, day_count)
        elif freq == 'daily':
            splcd = get_daily_spot_rates

        return splcd

    def apply_bootstrapper(self, data, freq = 'monthly'):
        """
        Apply yield curve bootstrapping to a dataset of bond data.

        Parameters:
            data (pd.DataFrame):
                A DataFrame containing bond market data.
                The data should include a "Date" index, with each date containing bond information.
            freq (str, optional):
                Frequency for the output spot rates. Options are:
                    - 'daily': Spot rates are calculated daily.
                    - 'monthly': Spot rates are calculated monthly.
                    - 'tenors': Spot rates are calculated at the specific bond tenors.
                Default is 'monthly'.

        Returns:
            pd.DataFrame:
                A DataFrame containing bootstrapped spot rates for each date in the dataset.

        Notes:
            - The input data is first processed using `process_data()` to ensure consistency.
            - For each date, the most liquid bond in each time-to-maturity interval is selected.
            - The `curve_bootstrapper` function is then applied to calculate spot rates.
        """
        results_list = []

        # process data
        data = process_data(data)

        for date in tqdm(data.index.get_level_values(0).unique()):
            curve_set_df = data.loc[date]
            most_liquid = get_most_liquid_bond_by_interval(curve_set_df)
            zero_rate_curve = self.curve_bootstrapper(most_liquid, date, freq)
            zero_rate_curve["Date"] = date
            results_list.append(zero_rate_curve[1:])

        final_df = pd.concat(results_list)
        final_df.set_index(["Date"], inplace=True)

        return final_df

    def apply_nelson_siegel(self, curve_df, ridge=False, alpha=0.1):
        """
        Apply the Nelson-Siegel model to a dataset to fit yield curves.

        Parameters:
            curve_df (pd.DataFrame):
                A DataFrame containing bond yield curve data. It must include:
                    - "Maturities": Maturities of the bonds.
                    - "Curve": Corresponding yield values.
                    The index should include unique dates.
            ridge (bool, optional):
                Whether to use ridge regularization during optimization. Default is False.
            alpha (float, optional):
                Regularization parameter for ridge regression. Only used if ridge=True.
                Default is 0.1.

        Returns:
            pd.DataFrame:
                A DataFrame containing the fitted Nelson-Siegel parameters for each date:
                    - Beta0 (Level): Long-term level of the yield curve.
                    - Beta1 (Slope): Short-term slope of the yield curve.
                    - Beta2 (Curvature): Medium-term curvature of the yield curve.
                    - Lambda: Decay factor controlling the curvature.

        Notes:
            - The Nelson-Siegel model is a parsimonious yield curve model with three parameters
              (Beta0, Beta1, Beta2) and a decay factor (Lambda).
            - For each date in the dataset, the function fits the Nelson-Siegel model to the input curve data.
            - If ridge=True, ridge regularization is applied to stabilize the optimization.

        Example:
            fitted_params = apply_nelson_siegel(curve_df, ridge=True, alpha=0.1)
            print(fitted_params.head())
        """
        fitted_results = []

        for date in tqdm(curve_df.index.get_level_values(0).unique()):
            spot_curve = curve_df.loc[date]
            maturities = spot_curve["Maturities"].values
            yields = spot_curve["Curve"].values

            params = fit_nelson_siegel(
                maturities=maturities, yields=yields, ridge=ridge, alpha=alpha
            )

            fitted_results.append({
                "Date": date,
                "Beta0 (Level)": params[0],
                "Beta1 (Slope)": params[1],
                "Beta2 (Curvature)": params[2],
                "Lambda": params[3],
            })

        fitted_results_df = pd.DataFrame(fitted_results)
        fitted_results_df.set_index("Date", inplace=True)

        return fitted_results_df

    def apply_nelson_siegel_svensonn(self, curve_df, ridge = False, alpha=0.1):
        """
        Apply the Nelson-Siegel-Svensson model to a dataset to fit yield curves.

        Parameters:
            curve_df (pd.DataFrame):
                A DataFrame containing bond yield curve data. It must include:
                    - "Maturities": Maturities of the bonds.
                    - "Curve": Corresponding yield values.
                    The index should include unique dates.
            ridge (bool, optional):
                Whether to use ridge regularization during optimization. Default is False.
            alpha (float, optional):
                Regularization parameter for ridge regression. Only used if ridge=True.
                Default is 0.1.

        Returns:
            pd.DataFrame:
                A DataFrame containing the fitted Nelson-Siegel-Svensson parameters for each date:
                    - Beta0 (Level): Long-term level of the yield curve.
                    - Beta1 (Slope): Short-term slope of the yield curve.
                    - Beta2 (Curvature): Medium-term curvature of the yield curve.
                    - Beta3 (Second Curvature): Additional curvature to fit longer-term bonds.
                    - Lambda1: Decay factor controlling the curvature.
                    - Lambda2: Second decay factor for additional flexibility.

        Notes:
            - The Nelson-Siegel-Svensson model is an extension of the Nelson-Siegel model with an
              additional curvature term (Beta3) and decay factor (Lambda2). It provides more flexibility
              to fit complex yield curve shapes.
            - For each date in the dataset, the function fits the Nelson-Siegel-Svensson model
              to the input curve data.
            - If ridge=True, ridge regularization is applied to stabilize the optimization.

        Example:
            fitted_params = apply_nelson_siegel_svensonn(curve_df, ridge=True, alpha=0.1)
            print(fitted_params.head())
        """
        fitted_results = []

        for date in tqdm(curve_df.index.get_level_values(0).unique()):
            spot_curve = curve_df.loc[date]
            maturities = spot_curve["Maturities"].values
            yields = spot_curve["Curve"].values

            params = fit_nelson_siegel_svensson(
                maturities=maturities, yields=yields, ridge=ridge, alpha=alpha
            )

            fitted_results.append({
                "Date": date,
                "Beta0 (Level)": params[0],
                "Beta1 (Slope)": params[1],
                "Beta2 (Curvature)": params[2],
                "Beta3 (Second Curvature)": params[3],
                "Lambda1": params[4],
                "Lambda2": params[5],
            })

        fitted_results_df = pd.DataFrame(fitted_results)
        fitted_results_df.set_index("Date", inplace=True)

        return fitted_results_df

    def interpolate_nelson_siegel(self, loadings_df, freq = 'monthly'):
        """
        Interpolate yield curves using the Nelson-Siegel (NS) model for a given set of parameters over a specified frequency.

        This function iterates over all dates in the provided loadings DataFrame, computes the interpolated yield curve for a range of maturities
        using the NS formula, and returns the results in a structured DataFrame. The interpolation frequency (e.g., monthly, daily, or quarterly)
        determines the granularity of the interpolated curves.

        Parameters:
            loadings_df (pd.DataFrame):
                A DataFrame containing NS model parameters (`β₀`, `β₁`, `β₂`, `τ`), indexed by date.
                Each row corresponds to a unique date with its associated NS parameters.
            freq (str, optional):
                The frequency for interpolation. Options include:
                    - 'monthly': Interpolates 361 maturities over 30 years (~monthly intervals).
                    - 'daily': Interpolates 361*30 maturities over 30 years (~daily intervals).
                    - 'quarterly': Interpolates 121 maturities over 30 years (~quarterly intervals).
                Default is 'monthly'.

        Returns:
            pd.DataFrame:
                A DataFrame indexed by date with the following columns:
                    - "Curve": A list of interpolated yields for each maturity.
                    - "Maturities": A list of maturities corresponding to the interpolated yields.
        """
        fitted_results = []
        for date in tqdm(loadings_df.index.get_level_values(0).unique()):
            current_loadings = loadings_df.loc[date]
            if freq == 'monthly':
                maturities = np.linspace(0, 30, 361)
                interpolated_curve = [nelson_siegel(current_loadings, t) for t in maturities]
            elif freq == 'daily':
                maturities = np.linspace(0, 30, 361 * 30)
                interpolated_curve = [nelson_siegel(current_loadings, t) for t in maturities]
            elif freq == 'quarterly':
                maturities = np.linspace(0, 30, 121)
                interpolated_curve = [nelson_siegel(current_loadings, t) for t in maturities]

            fitted_results.append(pd.DataFrame({
                "Date": date,
                "Curve": interpolated_curve,
                "Maturities": maturities,
            }))

        fitted_results_df = pd.concat(fitted_results)
        fitted_results_df.set_index("Date", inplace=True)

        return fitted_results_df


    def interpolate_nelson_siegel_svensson(self, loadings_df, freq = 'monthly'):
        """
        Interpolate yield curves using the Nelson-Siegel-Svensson (NSS) model for a given set of parameters over a specified frequency.

        This function iterates over all dates in the provided loadings DataFrame, computes the interpolated yield curve for a range of maturities
        using the NSS formula, and returns the results in a structured DataFrame. The interpolation frequency (e.g., monthly, daily, or quarterly)
        determines the granularity of the interpolated curves.

        Parameters:
            loadings_df (pd.DataFrame):
                A DataFrame containing NSS model parameters (`β₀`, `β₁`, `β₂`, `β₃`, `τ₁`, `τ₂`), indexed by date.
                Each row corresponds to a unique date with its associated NSS parameters.
            freq (str, optional):
                The frequency for interpolation. Options include:
                    - 'monthly': Interpolates 361 maturities over 30 years (~monthly intervals).
                    - 'daily': Interpolates 361*30 maturities over 30 years (~daily intervals).
                    - 'quarterly': Interpolates 121 maturities over 30 years (~quarterly intervals).
                Default is 'monthly'.

        Returns:
            pd.DataFrame:
                A DataFrame indexed by date with the following columns:
                    - "Curve": A list of interpolated yields for each maturity.
                    - "Maturities": A list of maturities corresponding to the interpolated yields.
        """
        fitted_results = []
        for date in tqdm(loadings_df.index.get_level_values(0).unique()):
            current_loadings = loadings_df.loc[date]
            if freq == 'monthly':
                maturities = np.linspace(0, 30, 361)
                interpolated_curve = [nelson_siegel_svensson(current_loadings, t) for t in maturities]
            elif freq == 'daily':
                maturities = np.linspace(0, 30, 361*30)
                interpolated_curve = [nelson_siegel_svensson(current_loadings, t) for t in maturities]
            elif freq == 'quarterly':
                maturities = np.linspace(0, 30, 121)
                interpolated_curve = [nelson_siegel_svensson(current_loadings, t) for t in maturities]

            fitted_results.append(pd.DataFrame({
                "Date": date,
                "Curve" : interpolated_curve,
                "Maturities" : maturities,
            }))

        fitted_results_df = pd.concat(fitted_results)
        fitted_results_df.set_index("Date", inplace=True)

        return fitted_results_df



