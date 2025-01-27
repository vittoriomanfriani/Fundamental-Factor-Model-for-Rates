from tqdm import tqdm
import statsmodels.api as sm
import numpy as np
import pandas as pd

def cross_sectional_regression_nelson_siegel(df, loadings_df, nss = False):
    """
        Performs cross-sectional regression of the Nelson-Siegel model on bond excess returns
        to estimate factor loadings and their impact on returns for each date in the dataset.

        Parameters:
        ----------
        df : pandas.DataFrame
            A multi-index DataFrame where:
            - The first index level corresponds to dates.
            - The columns include:
                - 'time to maturity': Time to maturity of each bond (in years).
                - 'Excess Returns': Excess returns of the bonds to be regressed.
        loadings_df : pandas.DataFrame
            A DataFrame containing the Nelson-Siegel lambda parameter for each date,
            indexed by the same date values as `df`.
        nss : bool
            Set to True if you pass Nelson Siegel Svensson Parmas

        Returns:
        -------
        pandas.DataFrame
            A DataFrame summarizing the regression results for each date. Columns include:
            - 'date': The date of the regression.
            - 'const': The constant term from the regression.
            - 'beta1': The estimated loading on the first Nelson-Siegel factor.
            - 'beta2': The estimated loading on the second Nelson-Siegel factor.
            - 'beta3': The estimated loading on the third Nelson-Siegel-Svensson factor.
            - 'r_squared': The R-squared of the regression.

        Methodology:
        -----------
        1. For each date in the dataset:
            a. Filter bonds with available data (`dropna()`).
            b. Compute the Nelson-Siegel factor loadings (`f1` and `f2`) based on the
               time to maturity (`t`) and the lambda parameter from `loadings_df`.
            c. Perform an Ordinary Least Squares (OLS) regression:
               \[
               y_t = \beta_0 + \beta_1 f1_t + \beta_2 f2_t + \epsilon_t
               \]
               where:
               - \( y_t \): Excess returns.
               - \( f1_t, f2_t \): Nelson-Siegel factor loadings.
        2. Store the regression coefficients, R-squared, and date for each regression.

        Notes:
        ------
        - The lambda parameter (\( \lambda \)) controls the Nelson-Siegel factor loadings
          and must be provided for each date in `loadings_df`.
        - Bonds with missing values for excess returns or time to maturity are excluded
          from the regression for that date.
    """
    results = []
    for date in tqdm(df.index.get_level_values(0).unique()):
        ex = df.loc[date].dropna()
        params = loadings_df.loc[date]
        if len(ex) == 0:  # Skip if no data is available for this date
            continue

        if nss == True:
            lambda1 = params.loc[date]['Lambda1']
            lambda2 = params.loc[date, 'Lambda2']
        else:
            lambda1 = params.loc[date, 'Lambda']

        t = ex['time to maturity']

        # Compute Nelson-Siegel factor loadings
        ex['f1'] = (1 - np.exp(-t / lambda1)) / (t / lambda1)
        ex['f2'] = (1 - np.exp(-t / lambda1)) / (t / lambda1) - np.exp(-t / lambda1)
        if nss == True:
            ex['f3'] = (1 - np.exp(-t / lambda2)) / (t / lambda2) - np.exp(-t / lambda2)

        # Filter rows with non-finite values
        ex = ex[np.isfinite(ex[['Excess Returns', 'f1', 'f2']]).all(axis=1)]

        if len(ex) == 0:  # Ensure data is still available after filtering
            continue

        y = ex['Excess Returns']

        if nss == True:
            X = sm.add_constant(ex[['f1', 'f2', 'f3']])
        else:
            X = sm.add_constant(ex[['f1', 'f2']])

        if X.empty or y.empty:  # Ensure X and y are non-empty
            continue

        try:
            # Fit the regression model
            model = sm.OLS(y, X).fit()
            # Store results
            if nss == True:
                results.append({
                    'date': date,
                    'const': model.params.get('const', np.nan),
                    'beta1': model.params.get('f1', np.nan),
                    'beta2': model.params.get('f2', np.nan),
                    'beta3': model.params.get('f3', np.nan),
                    'r_squared': model.rsquared
                })
            else:
                results.append({
                    'date': date,
                    'const': model.params.get('const', np.nan),
                    'beta1': model.params.get('f1', np.nan),
                    'beta2': model.params.get('f2', np.nan),
                    'r_squared': model.rsquared
                })
        except Exception as e:
            # Handle any unexpected errors during regression
            print(f"Error processing date {date}: {e}")
            continue

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results)

    return results_df