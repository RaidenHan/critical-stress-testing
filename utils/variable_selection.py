# %%
import os
from itertools import combinations
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr, kendalltau
from statsmodels.stats.outliers_influence import variance_inflation_factor


# %%
def unify_index(df1, df2):
    """Unify two DataFrames indexed by timestamp and maintain timepoints
    existing in both"""

    # Process DataFrame and Series separately
    if len(df1.shape) == 1:
        df1_col = df1.name
    else:
        df1_col = df1.columns
    if len(df2.shape) == 1:
        df2_col = df2.name
    else:
        df2_col = df2.columns
    df = pd.concat([df1, df2], axis=1).dropna()
    df1_unified = df.loc[:, df1_col]
    df2_unified = df.loc[:, df2_col]

    return df1_unified, df2_unified


def linear_regression_stats(data, targets, save=False):
    """ Run linear regression between independent variables and certain targets

    Parameters
    ----------
    data : DataFrame
        The data containing all variables
    targets : str or array_like
        Targeted variables. The function will only calculate the regression
        between targets and other variables.
    save : Bool
        If True, all output tables will be saved to backup folder

    Returns
    -------
    linear_tables : dict
        The targeted variables and corresponding linear summaries with other
        variables

    """

    linear_tables = dict()
    # record the four indicators from the linear regression
    params = ['alpha', 'beta', 'p-value', 'adjusted_r-squared', 'pearson_r',
              'spearman_r', 'kendal_tau']
    # Ensure the iteration works
    if type(targets) == str:
        targets = list(targets)
    for target in targets:
        y = data[target]
        # Only run regressions between non-target variables and targeted one
        ind_var = data.drop(targets, axis=1).columns
        target_dict = dict()
        for var in ind_var:
            x = data.loc[:, var]
            # Deal with missing values
            x, y_tmp = unify_index(x, y)
            # Fit a linear regression model
            X = sm.add_constant(x)
            result = sm.OLS(y_tmp, X).fit()
            target_dict[var] = [
                result.params[0], result.params[1], result.pvalues[1],
                result.rsquared_adj, pearsonr(x, y_tmp)[0],
                spearmanr(x, y_tmp)[0], kendalltau(x, y_tmp)[0]]
        # Put all data into a DataFrame
        target_frame = pd.DataFrame(target_dict).T
        target_frame.columns = params
        linear_tables[target] = target_frame
        if save:
            os.makedirs('stats', exist_ok=True)
            target_frame.to_csv(f'stats/{target}_reg_stats.csv')

    return linear_tables


# %%
def variable_selection(data, order, limits=4, threshold=10):
    """ Select certain variables in given order for analysis of target variable
    based on VIF

    Parameters
    ----------
    data : DataFrame
        The data containing all variables
    order : array_like
        Independent variables in certain order. The function will add variable
        one by one to select and make sure there is no multicollinearity
    limits : int
        The number of to be selected independent variables, greater than 1
    threshold : float
        The threshold for VIF. If after adding a variable, any variable's VIF
        is larger than the threshold, the variable will be dropped.

    Returns
    -------
    selected_variables : list
        The list containing all selected variables
    """

    for n in range(limits, len(order)):
        for com in combinations(order[:n], limits):
            X = data.loc[:, com].dropna()
            vif = [variance_inflation_factor(X.values, i)
                   for i in range(X.shape[1])]
            if max(vif) < threshold:
                return list(com)

    print('Cannot find enough variables without multicollinearity!')
    return list()
