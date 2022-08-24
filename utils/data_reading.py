# %%
import os
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime


# %%
def load_stock_benchmark(s_day, e_day):
    """ Load S&P 500 ETF data from Yahoo! finance

    Parameters
    ----------
    s_day : str
        The start date of data sample
    e_day : str
        The end date of data sample

    Returns
    -------
    stock : DataFrame
        S&P 500 Data from Yahoo! finance
    """

    stock = pdr.get_data_yahoo('SPY', s_day, e_day, interval='mo')['Adj Close']
    stock = stock.iloc[:-1]
    stock.name = 'SPY'

    return stock


def load_bond_benchmark(s_day, e_day):
    """ Load 10-Year Treasury Maturity Rate data from Yahoo! Finance

    Parameters
    ----------
    s_day : str
        The start date of data sample
    e_day : str
        The end date of data sample

    Returns
    -------
    stock : DataFrame
        10-Year Treasury Maturity Rate data from Yahoo! Finance
    """

    bond = pdr.get_data_yahoo('^TNX', s_day, e_day, interval='mo')['Adj Close']
    bond = bond.iloc[:-1]
    bond.name = 'TNX'

    return bond


# %%
def load_seasonal_indicators(inds, s_day, e_day):
    """ Loading seasonal data from FRED and using linear interpolation to
    transform it into monthly data

    Parameters
    ----------
    inds : list
        Parameters for all seasonal data
    s_day : str
        The start date of data sample
    e_day : str
        The end date of data sample

    Returns
    -------
    seasonal_inds : DataFrame
        Transformed seasonal indicators
    """

    data_list = list()
    for ind in inds:
        data = pdr.get_data_fred(ind, s_day, e_day)
        data = data.resample('MS').interpolate(method='linear')
        data_list.append(data)
    seasonal_inds = pd.concat(data_list, axis=1)

    return seasonal_inds


def load_monthly_indicators(inds, s_day, e_day):
    """ Loading monthly data from FRED

    Parameters
    ----------
    inds : list
        Parameters for all monthly data
    s_day : str
        The start date of data sample
    e_day : str
        The end date of data sample

    Returns
    -------
    monthly_inds : DataFrame
        Monthly indicators
    """

    data_list = list()
    for ind in inds:
        data = pdr.get_data_fred(ind, s_day, e_day)
        data_list.append(data)
    monthly_inds = pd.concat(data_list, axis=1)

    return monthly_inds


def load_daily_weekly_indicators(inds, s_day, e_day):
    """ Loading weekly and daily data from FRED and maintain the first item of
    each month to transform it into monthly data

    Parameters
    ----------
    inds : list
        Parameters for all weekly and daily data
    s_day : str
        The start date of data sample
    e_day : str
        The end date of data sample

    Returns
    -------
    daily_weekly_inds : DataFrame
        Transformed weekly and daily indicators
    """

    data_list = list()
    for ind in inds:
        data = pdr.get_data_fred(ind, s_day, e_day)
        data = data.resample('MS').first()
        data_list.append(data)
    daily_weekly_inds = pd.concat(data_list, axis=1)

    return daily_weekly_inds


# %%
def get_online_data(inds_dict, s_day='1985-01-01',
                    e_day=datetime.today().strftime('%Y-%m-%d'), save=False):
    """

    Parameters
    ----------
    inds_dict : dict
        Dictionary of indicators, the keys are frequencies and the values are
        required indicators
    s_day : str, optional
        The start date of data sample
    e_day : str, optional
        The end date of data sample
    save : bool, optional
        If Ture, the data will be saved to backup.all_data.csv

    Returns
    -------
    all_data : DataFrame
        All macroeconomics indicators data and two kinds of benchmark data
    """

    inds_list = []
    stock_benchmark = load_stock_benchmark(s_day, e_day)
    bond_benchmark = load_bond_benchmark(s_day, e_day)
    if 'seasonal' in inds_dict.keys():
        seasonal_inds = load_seasonal_indicators(
            inds_dict['seasonal'], s_day, e_day)
        inds_list.append(seasonal_inds)
    if 'monthly' in inds_dict.keys():
        monthly_inds = load_monthly_indicators(
            inds_dict['monthly'], s_day, e_day)
        inds_list.append(monthly_inds)
    if 'daily/weekly' in inds_dict.keys():
        weekly_inds = load_daily_weekly_indicators(
            inds_dict['daily/weekly'], s_day, e_day)
        inds_list.append(weekly_inds)
    all_data = pd.concat([stock_benchmark, bond_benchmark] + inds_list, axis=1)
    if save:
        os.makedirs('data', exist_ok=True)
        all_data.to_csv('data/dataset.csv')

    return all_data
