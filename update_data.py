#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIM 500 Project #1
Critical Stress Testing for Time Series Forecasting in Python

The script will pull S&P 500 ETF, Treasury Yield 10-years and macroeconomic
indicators data from Yahoo! Finance and FRED database from 1985 to present and
store them locally.

@author: Raiden Han
"""

from utils.data_reading import *


def main():
    inds = {'seasonal': ['GDP', 'GDPC1'],
            'monthly': ['CPIAUCSL', 'UNRATE', 'EMVOVERALLEMV', 'APU00007471A',
                        'PAYEMS', 'MCOILWTICO', 'DSPIC96', 'PCE', 'GS3M',
                        'GS5', 'CSUSHPINSA', 'PPIACO', 'FEDFUNDS', 'IQ12260'],
            'daily/weekly': ['WM2NS', 'WALCL', 'MORTGAGE30US',
                             'BAMLC0A4CBBBEY', 'T10YIE', 'T10Y2Y',
                             'BAMLH0A0HYM2', 'DTWEXBGS', 'VIXCLS']}
    get_online_data(inds, save=True)


if __name__ == '__main__':
    main()
