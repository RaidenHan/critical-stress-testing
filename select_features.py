#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIM 500 Project #1
Critical Stress Testing for Time Series Forecasting in Python

The script will read the local dataset, calculate the linear regression
statistics for each macroeconomic variable and the target variable, and select
four predictors without multicollinearity for each target variable based on
adjusted R-squared. Then, it will compare the Holt Winters exponential
smoothing method with the ARIMA model and choose the method with the smaller
MSE on the validation set for prediction.

@author: Raiden Han
"""

from utils.variable_selection import *
from utils.marco_ind_prediction import *
from utils.visualization import plot_heatmap


def main():
    try:
        df = pd.read_csv('data/dataset.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Dataset does not exist! Please run the update_data script.")
        return
    # Calculate the linear regression statistics
    stats_dict = linear_regression_stats(df, ['SPY', 'TNX'], save=True)
    # Select the features based on adjusted R-squared
    spy_order = stats_dict['SPY'].sort_values(
        'adjusted_r-squared', ascending=False).index
    tnx_order = stats_dict['TNX'].sort_values(
        'adjusted_r-squared', ascending=False).index
    spy_feature_name = variable_selection(df, spy_order)
    tnx_feature_name = variable_selection(df, tnx_order)
    all_feature_name = list(set(spy_feature_name + tnx_feature_name))
    # Save the features
    spy_features = df.loc[:, spy_feature_name]
    tnx_features = df.loc[:, tnx_feature_name]
    spy_features.to_csv('data/spy_features.csv')
    tnx_features.to_csv('data/tnx_features.csv')
    # Plot the heatmaps
    plot_heatmap(spy_features, 'spy features', save=True)
    plot_heatmap(tnx_features, 'tnx features', save=True)
    # Compare the time series models
    all_features = df.loc[:, all_feature_name]
    time_series_val(all_features)


if __name__ == '__main__':
    main()
