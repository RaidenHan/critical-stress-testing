#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIM 500 Project #1
Critical Stress Testing for Time Series Forecasting in Python

The script will use the better time series model for each feature from the
previous step to predict features on the test set. Optionally, the script will
ask for shocks in line with the Dodd-Frank Act Stress Test scenarios.

@author: Raiden Han
"""

from utils.marco_ind_prediction import *


def main():
    # Read the local datasets
    try:
        spy_features = pd.read_csv('data/spy_features.csv',
                                   index_col=0, parse_dates=True)
        tnx_features = pd.read_csv('data/tnx_features.csv',
                                   index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Dataset does not exist! Please run the select_features script.")
        return
    # Discard duplicate features
    all_features = pd.concat(
        [spy_features, tnx_features], axis=1).dropna(axis=0)
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]
    # Get the better time series model for each feature
    with open('stats/features_stats.json', 'r') as f:
        feature_stats = json.load(f)
    feature_methods = {feature: feature_stats[feature]['best_model']
                       for feature in feature_stats}
    # Input forecast length
    period = int(input("Please enter the length of the predicted period "
                       "(integer greater than zero): "))
    # Input shock
    feature_shocks = {}
    shock_flag = input("Do you want to specify values of an economic variable "
                       "in the forecast? (y/n): ")
    while shock_flag == 'y':
        for i in range(all_features.shape[1]):
            print(f"{i + 1}: {all_features.columns[i]}", end='\t')
        feature_ind = int(input("\nSelect the feature number: ")) - 1
        shock_period = input("Enter the number of periods in the forecast "
                             "(separated by commas):")
        shock = input("Enter the value you want to specify (same length as "
                      "entered above, separated by commas):")
        shock_period = [int(x) for x in shock_period.split(',')]
        shock = [float(x) for x in shock.split(',')]
        feature_shocks[
            all_features.columns[feature_ind]] = [shock_period, shock]
        shock_flag = input("Do you want to specify values of another economic "
                           "variable in the forecast? (y/n): ")
    # Predict features
    s_day = all_features.index[0]
    e_day = all_features.index[-1]
    feature_prediction = forecast_macro_inds(
        all_features, feature_methods, s_day, e_day, period, feature_shocks)
    # Save the forecast
    spy_feature_prediction = feature_prediction.loc[:, spy_features.columns]
    tnx_feature_prediction = feature_prediction.loc[:, tnx_features.columns]
    spy_feature_prediction.to_csv('data/spy_feature_prediction.csv')
    tnx_feature_prediction.to_csv('data/tnx_feature_prediction.csv')


if __name__ == '__main__':
    main()
