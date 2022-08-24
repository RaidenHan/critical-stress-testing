#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIM 500 Project #1
Critical Stress Testing for Time Series Forecasting in Python

The script needs to be run at the end. It will read the local dataset and plot
multiple time series images.

@author: Raiden Han
"""

import json
import pandas as pd
from utils.variable_selection import unify_index
from utils.visualization import *


def main():
    # Read the local datasets
    try:
        df = pd.read_csv('data/dataset.csv', index_col=0, parse_dates=True)
        spy_features = pd.read_csv('data/spy_features.csv',
                                   index_col=0, parse_dates=True)
        tnx_features = pd.read_csv('data/tnx_features.csv',
                                   index_col=0, parse_dates=True)
        hw_features_val = pd.read_csv('data/val_hw_prediction.csv',
                                      index_col=0, parse_dates=True)
        arima_features_val = pd.read_csv('data/val_arima_prediction.csv',
                                         index_col=0, parse_dates=True)
        spy_val = pd.read_csv('data/spy_val_forecast.csv',
                              index_col=0, parse_dates=True)
        tnx_val = pd.read_csv('data/tnx_val_forecast.csv',
                              index_col=0, parse_dates=True)
        spy_features_test = pd.read_csv('data/spy_feature_prediction.csv',
                                        index_col=0, parse_dates=True)
        tnx_features_test = pd.read_csv('data/tnx_feature_prediction.csv',
                                        index_col=0, parse_dates=True)
        spy_test = pd.read_csv('data/spy_forecast.csv',
                               index_col=0, parse_dates=True)
        tnx_test = pd.read_csv('data/tnx_forecast.csv',
                               index_col=0, parse_dates=True)
        with open('stats/features_stats.json', 'r') as f:
            feature_stats = json.load(f)
            feature_methods = {feature: feature_stats[feature]['best_model']
                               for feature in feature_stats}
    except FileNotFoundError:
        print("Prediction does not exist! "
              "Please run the predict_target script.")
        return
    # Extract original target variables
    spy = df.loc[:, 'SPY']
    tnx = df.loc[:, 'TNX']
    # Combine DataFrame's and discard duplicate features
    all_features = pd.concat(
        [spy_features, tnx_features], axis=1).dropna(axis=0)
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]
    all_features_pred = pd.concat(
        [spy_features_test, tnx_features_test], axis=1).dropna(axis=0)
    all_features_pred = all_features_pred.loc[
                        :, ~all_features_pred.columns.duplicated()]
    # Unify the indices of features and targets
    spy, all_features = unify_index(spy, all_features)
    tnx, all_features = unify_index(tnx, all_features)
    # Plotting
    plot_one_to_one_trends(spy_features, spy, save=True)
    plot_one_to_one_trends(tnx_features, tnx, save=True)
    plot_val_feature_preds(all_features, hw_features_val, arima_features_val,
                           feature_methods, save=True)
    plot_val_target_preds(spy, spy_val, save=True)
    plot_val_target_preds(tnx, tnx_val, save=True)
    plot_feature_predictions(all_features, all_features_pred, save=True)
    plot_target_predictions(spy, spy_test, save=True)
    plot_target_predictions(tnx, tnx_test, save=True)


if __name__ == '__main__':
    main()
