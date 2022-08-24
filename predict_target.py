#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIM 500 Project #1
Critical Stress Testing for Time Series Forecasting in Python

The script will perform in-sample prediction and out-of-sample prediction
for SPY and TNX using two models, OLS and LSTM, and save the statistical
results of in-sample prediction.

@author: Raiden Han
"""

from utils.variable_selection import unify_index
from utils.regression_model import *


def main():
    # Read the local datasets
    try:
        df = pd.read_csv('data/dataset.csv', index_col=0, parse_dates=True)
        spy_features = pd.read_csv('data/spy_features.csv',
                                   index_col=0, parse_dates=True)
        tnx_features = pd.read_csv('data/tnx_features.csv',
                                   index_col=0, parse_dates=True)
        spy_features_test = pd.read_csv('data/spy_feature_prediction.csv',
                                        index_col=0, parse_dates=True)
        tnx_features_test = pd.read_csv('data/tnx_feature_prediction.csv',
                                        index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Prediction does not exist! "
              "Please run the predict_features script.")
        return
    # Deal with missing values
    spy = df.loc[:, 'SPY']
    tnx = df.loc[:, 'TNX']
    spy, spy_features = unify_index(spy, spy_features)
    tnx, tnx_features = unify_index(tnx, tnx_features)
    # Split the dataset into train and validation sets
    train_ratio = 0.9
    spy_features_val_train, spy_features_val_test = train_test_split(
        spy_features, train_ratio)
    tnx_features_val_train, tnx_features_val_test = train_test_split(
        tnx_features, train_ratio)
    spy_val_train, spy_val_test = train_test_split(spy, train_ratio)
    tnx_val_train, tnx_val_test = train_test_split(tnx, train_ratio)
    # Predict on the validation set
    spy_val_pred = reg_predict(
        spy_features_val_train, spy_val_train, spy_features_val_test)
    tnx_val_pred = reg_predict(
        tnx_features_val_train, tnx_val_train, tnx_features_val_test)
    # Evaluate the model performance
    performances = {
        'spy': model_evaluation(spy_val_test, spy_val_pred, ['ols', 'lstm']),
        'tnx': model_evaluation(tnx_val_test, tnx_val_pred, ['ols', 'lstm'])}
    # Out-of-sample prediction
    spy_pred = reg_predict(spy_features, spy, spy_features_test)
    tnx_pred = reg_predict(tnx_features, tnx, tnx_features_test)
    # Save the predictions and model performance statistics
    spy_val_pred.to_csv('data/spy_val_forecast.csv')
    tnx_val_pred.to_csv('data/tnx_val_forecast.csv')
    spy_pred.to_csv('data/spy_forecast.csv')
    tnx_pred.to_csv('data/tnx_forecast.csv')
    with open(f'stats/val_prediction_stats.json', 'w', encoding='utf-8') as f:
        json.dump(performances, f, ensure_ascii=False, indent=4, default=str)


if __name__ == '__main__':
    main()
