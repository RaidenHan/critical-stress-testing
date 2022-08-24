# %%
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    median_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping


# %%
def train_test_split(dataset, train_ratio):
    """Divide the samples into training test sets in order"""
    break_point = dataset.index[round(dataset.shape[0] * train_ratio) - 1]
    train_set = dataset[:break_point]
    test_set = dataset[break_point:].iloc[1:]

    return train_set, test_set


def linear_reg_predict(X_train, y_train, X_test):
    """Conduct the linear regression stress testing

    Parameters
    ----------
    X_train : DataFrame
        The existing marco indicator data
    y_train : Series
        The existing benchmark data
    X_test : DataFrame
        The forecast marco indicator data

    Returns
    -------
    y_pred : Series
        The forecast benchmark data

    """

    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train).fit()
    y_pred = model.predict(sm.add_constant(X_test, has_constant='add'))

    return y_pred


def create_dataset(X, y, time_steps=1):
    """Helper function for lstm_reg_predict to help create inputs and outputs
    containing multiple time points"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps + 1):
        v = X[i:i + time_steps, :]
        Xs.append(v)
        ys.append(y[i + time_steps - 1])

    return np.array(Xs), np.array(ys)


def lstm_reg_predict(X_train, y_train, X_test, time_window=12):
    """Conduct the LSTM regression stress testing

    Parameters
    ----------
    X_train : DataFrame
        The existing marco indicator data
    y_train : Series
        The existing benchmark data
    X_test : DataFrame
        The forecast marco indicator data
    time_window : int
        The number of time points contained in each input

    Returns
    -------
    y_pred : Series
        The forecast benchmark data

    """

    # Data Preprocessing
    time_index = X_test.index
    X_test = pd.concat([X_train.iloc[-time_window + 1:, :], X_test])
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    # Create Input and Output Samples
    X_train, y_train = create_dataset(X_train, y_train, time_steps=time_window)
    X_test, _ = create_dataset(X_test, X_test[:, 0], time_steps=time_window)
    # LSTM Model
    model = Sequential()
    model.add(LSTM(4, input_shape=(time_window, 4)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stopping = EarlyStopping(
        monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, callbacks=[early_stopping],
                        epochs=500, verbose=1, validation_split=0.2,
                        batch_size=16, shuffle=False)
    y_pred = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred)
    y_pred = pd.Series(y_pred.reshape(1, -1)[0], index=time_index)

    return y_pred


def reg_predict(X_train, y_train, X_test, time_window=12):
    """ Combine the results of linear regression and LSTM regression into
    one DataFrame output

    Parameters
    ----------
    X_train : DataFrame
        The existing marco indicator data
    y_train : Series
        The existing benchmark data
    X_test : DataFrame
        The forecast marco indicator data
    time_window : int
        The number of time points contained in each input

    Returns
    -------
    y_pred : DataFrame
        The forecast benchmark data

    """

    ols_y_pred = linear_reg_predict(X_train, y_train, X_test)
    lstm_y_pred = lstm_reg_predict(X_train, y_train, X_test, time_window)
    y_pred = pd.concat([ols_y_pred, lstm_y_pred], axis=1)
    y_pred.columns = ['OLS', 'LSTM']

    return y_pred


def model_evaluation(y_true, y_preds, labels):
    """ Evaluate the model performance in terms of mean squared error, mean
    absolute error, and median absolute error

    Parameters
    ----------
    y_true : Series
        Ground truth (correct) target values
    y_preds : DataFrame
        Estimated target values
    labels : list
        Labels for each prediction model

    Returns
    -------
    performance : dict
        Model's performance

    """

    performance = {'start_date': y_true.index[0], 'end_date': y_true.index[-1]}
    for i in range(y_preds.shape[1]):
        label = labels[i]
        y_pred = y_preds.iloc[:, i]
        performance[label] = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'meae': median_absolute_error(y_true, y_pred)
        }

    return performance
