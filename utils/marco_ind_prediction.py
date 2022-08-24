# %%
import json
import pandas as pd
import pmdarima as pm
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import ExponentialSmoothing


# %%
def hw_ind_predict(data, s_day, e_day, period):
    """Using the Holt Winter's method to predict the macro indicator

    Parameters
    ----------
    data : Series
        The macro indicator time series. The frequency is MS
    s_day : str
        The start date of the sample for prediction
    e_day : str
        The end date of the sample for prediction
    period : int
        The number of periods to be predicted

    Returns
    -------
    predicted_data : Series
        The predicted marco data

    """

    # Apply a MaxAbsScaler to avoid overflow
    train_set = data[s_day:e_day].to_frame()
    train_index = train_set.index
    scaler = MaxAbsScaler()
    train_set = pd.Series(scaler.fit_transform(train_set).reshape(-1, ),
                          index=train_index)
    model = ExponentialSmoothing(train_set, trend='mul',
                                 seasonal='mul', freq='MS',
                                 initialization_method='estimated').fit()
    predicted_data = model.forecast(period).to_frame()
    test_index = predicted_data.index
    predicted_data = pd.Series(scaler.inverse_transform(
        predicted_data).reshape(-1, ), index=test_index)

    return predicted_data


def arima_ind_predict(data, s_day, e_day, period):
    """Using the ARIMA utils to predict the macro indicator

    Parameters
    ----------
    data : Series
        The macro indicator time series
    s_day : str
        The start date of the sample for prediction
    e_day : str
        The end date of the sample for prediction
    period : int
        The number of periods to be predicted

    Returns
    -------
    predicted_data : Series
        The predicted marco data

    """

    train_set = data[s_day:e_day]
    model = pm.auto_arima(train_set, start_p=1, start_q=1)
    predicted_data = model.predict(period)
    ts_index = pd.date_range(start=e_day, periods=period + 1,
                             freq='MS', inclusive='right')
    predicted_data = pd.Series(predicted_data, index=ts_index)

    return predicted_data


def ts_accuracy_test(data, train_ratio):
    """ Compare the squared error of both models and return predictions

    Parameters
    ----------
    data : Series
        The macro indicator time series
    train_ratio : float
        The size of the train sample

    Returns
    -------
    stats : dict
        The statistics for the two time series models
    hw_prediction : Series
        Holt-Winter prediction
    arima_prediction : Series
        ARIMA prediction

    """

    # Split the data into training and test sets
    data = data.dropna()
    break_point = round(data.shape[0] * train_ratio) - 1
    s_day = data.index[0]
    e_day = data.index[break_point]
    period = data.shape[0] - break_point - 1
    test_set = data[e_day:].iloc[1:]
    # Predict on the validation set and compare the MSE
    hw_prediction = hw_ind_predict(data, s_day, e_day, period)
    arima_prediction = arima_ind_predict(data, s_day, e_day, period)
    hw_error = mean_squared_error(test_set, hw_prediction)
    arima_error = mean_squared_error(test_set, arima_prediction)
    # Save the statistics
    stats = {'indicator': data.name,
             'train_start': s_day, 'test_start': test_set.index[0],
             'hw_model': {'mse': hw_error,
                          'std_mse': hw_error / test_set.std()},
             'arima_model': {'mse': arima_error,
                             'std_mse': arima_error / test_set.std()}}
    if hw_error > arima_error:
        stats['best_model'] = 'arima'
    else:
        stats['best_model'] = 'hw'

    return stats, hw_prediction, arima_prediction


def time_series_val(features, train_ratio=0.9):
    """ Compare feature time series model results on validation sets, save
    predictions and statistics, and return the better model for each feature

    Parameters
    ----------
    features : DataFrame
        The complete features' dataset
    train_ratio : float
        Proportion of training set

    Returns
    -------
    feature_model : dict
        Features and their corresponding better models

    """

    feature_methods = {}
    feature_stats = {}
    val_hw_prediction_list = []
    val_arima_prediction_list = []
    # Calculate and compare for each feature
    for name, feature in features.iteritems():
        stats, hw_prediction, arima_prediction = ts_accuracy_test(
            feature, train_ratio)
        feature_methods[name] = stats['best_model']
        feature_stats[name] = stats
        val_hw_prediction_list.append(hw_prediction)
        val_arima_prediction_list.append(arima_prediction)
    # Combine one model's all predictions to a single DataFrame
    val_hw_prediction = pd.concat(val_hw_prediction_list, axis=1)
    val_hw_prediction.columns = features.columns
    val_arima_prediction = pd.concat(val_arima_prediction_list, axis=1)
    val_arima_prediction.columns = features.columns
    # Save all results
    with open(f'stats/features_stats.json', 'w', encoding='utf-8') as f:
        json.dump(feature_stats, f, ensure_ascii=False, indent=4, default=str)
    val_hw_prediction.to_csv('data/val_hw_prediction.csv')
    val_arima_prediction.to_csv('data/val_arima_prediction.csv')

    return feature_methods


def forecast_macro_ind(data, method, s_day, e_day, period,
                       shock_period=None, shock=None):
    """ Introduce a shock/shocks to a certain macro indicator and forecast

    Parameters
    ----------
    data : Series
        Time series data of the macro indicator
    method : str, ‘hw' or 'arima'
        The method to be used to predict the future indicators
    s_day : str
        The start date of the sample for prediction
    e_day : str
        The end date of the sample for prediction
    period : int
        The number of periods to be predicted
    shock_period : array_like
        The future period when shocks happen
    shock : array_like
        The future shocks, corresponding to the shock_period parameter

    Returns
    -------
    shocked_data : Series
        The predicted marco data with shocks
    """

    # Acquire the prediction function
    if method == 'hw':
        pred_func = hw_ind_predict
    elif method == 'arima':
        pred_func = arima_ind_predict
    else:
        raise ValueError("Invalid prediction method!")
    # Define the original training set
    train = data.loc[s_day:e_day]
    # Find the date series and gaps between two adjacent shocks
    if shock_period and shock:
        date_range = pd.date_range(start=e_day, periods=period + 1, freq='MS')
        shock_period_diff = [
            shock_period[i] - shock_period[i - 1] - 1 if i > 0 else
            shock_period[i] - 1 for i in range(len(shock_period))]
        last_period = period - shock_period[-1]
        # Forecast the macro indicator value on every gap
        for i in range(len(shock_period)):
            if shock_period_diff[i] > 0:
                # Forecast on one gap
                shocked_data = pred_func(
                    train, s_day, e_day, shock_period_diff[i])
                train = pd.concat([train, shocked_data])
            # Fill the shock
            train.loc[date_range[shock_period[i]]] = shock[i]
            e_day = date_range[shock_period[i]]
    else:
        last_period = period
    # Forecast on the last period
    if last_period > 0:
        shocked_data = pred_func(train, s_day, e_day, last_period)
        train = pd.concat([train, shocked_data])
    shocked_data = train.iloc[-period:]

    return shocked_data


def forecast_macro_inds(features, feature_methods, s_day, e_day, period,
                        feature_shocks=None):
    """ Forecast the macro predictors for a given period with optional shocks

    Parameters
    ----------
    features : DataFrame
        The original features' dataset
    feature_methods : dict
        Features and their corresponding better models
    s_day : str
        The start date of the sample for prediction
    e_day : str
        The end date of the sample for prediction
    period : int
        The number of periods to be predicted
    feature_shocks : dict
        Features and their corresponding shock periods and shocks

    Returns
    -------
    feature_prediction : DataFrame
        Forecasted features

    """

    if feature_shocks is None:
        feature_shocks = {}
    feature_prediction_list = []
    for name, feature in features.iteritems():
        method = feature_methods[name]
        if name in feature_shocks:
            shock_period, shock = feature_shocks[name]
        else:
            shock_period = shock = None
        feature_prediction_list.append(forecast_macro_ind(
            feature, method, s_day, e_day, period, shock_period, shock))
    feature_prediction = pd.concat(feature_prediction_list, axis=1)
    feature_prediction.columns = features.columns

    return feature_prediction


###############################################################################
# %%
def marco_ind_forecast(data, method, period=36):
    """Forecast the future macro indicators for a certain period

    Parameters
    ----------
    data : DataFrame
        The existing sample for data
    method : Dict
        The dictionary containing forecasting models as keys and indicator
        names as values
    period : int
        The number of periods to be predicted

    Returns
    -------
    predicted_data : DataFrame
        The predicted marco data

    """

    hw_ind_list = []
    arima_ind_list = []
    s_day = data.index[0]
    e_day = data.index[-1]
    for ind in method['HW']:
        hw_ind_list.append(hw_ind_predict(data[ind], s_day, e_day, period))
    for ind in method['ARIMA']:
        arima_ind_list.append(arima_ind_predict(data[ind],
                                                s_day, e_day, period))
    predicted_data = pd.concat(hw_ind_list, axis=1)
    arima_predicted_data = pd.DataFrame(arima_ind_list,
                                        columns=predicted_data.index).T
    predicted_data = pd.merge(predicted_data, arima_predicted_data,
                              left_index=True, right_index=True)
    predicted_data.columns = method['HW'] + method['ARIMA']

    return predicted_data


# %%
def intro_shock(data, method, s_day, e_day, period, shock_period, shock):
    """ Introduce a shock (series) to a certain macro indicator

    Parameters
    ----------
    data : Series
        Time series data of the macro indicator
    method : str, ‘HW' or 'ARIMA'
        The method to be used to predict the future indicators
    s_day : str
        The start date of the sample for prediction
    e_day : str
        The end date of the sample for prediction
    period : int
        The number of periods to be predicted
    shock_period : int or array_like
        The future period when shocks happen
    shock : float or array_like
        The future shocks, corresponding to the shock_period parameter

    Returns
    -------
    shocked_data : Series
        The predicted marco data with shocks
    """

    # Convert a single shock into a list
    if type(shock_period) == int or type(shock_period) == float:
        shock_period = [int(shock_period)]
    if type(shock) == int or type(shock) == float:
        shock = [shock]
    date_range = pd.date_range(start=e_day, periods=period, freq='MS')
    shock_period_diff = [
        shock_period[i] - shock_period[i - 1] - 1 if i > 0 else
        shock_period[i] - 1 for i in range(len(shock_period))]
    last_period = period - shock_period[-1]
    if method == 'HW':
        for i in range(len(shock_period)):
            if shock_period_diff[i] > 0:
                shocked_data = hw_ind_predict(data, s_day, e_day,
                                              shock_period_diff[i])
            elif i == 0:
                data.loc[date_range[shock_period[i] - 1]] = shock[i]
                e_day = date_range[shock_period[i] - 1]
                continue
            shocked_data.loc[date_range[shock_period[i] - 1]] = shock[i]
            data = pd.concat([data, shocked_data])
            e_day = date_range[shock_period[i] - 1]
        if last_period > 0:
            shocked_data = hw_ind_predict(data, s_day, e_day, last_period)
            shocked_data = pd.concat([data, shocked_data])
    if method == 'ARIMA':
        for i in range(len(shock_period)):
            if shock_period_diff[i] > 0:
                shocked_data = arima_ind_predict(data, s_day, e_day,
                                                 shock_period_diff[i])
            if i == 0:
                shocked_data = pd.Series(shocked_data,
                                         index=date_range[
                                               :shock_period[i] - 1])
            else:
                shocked_data = pd.Series(shocked_data,
                                         index=date_range[shock_period[i - 1]:
                                                          shock_period[i] - 1])
            shocked_data.loc[date_range[shock_period[i] - 1]] = shock[i]
            data = pd.concat([data, shocked_data])
            e_day = date_range[shock_period[i] - 1]
        if last_period > 0:
            shocked_data = arima_ind_predict(data, s_day, e_day, last_period)
            shocked_data = pd.Series(shocked_data,
                                     index=date_range[-last_period:])
            shocked_data = pd.concat([data, shocked_data])

    return shocked_data


def predict_validation_inds(features, feature_methods=None, period=12):
    """ Predict the indicators on the validation set with the better method

    Parameters
    ----------
    features : DataFrame
        The complete features' dataset
    train_ratio : float
        Proportion of training set
    feature_methods : dict
        Features and their corresponding methods, either hw or arima

    Returns
    -------
    val_prediction : DataFrame
        Predicted features on the validation set

    """

    # Check if all features have their methods
    if feature_methods is None:
        feature_methods = {}
    if sorted(features.columns) != sorted(feature_methods.keys()):
        raise ValueError(
            "The input feature set and training methods do not correspond!")
    # Split the data into training and validation sets
    break_point = round(features.shape[0] * train_ratio) - 1
    s_day = features.index[0]
    e_day = features.index[break_point]
    period = features.shape[0] - break_point - 1
    # Predict each feature on the validation set
    pred_val_list = []
    for name, feature in features.iteritems():
        if feature_methods[name] == 'hw':
            pred_val_list.append(hw_ind_predict(feature, s_day, e_day, period))
        elif feature_methods[name] == 'arima':
            pred_val_list.append(
                arima_ind_predict(feature, s_day, e_day, period))
        else:
            raise ValueError(f"{name}'s model is not hw or arima.")
    val_prediction = pd.concat(pred_val_list, axis=1)
    # Save the DataFrame
    val_prediction.to_csv('data/val_prediction.csv')

    return val_prediction
