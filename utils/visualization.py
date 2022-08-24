import os
import string
import matplotlib.pyplot as plt
import seaborn as sns
from utils.variable_selection import unify_index


def plot_heatmap(data, label, save=False):
    """Plot a heatmap to show Pearson correlation coefficients

    Parameters
    ----------
    data : DataFrame
        DataFrame containing all variables
    label : str
        Topic for this heatmap
    save : Bool, optional
        If True, the plotting will be saved to the figures folder

    """

    corr_coef = data.dropna().corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_coef, cmap='viridis', ax=ax)
    ax.set_title(f'{string.capwords(label)} Heatmap')
    fig.tight_layout()
    if save:
        os.makedirs('figures', exist_ok=True)
        fig.savefig(f'figures/{label.lower().replace(" ", "_")}.png')
    plt.show()

    return


def plot_one_to_one_trend(s1, s2, save=False):
    """ Plot the trend of two sets of time series data on the same graph

    Parameters
    ----------
    s1 : Series
        Time series data with name
    s2 : Series
        Time series data with name
    save : bool
        If True, the plotting will be saved to the figures folder

    """

    s1, s2 = unify_index(s1, s2)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    p1, = ax1.plot(s1, color='tab:blue', label=s1.name)
    p2, = ax2.plot(s2, color='tab:red', label=s2.name)
    ax1.set_ylabel(s1.name)
    ax2.set_ylabel(s2.name)
    ax1.yaxis.label.set_color('tab:blue')
    ax2.yaxis.label.set_color('tab:red')
    ax1.legend(handles=[p1, p2], loc='best')
    fig.tight_layout()

    if save:
        os.makedirs('figures/one_to_one_trend', exist_ok=True)
        fig.savefig(f'figures/one_to_one_trend/'
                    f'{s1.name.lower()}_{s2.name.lower()}.png')
    plt.show()

    return


def plot_one_to_one_trends(features, target, save=False):
    """Plot one-to-one relationships of all features with the target variable

    Parameters
    ----------
    features : DataFrame
        All features' dataset
    target : Series
        The target variable
    save : bool
        If True, the plotting will be saved to the figures folder

    """

    for _, feature in features.iteritems():
        plot_one_to_one_trend(target, feature, save)

    return


def plot_val_pred(original, better, other, labels, save=False):
    """ Plot the comparison of the two models for prediction results on the
    validation set

    Parameters
    ----------
    original : Series
        Original feature data
    better : Series
        The better prediction
    other : Series
        The other prediction
    labels : list
        Names of the two models
    save : bool
        If True, the plotting will be saved to the figures folder

    """

    # Add the last value in the training set to the predictions
    pred_length = better.shape[0]
    better.loc[original.index[-pred_length - 1]] = original.iloc[
        -pred_length - 1]
    other.loc[original.index[-pred_length - 1]] = original.iloc[
        -pred_length - 1]
    better = better.sort_index()
    other = other.sort_index()
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(original, color='tab:blue', label='original')
    ax.plot(better, color='tab:cyan', label=labels[0] + ' (better)')
    ax.plot(other, color='tab:green',
            linestyle='dashed', label=labels[1])
    ax.legend(loc='best')
    ax.set_ylabel(original.name)
    ax.set_title(f'{original.name} Prediction on the Validation Set')
    fig.tight_layout()
    if save:
        os.makedirs('figures/val', exist_ok=True)
        fig.savefig(f'figures/val/{original.name}_val_pred.png')
    plt.show()


def plot_val_feature_preds(features, hw_pred, arima_pred,
                           feature_methods, save=False):
    """ Plot the comparison of the two models for all feature prediction
    results on the validation set

    Parameters
    ----------
    features : DataFrame
        Original (True) feature values
    hw_pred : DataFrame
        Holt-Winter model's feature prediction
    arima_pred : DataFrame
        ARIMA's feature prediction
    feature_methods : dict
        Features and their corresponding better models
    save : bool
        If True, the plotting will be saved to the figures folder

    """

    for name, feature in features.iteritems():
        if feature_methods[name] == 'hw':
            plot_val_pred(feature, hw_pred[name], arima_pred[name],
                          ['HW', 'ARIMA'], save)
        elif feature_methods[name] == 'arima':
            plot_val_pred(feature, arima_pred[name], hw_pred[name],
                          ['ARIMA', 'HW'], save)
        else:
            raise ValueError("Invalid time series model!")

    return


def plot_val_target_preds(target, preds, save=False):
    """ Plot the comparison of the two models for target prediction results
    on the validation set

    Parameters
    ----------
    target : Series
        Original (True) target variable values
    preds : DataFrame
        Predicted target values on the validation set
    save : bool
        If True, the plotting will be saved to the figures folder

    """

    plot_val_pred(target, preds['OLS'], preds['LSTM'], ['OLS', 'LSTM'], save)

    return


def plot_feature_prediction(original, prediction, label, save=False):
    """ Plot the future feature values predicted using a time series model

    Parameters
    ----------
    original : Series
        Original (True) feature values
    prediction : Series
        Predicted feature values
    label : str
        The feature name
    save : bool
        If True, the plotting will be saved to the figures folder

    """
    # Add the last value in the training set to the predictions
    pred_length = prediction.shape[0]
    prediction.loc[original.index[-1]] = original.iloc[-1]
    prediction = prediction.sort_index()
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(original, color='tab:blue', label='original')
    ax.plot(prediction, color='tab:blue', linestyle='dashed',
            label='predicted')
    ax.legend(loc='best')
    ax.set_title(f'Out-of-Sample Forecasts of {label}')
    ax.set_ylabel(label)
    fig.tight_layout()
    if save:
        os.makedirs('figures/forecast', exist_ok=True)
        fig.savefig(f'figures/forecast/{label}_forecast.png')
    plt.show()

    return


def plot_feature_predictions(original, prediction, save=False):
    """ Plot the forecasted values for all features

    Parameters
    ----------
    original : Series
        Original (True) feature values
    prediction : Series
        Predicted feature values
    save : bool
        If True, the plotting will be saved to the figures folder

    """

    for name, original_feature in original.iteritems():
        predicted_feature = prediction.loc[:, name]
        plot_feature_prediction(
            original_feature, predicted_feature, name, save)

    return


def plot_target_predictions(original, prediction, save=False):
    """ Plot the forecasted values for the target value

    Parameters
    ----------
    original : Series
        Original (True) feature values
    prediction : DataFrame
        Predicted feature values from both models
    save : bool
        If True, the plotting will be saved to the figures folder

    """

    # Add the last true value to the predictions
    prediction.loc[original.index[-1], :] = original.iloc[-1]
    prediction = prediction.sort_index()
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(original, color='tab:blue', label='original')
    ax.plot(
        prediction['OLS'], color='tab:cyan', linestyle='dashed', label='OLS')
    ax.plot(prediction['LSTM'],
            color='tab:green', linestyle='dashed', label='LSTM')
    ax.legend(loc='best')
    ax.set_title(f'Out-of-Sample Forecasts of {original.name}')
    ax.set_ylabel(original.name)
    fig.tight_layout()
    if save:
        os.makedirs('figures/forecast', exist_ok=True)
        fig.savefig(f'figures/forecast/{original.name}_forecast.png')
    plt.show()

    return
