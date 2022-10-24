from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame

import machine_learning as ml
from machine_learning import Dataset, utils

sns.set(style='darkgrid')


def preProcess(df: DataFrame) -> DataFrame:
    """
    Pre-Process the BTC dataset, by generalising feature names, correcting
    datatypes, normalising values and handling invalid instances.

    :param df: BTC dataset, should be a DataFrame
    :return: df - DataFrame
    """
    df = ml.handleMissingData(df)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', drop=False, inplace=True)

    logging.info(f"Pre-Processed dataset")
    return df


def exploratoryDataAnalysis(df: DataFrame, dataset_name: str = '', results_dir: str = '') -> None:
    """
    Performs initial investigations on data to discover patterns, to spot
    anomalies, to test hypotheses and to check assumptions with the help
    of summary statistics and graphical representations.

    :param df: BSS dataset, should be a DataFrame
    :param dataset_name: Name of dataset, should be a str
    :param results_dir: Save location for figures, should be a str
    :return: None
    """
    logging.info("Exploratory Data Analysis")

    fig = plt.figure(figsize=(9.5, 7.5))
    graph = sns.heatmap(df.corr(), annot=True, square=True, cmap='Greens', fmt='.2f')
    graph.set_xticklabels(graph.get_xticklabels(), rotation=40)
    fig.suptitle(f"Corresponding Matrix - {dataset_name}")
    if results_dir:
        plt.savefig(utils.joinPath(results_dir, fig._suptitle.get_text(), ext='.png'))

    df_month = df.resample('MS').mean()
    df_week = df.resample('W').mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_month.index, df_month['Close'], c='r', label='Monthly')
    ax.plot(df_week.index, df_week['Close'], c='g', label='Weekly')
    ax.plot(df.index, df['Close'], c='b', label='Daily')
    ax.set(xlabel='Date', ylabel='Close ($USD)')
    plt.legend()
    fig.suptitle(f"Closing Price Vs Date - {dataset_name}")
    if results_dir:
        plt.savefig(utils.joinPath(results_dir, fig._suptitle.get_text(), ext='.png'))

    plt.show()  # displays all figures


def processData(df: DataFrame) -> DataFrame:
    """
    Processes and adapts the BTC dataset to suit time series by adding a
    datetime index. Feature engineering has also been applied by adding
    historical data. It also includes some forms of feature selection,
    certain features will be dropped to reduce dimensionality and
    multi-collinearity which were identified in the previous corresponding
    matrix.

    :param df: BTC dataset, should be a DataFrame
    :return: df - DataFrame
    """
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    # Adds historical data
    df['prev'] = df['Close'].shift()
    df['diff'] = df['Close'].diff()
    df['diff'] = df['diff'].apply(lambda x: int(bool(x > 0)))  # binary encoding
    df['prev_2'] = df['prev'].shift()
    df['diff_2'] = df['prev'].diff()
    df['diff_2'] = df['diff_2'].apply(lambda x: int(bool(x > 0)))  # binary encoding

    df = ml.handleMissingData(df)

    # Changes datatypes
    for col in ['prev', 'diff', 'prev_2', 'diff_2']:
        df[col] = df[col].astype('float64')

    logging.info("Dataset processed")
    return df


def processDataset(dataset: Dataset, overwrite: bool = False) -> Dataset:
    """
    Pre-processes the dataset if applicable, then processes the dataset.

    :param dataset: the dataset, should be a Dataset
    :param overwrite: overwrite existing file, should be a bool
    :return: dataset - Dataset
    """
    name = dataset.name
    dataset.update(name=name + '-pre-processed')

    if overwrite or not dataset.load():
        dataset.apply(preProcess)
        dataset.save()

    dataset.apply(processData)
    dataset.update(name=name + '-processed')
    return dataset


def main(dir_: str) -> None:
    """
    Pre-processes and Processes the datasets.

    :param dir_: Project's path directory, should be a str
    :return: None
    """
    config = ml.Config(dir_, 'BTC-USD')

    results_dir = utils.makePath(dir_, config.results_folder, 'process')

    # Loads the BSS dataset
    dataset = ml.Dataset(config.dataset)
    if not dataset.load():
        raise Exception("Failed to load dataset")

    dataset_name = dataset.name
    print(dataset.df.shape)
    print(dataset.df.head())
    dataset.apply(preProcess)
    dataset.update(name=(dataset_name + '-pre-processed'))
    dataset.save()

    dataset.df.drop('Date', axis=1, inplace=True)

    exploratoryDataAnalysis(dataset.df, dataset_name=dataset.name, results_dir=results_dir)

    dataset.apply(processData)
    dataset.update(name=(dataset_name + '-processed'))
    print(dataset.df.axes)
    print(dataset.df.head())
    print(dataset.df.dtypes)
    fig = plt.figure(figsize=(8, 8))
    graph = sns.heatmap(dataset.df.corr(), annot=True, square=True, cmap='Greens', fmt='.2f')
    graph.set_xticklabels(graph.get_xticklabels(), rotation=40)
    fig.suptitle(f"Processed Corresponding Matrix - {dataset.name}")
    plt.savefig(utils.joinPath(results_dir, fig._suptitle.get_text(), ext='.png'))
    plt.show()
