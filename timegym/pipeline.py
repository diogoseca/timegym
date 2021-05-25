
from data.synthetic import basic_synthetics
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_y
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import RecursiveRegressionForecaster
from sktime.forecasting.compose import DirectRegressionForecaster
#from sktime.forecasting.compose import MultioutputRegressionForecaster
from sklearn.pipeline import Pipeline as SkPipeline
from sktime.performance_metrics.forecasting import smape_loss
# other metrics: https://stats.stackexchange.com/questions/425390/how-do-i-decide-when-to-use-mape-smape-and-mase-for-time-series-analysis-on-sto
import types
from sklearn.base import TransformerMixin, RegressorMixin, BaseEstimator
from collections import defaultdict
from sktime.forecasting.naive import NaiveForecaster
import inspect
from tqdm.auto import tqdm
#from pqdm.threads import pqdm
#from pqdm.processes import pqdm
import seaborn as sns


class FunctionTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, transformation, **hyperparameters):
        self.transformation = transformation
        self.hyperparameters = hyperparameters

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.transformation(X, **self.hyperparameters)
    
    
class Pipeline(BaseEstimator, TransformerMixin):
    
    def __init__(self, strategy='recursive', window_length=100):
        """
        """
        self.strategy = strategy
        self.window_length = window_length
        self.steps = []
    
    
    def sample_hyperparameters(search_space: dict):
        raise NotImplementedError
        #for hyperparameter in search_space:
            
        
    
    def add(self, transformation, append=False, **hyperparameters):
        """Adds one transformation step to the pipeline
        
        Parameters
        ----------
        transformation : sklearn.base.TransformerMixin, function
            One transformation class, or a function.
        **hyperparamters : string, int, float, optuna.suggestion, optional (default=None)
            other hyperparameters to be initialized  of series, will be displayed in figure legend
        Returns
        -------
        fig : plt.Figure
        ax : plt.Axis
        """
        if hyperparameters:
            for h in hyperparameters:
                if isinstance(hyperparameters[h], tuple):
                    #TODO: use optuna to generate these hyperparameters dymically if requested by user
                    raise NotImplementedError
        
        if inspect.isfunction(transformation):
            transformation = FunctionTransformation(partial(transformation, **hyperparameters))
        elif inspect.isclass(transformation):
            transformation = transformation(**hyperparameters)
            if isinstance(transformation, BaseEstimator):
                pass
            elif not isinstance(transformation, TransformerMixin):
                raise NotImplementedError
                #transformation = FunctionTransformation(transformations)
            
        self.steps.append(transformation)
        
    
    def predict(self, y_train, y_test):
        # TODO: add support for time series regression
        #DirectTimeSeriesRegressionForecaster
        #RecursiveTimeSeriesRegressionForecaster
        
        if isinstance(self.steps[-1], RegressorMixin):
            # convert pipeline steps to a sklearn pipeline
            pipeline = SkPipeline([(type(step).__name__, step) for step in self.steps])
    
            # create the forecaster according to one of the following strategies: 
            # 'recursive' (default), 'direct', 'combination', 'multioutput'
            if self.strategy=='recursive':
                forecaster = RecursiveRegressionForecaster(pipeline, window_length=self.window_length)
            elif self.strategy=='direct': 
                forecaster = DirectRegressionForecaster(pipeline)
            elif self.strategy=='combination':
                raise NotImplementedError 
            elif self.strategy=='multioutput':
                #forecaster = MultioutputRegressionForecaster(pipeline)
                #better code still yet to come
                raise NotImplementedError
        else: #ASSUME THAT LAST STEP IS A FORECASTER
            # convert pipeline steps to a sklearn pipeline (except for the last step
            pipeline = SkPipeline([(type(step).__name__, step) for step in self.steps[:-1]])
            y_train = pipeline.fit_transform(y_train)
            forecaster = self.steps[-1]
                
        forecaster = forecaster.fit(y_train)
        y_pred = forecaster.predict(ForecastingHorizon(y_test.index, is_relative=False))
        return y_pred
    
    def plot_tests(self, datasets=basic_synthetics, tests_per_dataset=3, naive=None):
        """
        plot_tests plots a matrix plot of forecasts including the ideal 'oracle' values.
        
        Parameters
        ----------
        datasets : list of datasets, optional (default=timegym.synthetic.basic_synthetics)
            List of datasets used for the tests.
            
        tests_per_dataset : int, optional (default=3)
            Number of tests generated per dataset.
            
        naive : str{"last", "mean", "drift"}, optional (default=None)
            Naive strategy used to make forecasts:
            * "last" : forecast the last value in the
                        training series when sp is 1.
                        When sp is not 1,
                        last value of each season
                        in the last window will be
                        forecasted for each season.
            * "mean" : forecast the mean of last window
                         of training series when sp is 1.
                         When sp is not 1, mean of all values
                         in a season from last window will be
                         forecasted for each season.
            * "drift": forecast by fitting a line between the
                        first and last point of the window and
                         extrapolating it into the future
        """
        self.__forecaster = self.steps[-1]
        
        fig, axes = plt.subplots(nrows=len(datasets), ncols=tests_per_dataset, 
                                 figsize=(tests_per_dataset*10,len(datasets)*5),
                                 squeeze=False)

        for d, dataset in enumerate(datasets):
            axes[d, 0].set_ylabel(dataset.shortname)
            for t in range(tests_per_dataset):
                # training and test
                y_train, y_test, oracle = dataset.get_data()
                y_pred = self.predict(y_train, y_test)
                smape_pred = smape_loss(oracle[y_test.index], y_pred) # pred vs oracle
                smape_noise = smape_loss(oracle[y_test.index], y_test) # noisy-signal vs oracle 
                pd.Series(oracle).plot(ax=axes[d, t], color='gray', label='Oracle (noiseless signal)', legend=True, linewidth=1)
                pd.Series(pd.concat([y_train, y_test])).plot(ax=axes[d, t], color='gray', label='Observations (sMAPE: {:.3f})'.format(smape_noise), legend=True, marker='.', markersize=2, linewidth=0)
                pd.Series(y_pred).plot(ax=axes[d, t], color='blue', label='Predictions (sMAPE: {:.3f})'.format(smape_pred), legend=True, marker='x', markersize=1.5, linewidth=0)
                if naive:
                    naive_forecaster = NaiveForecaster(strategy=naive).fit(y_train)
                    y_pred_naive = naive_forecaster.predict(ForecastingHorizon(y_test.index, is_relative=False))
                    smape_naive = smape_loss(oracle[y_test.index], y_pred_naive) # naive vs oracle
                    pd.Series(y_pred_naive).plot(ax=axes[d, t], color='green', label='Naive (sMAPE: {:.3f})'.format(smape_naive), legend=True, marker='.', markersize=1, linewidth=0)
                split_date = y_train.index[-1] + (y_test.index[0] - y_train.index[-1])/2
                axes[d, t].axvline(split_date, color='blue', linewidth=2, linestyle='--')
                axes[d, t].set_xlim(xmin=oracle.index[0], xmax=oracle.index[-1])
                axes[d, t].legend(loc='upper left')
        return fig
    
    
    def plot_error(self, datasets=basic_synthetics, tests_per_dataset=100, n_jobs=4):
        self.__forecaster = self.steps[-1]
        
        fig, ax = plt.subplots(nrows=len(datasets), ncols=1, squeeze=False,
                                 figsize=(12, len(datasets)*5))
        
        def prediction_smape(d):
            y_train, y_test, oracle = datasets[d].get_data()
            y_pred = self.predict(y_train, y_test)
            return smape_loss(oracle[y_test.index], y_pred)

        tests_datasets_indices = list(range(len(datasets))) * tests_per_dataset
        results = []
        for d in tqdm(tests_datasets_indices):
            results.append(prediction_smape(d))
        results = pd.Series(results, index=tests_datasets_indices)
        
        # plot results
        for d in range(len(datasets)):
            sns.histplot(results.loc[d], ax=ax[d, 0], kde=True, stat='count', bins=tests_per_dataset//5)
            ax[d, 0].set
        return fig
    
