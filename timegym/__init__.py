"""
TimeGym
~~~~~~~~~~~~~~~~~~~~~
TimeGym is a simple way to engineer and test forecasting pipelines.
Basic usage:
   >>> from timegym import Pipeline
   >>> from sklearn.preprocessing import PCA
   >>> from xgboost import XGBRegressor

   >>> pipeline = Pipeline()
   >>> pipeline.add(PCA(0.95))
   >>> pipeline.add(XGBRegressor)
   >>> pipeline.test()
   
Full documentation is at <https://timegym.readthedocs.io>.
:copyright: (c) 2021 by Diogo Seca.
:license: MIT, see LICENSE for more details.
"""

__title__ = 'timegym'
__description__ = 'TimeGym is a simple way to engineer and test forecasting pipelines.'
__url__ = 'https://github.com/diogoseca/timegym'
__version__ = '0.0.1'
__author__ = 'Diogo Seca'
__author_email__ = 'diogoseca@gmail.com'
__license__ = 'MIT'
__copyright__ = 'Copyright 2021 Diogo Seca'
