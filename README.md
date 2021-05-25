# TimeGym

<<<<<<< HEAD
[![PyPI](https://img.shields.io/pypi/v/timegym)](pypi link)
[![GitHub release](https://img.shields.io/github/v/release/diogoseca/timegym)](https://github.com/diogoseca/timegym/releases/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/diogoseca/timegym/graphs/commit-activity)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diogoseca/timegym)
[![Binder](https://binder.pangeo.io/badge_logo.svg)](https://binder.pangeo.io/v2/gh/diogoseca/timegym/master)
=======
Install using pip:

```
pip install timegym
```
>>>>>>> 57e264da95ae36252c4615a59433b16bd8e68438

**TimeGym** is a simple way to engineer and test forecasting pipelines.

```python
from timegym import Pipeline
from sklearn.preprocessing import PCA
from xgboost import XGBRegressor

pipeline = Pipeline()
pipeline.add(PCA(0.95))
pipeline.add(XGBRegressor)
pipeline.test()
```

## Features

* Feature 1
* TimeGym is compatible with the most popular machine learning and forecasting packages: sklearn, sktime, gluonts, keras, xgboost, catboost, etc.
* Easy install via pip using `pip install timegym`

## Installation

Easy install via pip install:

```bash
pip install timegym
```

## Need help?

[Leave us a ticket on GitHub](https://github.com/diogoseca/timegym/issues/new/choose).

We are commited to making _timegym_ better.
