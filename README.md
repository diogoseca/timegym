# TimeGym
A package for prototyping and debuging forecasting pipelines. TimeGym is compatible with the most popular machine learning and forecasting packages: sklearn, sktime, gluonts, keras, xgboost, catboost, etc.


# Quick Start

**Step 1. Specify the forecasting pipeline.**  
The pipeline should include all steps of preprocessing, statistical modeling, machine learning, and hyperparameter optimization:

```python
from timegym import Pipeline
from sklearn.preprocessing import PCA
from xgboost import XGBRegressor

pipeline = Pipeline()
pipeline.add(PCA(0.95))
pipeline.add(XGBRegressor)
```

**Step 2. Test the pipeline**  
Let's see how the pipeline reacts to the most common problems in forecasting:

```python
pipeline.test()
```

After a few minutes (time varies according to the pipeline) you should see results.

# Need help?

[Leave us a ticket on GitHub](https://github.com/diogoseca/timegym/issues/new/choose).

We are commited to making _timegym_ better.