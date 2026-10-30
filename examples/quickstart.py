"""Run with python examples/quickstart.py after installing timegym."""
from timegym import Pipeline, Scenario, NaiveForecaster

pipeline = Pipeline(NaiveForecaster(strategy="last"))
report = pipeline.test(
    scenarios=[Scenario("random_walk", context_length=120, horizon=12)],
    seeds=[0, 1, 2],
)
report.assert_passed()
print(report.summary)
print(report.cases[0].metrics)
