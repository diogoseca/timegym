"""Run with python examples/quickstart.py after installing timegym."""
from timegym import LastValueForecaster, Pipeline, Scenario

pipeline = Pipeline(LastValueForecaster())
result = pipeline.run(
    scenarios=[Scenario("random_walk", history_length=120, horizon=12)],
    seeds=[0, 1, 2],
)
result.assert_passed()
print(result.report.summary)
print(result.cases[0].metrics)
