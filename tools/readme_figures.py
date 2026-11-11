"""Draw the README's scorecard figures: python tools/readme_figures.py [folder]

Runs the README's "More things to do with a scorecard" examples with the
last-value forecaster and saves each figure as docs/images/readme-*.png
(or into the folder given). Rerun it whenever the plots change style.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from timegym import NaiveForecaster, Pipeline, Scenario

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/images")
folder.mkdir(parents=True, exist_ok=True)
pipeline = Pipeline(NaiveForecaster())


def save(figure, name):
    figure.savefig(folder / name, dpi=110, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {folder / name}")


trend = pipeline.scorecard(["trend"], instances=3)
trend.save(folder / "readme-trend-scorecard.png")
print(f"Saved {folder / 'readme-trend-scorecard.png'}")
save(trend.run.plot_forecasts(), "readme-trend-forecasts.png")
save(
    pipeline.scorecard(["trend"], instances=20).run.plot_errors(metric="mse"),
    "readme-trend-errors.png",
)

suite = [
    Scenario("seasonal", context_length=240, horizon=24),
    Scenario("random_walk", parameters={"drift": 0.1}),
]
save(pipeline.run(suite, seed=0, instances=3).plot_forecasts(), "readme-custom-scenarios.png")
