"""Draw the README's figures: python tools/readme_figures.py [folder]

Saves the scorecard of the built-in LastValueForecaster as docs/images/scorecard.png
and runs the README's "More things to do with a scorecard" examples, saving each
figure as docs/images/readme-*.png (or into the folder given). Rerun it whenever
the plots change style.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from timegym import LastValueForecaster, Pipeline

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/images")
folder.mkdir(parents=True, exist_ok=True)
pipeline = Pipeline(LastValueForecaster())


def save(figure, name):
    figure.savefig(folder / name, dpi=110, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {folder / name}")


pipeline.scorecard().save(folder / "scorecard.png")
print(f"Saved {folder / 'scorecard.png'}")
trend = pipeline.scorecard(["trend"], instances=3)
save(trend.run.plot_forecasts(), "readme-trend-forecasts.png")
save(
    pipeline.scorecard(["trend"], instances=20).run.plot_errors(metric="mse"),
    "readme-trend-errors.png",
)
