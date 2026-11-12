"""Draw a forecaster's scorecard: python examples/scorecard.py [output.png]

Runs the built-in LastValueForecaster on ten kinds of series at low and high noise,
prints the Markdown summary and saves the image (PNG, SVG or PDF by extension).
Put your own forecaster in the Pipeline to score it instead.
"""
import sys

from timegym import LastValueForecaster, Pipeline

card = Pipeline(LastValueForecaster()).scorecard()
print(card)
output = sys.argv[1] if len(sys.argv) > 1 else "scorecard.png"
card.save(output)
print(f"\nSaved {output}")
