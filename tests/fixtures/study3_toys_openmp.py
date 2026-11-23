"""The toy learners of study3_toys.py in a module that also loads scikit-learn's OpenMP runtime, as the study's
learner module does. Used to test that a unit's process runs every thread pool on one thread."""
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401  (loads the OpenMP runtime)

from study3_toys import SeedStream, build  # noqa: F401
