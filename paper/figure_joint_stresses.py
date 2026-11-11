"""Joint financial uncertainty: preserve target paths, break their pairing."""
import hashlib
from pathlib import Path

import numpy as np
from matplotlib import colormaps

from vector_canvas import Canvas
from figure_common import AXIS, BLUE, GRAY, GRID, MUTED, notebook_literal


def build_joint_stresses(root: Path):
    """Return the joint reference scene and its paired sample evidence."""
    from timegym import Scenario, metrics

    notebook_path = root / "tutorials/12_joint_financial_uncertainty.ipynb"
    spec = notebook_literal(notebook_path, "joint-demo-specs", "joint_specs", expect_len=4)[1]
    case = Scenario(context_length=192, horizon=6, n_targets=2, **spec).generate(seed=11)
    forecast = case.reference.forecast(n_samples=768, seed=91)
    original = forecast.samples
    permutation = np.random.default_rng(18).permutation(768)
    broken = original.copy()
    broken[:, :, 1] = original[permutation, :, 1]
    np.testing.assert_array_equal(np.sort(original, axis=0), np.sort(broken, axis=0))
    h = 3
    clouds = [(draw[:, h]-case.reference.mean[h])/np.sqrt(case.reference.variance[h]) for draw in (original, broken)]
    correlations = [np.corrcoef(draw.reshape(768, -1), rowvar=False) for draw in (original, broken)]
    aggregate_sd = [float(points.mean(axis=1).std(ddof=1)) for points in clouds]
    crps = [float(metrics.crps(case.observed.to_numpy(), draw)) for draw in (original, broken)]
    np.testing.assert_allclose(crps[0], crps[1], rtol=0, atol=1e-13)

    canvas = Canvas(448, 412)
    canvas.text(0, 11, "Same marginal uncertainty. Different joint futures.", size=10.2, weight="bold")
    canvas.text(0, 29, "Merton log prices | Whole target trajectories are re-paired across samples", size=7.5)
    cmap = colormaps["RdBu_r"]
    for col, (title, points, corr) in enumerate(zip(("Paired futures", "Partners shuffled"), clouds, correlations)):
        left, top, size = 61+col*218, 63, 112
        canvas.text(left+size/2, 48, title, size=9, weight="bold", align="center")
        xx = lambda x: left+(x+4.5)/9*size
        yy = lambda y: top+size-(y+4.5)/9*size
        canvas.line(left, top, left, top+size, color=AXIS, width=.5)
        canvas.line(left, top+size, left+size, top+size, color=AXIS, width=.5)
        canvas.line(xx(0), top, xx(0), top+size, color=GRID, width=.45)
        canvas.line(left, yy(0), left+size, yy(0), color=GRID, width=.45)
        for x, y in points[:384]:
            if abs(x) <= 4.5 and abs(y) <= 4.5:
                canvas.circle(xx(x), yy(y), r=.6, color=GRAY)
        canvas.cross(xx(0), yy(0), r=2.2, color=BLUE, width=.7)
        for value in (-4, 0, 4):
            canvas.text(xx(value), top+size+8, str(value), size=6.5, align="center")
            canvas.text(left-5, yy(value), str(value), size=6.5, align="right")
        canvas.text(left+size/2, 195, "Standardized target 0", size=7, align="center")
        canvas.text(left, 57, "Target 1", size=6.5, color=MUTED)
        canvas.text(left+size/2, 209, "Across horizons and targets", size=8, align="center")
        top = 232
        width = size/12
        for i in range(12):
            for j in range(12):
                color = cmap((corr[i, j]+1)/2)
                fill = "#"+"".join(f"{round(255*c):02x}" for c in color[:3])
                canvas.rect(left+j*width, top+i*width, width+.05, width+.05, fill=fill)
        for coordinate in (0, 4, 10):
            label = f"h{coordinate//2+1}:y0"
            canvas.text(left-4, top+(coordinate+.5)*width, label, size=6, align="right")
            canvas.text(left+(coordinate+.5)*width, top+size+8, label, size=6, align="center")
        canvas.text(left+size/2, 369, f"Aggregate SD: {aggregate_sd[col]:.2f}", size=8, weight="bold", align="center")
    canvas.text(224, 223, "Correlation coordinates: (h1,y0), (h1,y1), (h2,y0), ...", size=7, align="center")
    for i in range(80):
        color = cmap(1-i/79)
        fill = "#"+"".join(f"{round(255*c):02x}" for c in color[:3])
        canvas.rect(421, 232+i*112/80, 6, 112/80+.05, fill=fill)
    for y, text in ((232, "+1"), (288, "0"), (344, "-1")):
        canvas.text(431, y, text, size=6.2)
    canvas.text(224, 390, "Each series keeps its values and temporal paths; only cross-series pairing changes.", size=7, align="center")
    canvas.text(224, 405, "Aggregate = average standardized error at horizon 4. Marginal CRPS is unchanged.", size=7, align="center")
    metadata = dict(
        specification_source=str(notebook_path.relative_to(root)), specification_cell="joint-demo-specs",
        input_notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        manifest=case.manifest, sample_count=768, sample_seed=91, permutation_seed=18,
        horizon_displayed=4, cloud_display_count=384, cloud_limits=[-4.5, 4.5],
        cloud_visible_counts=[int(np.all(np.abs(p[:384]) <= 4.5, axis=1).sum()) for p in clouds],
        original_samples=original.tolist(), permutation=permutation.tolist(),
        reference_mean=case.reference.mean.tolist(), reference_variance=case.reference.variance.tolist(),
        correlation_maps=[c.tolist() for c in correlations], aggregate_sd=aggregate_sd, marginal_crps=crps,
        interpretation="Selected mechanism demonstration; no new empirical fault-study or financial-performance claim")
    return canvas, metadata
