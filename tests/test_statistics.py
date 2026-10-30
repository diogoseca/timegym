import numpy as np
import pytest

from timegym.statistics import paired_t_interval, wilson_interval


def test_wilson_known_half_success_interval():
    result = wilson_interval([True, False] * 5)
    assert result['count'] == 5 and result['n'] == 10 and result['rate'] == .5
    np.testing.assert_allclose(result['ci95'], [.236593090512564, .763406909487436])


def test_paired_t_known_interval_and_constant_differences():
    result = paired_t_interval([1., 2., 3., 4., 5.])
    assert result['mean'] == 3 and result['n'] == 5
    np.testing.assert_allclose(result['ci95'], [1.036756838522439, 4.963243161477561])
    assert paired_t_interval([2., 2.])['ci95'] == [2., 2.]


@pytest.mark.parametrize('values', [[], [2], [np.nan], [[True, False]]])
def test_wilson_rejects_nonbinary_or_empty_vectors(values):
    with pytest.raises(ValueError, match='binary observations'):
        wilson_interval(values)


@pytest.mark.parametrize('values', [[], [1], [1, np.inf], [[1, 2]]])
def test_paired_t_rejects_invalid_case_vectors(values):
    with pytest.raises(ValueError, match='at least two finite cases'):
        paired_t_interval(values)
