import pytest

from mltemplate.training.schedulers import warmup_cosine_factor


def f(step):
    return warmup_cosine_factor(step, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)


def test_warmup_is_linear_and_reaches_peak():
    assert f(0) == pytest.approx(0.1)
    assert f(4) == pytest.approx(0.5)
    assert f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0)


def test_cosine_decays_to_min():
    values = [f(s) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))
    assert f(55) == pytest.approx(0.55)  # halfway: midpoint between 1.0 and 0.1
    assert f(100) == pytest.approx(0.1)
    assert f(500) == pytest.approx(0.1)
