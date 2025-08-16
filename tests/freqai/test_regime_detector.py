import pandas as pd

from freqtrade.freqai.regime.regime_detector import RegimeDetector


def test_regime_detector_basic():
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    detector = RegimeDetector(n_clusters=2, volatility_window=2, trend_window=2)
    regimes = detector.detect(df)
    assert len(regimes) == len(df)
    # ensure labels are between 0 and n_clusters - 1
    assert regimes.min() >= 0
    assert regimes.max() < 2
