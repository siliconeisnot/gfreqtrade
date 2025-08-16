"""Tools to detect market regimes using clustering."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame, Series
from sklearn.cluster import KMeans


@dataclass
class RegimeDetector:
    """Simple regime detector based on KMeans clustering.

    The detector extracts basic volatility and trend features from closing prices
    before applying a ``KMeans`` clustering algorithm.  The resulting cluster label
    represents the market regime.
    """

    n_clusters: int = 2
    volatility_window: int = 20
    trend_window: int = 50
    random_state: int = 0

    def _prepare(self, df: DataFrame) -> DataFrame:
        """Compute volatility and trend features from a dataframe."""
        price = df["close"]
        # Volatility estimated by rolling standard deviation of returns
        volatility = price.pct_change().rolling(self.volatility_window).std()
        # Trend estimated by percent change of a moving average
        trend = price.rolling(self.trend_window).mean().pct_change()
        feats = pd.concat([volatility, trend], axis=1)
        feats.columns = ["volatility", "trend"]
        return feats.dropna()

    def detect(self, df: DataFrame) -> Series:
        """Return regime labels for each row of ``df``."""
        feats = self._prepare(df)
        if feats.empty:
            return pd.Series([0] * len(df), index=df.index)
        model = KMeans(n_clusters=self.n_clusters, n_init="auto", random_state=self.random_state)
        labels = model.fit_predict(feats)
        series = pd.Series(labels, index=feats.index)
        # Forward fill to align to original dataframe length
        return series.reindex(df.index, method="ffill").fillna(0).astype(int)
