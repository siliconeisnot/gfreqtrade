import logging

from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IStrategy


logger = logging.getLogger(__name__)


class FreqaiAutoGluonStrategy(IStrategy):
    """Example strategy using AutoGluon predictions."""

    minimal_roi = {"0": 0.1, "240": -1}
    stoploss = -0.05
    process_only_new_candles = True
    startup_candle_count: int = 40
    can_short = True

    # Threshold beyond which open positions will be closed regardless of direction
    prediction_threshold = DecimalParameter(0.0, 0.05, default=0.02, decimals=3, space="sell")

    plot_config = {
        "main_plot": {},
        "subplots": {"&-s_close": {"&-s_close": {"color": "blue"}}},
    }

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """Enter long/short depending on predicted close change."""
        df.loc[
            (df["do_predict"] == 1) & (df["&-s_close"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "long")
        df.loc[
            (df["do_predict"] == 1) & (df["&-s_close"] < 0),
            ["enter_short", "enter_tag"],
        ] = (1, "short")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """Exit when prediction changes sign or passes configurable threshold."""
        df.loc[
            (df["do_predict"] == 1)
            & ((df["&-s_close"] < 0) | (df["&-s_close"].abs() > self.prediction_threshold.value)),
            "exit_long",
        ] = 1
        df.loc[
            (df["do_predict"] == 1)
            & ((df["&-s_close"] > 0) | (df["&-s_close"].abs() > self.prediction_threshold.value)),
            "exit_short",
        ] = 1
        return df
