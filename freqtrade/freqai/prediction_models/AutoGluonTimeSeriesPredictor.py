import logging
from typing import Any

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class AutoGluonTimeSeriesPredictor(BaseRegressionModel):
    """AutoGluon TimeSeries AutoML regressor.

    This model leverages AutoGluon's :class:`~autogluon.timeseries.TimeSeriesPredictor`
    to automatically train models for time series forecasting.

    The model requires the optional dependency ``autogluon.timeseries`` to be
    installed.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """Train an AutoGluon TimeSeriesPredictor.

        Any argument accepted by :meth:`autogluon.timeseries.TimeSeriesPredictor.fit`
        can be provided via ``model_training_parameters`` in the FreqAI config.  In
        addition, ``freq`` can be specified to override the data frequency used when
        constructing the predictor.

        Example::

            "freqai": {
                "model_training_parameters": {
                    "freq": "1h",
                    "prediction_length": 1,
                    "time_limit": 600
                }
            }
        """
        try:
            from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "AutoGluon is not installed. Install it with "
                "'pip install autogluon.timeseries' to use AutoGluonTimeSeriesPredictor."
            ) from e

        train_df = data_dictionary["train_features"].copy()
        train_df[dk.label_list[0]] = data_dictionary["train_labels"].squeeze()

        tuning_df = None
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            tuning_df = data_dictionary["test_features"].copy()
            tuning_df[dk.label_list[0]] = data_dictionary["test_labels"].squeeze()

        train_len = len(train_df)
        full_dates = data_dictionary["train_dates"]
        train_dates = full_dates.iloc[:train_len]
        train_df["timestamp"] = train_dates.values
        train_df["item_id"] = dk.pair
        train_tsd = TimeSeriesDataFrame.from_data_frame(
            train_df, id_column="item_id", timestamp_column="timestamp"
        )

        tuning_tsd = None
        if tuning_df is not None:
            test_len = len(tuning_df)
            test_dates = full_dates.iloc[train_len : train_len + test_len]
            tuning_df["timestamp"] = test_dates.values
            tuning_df["item_id"] = dk.pair
            tuning_tsd = TimeSeriesDataFrame.from_data_frame(
                tuning_df, id_column="item_id", timestamp_column="timestamp"
            )

        train_params = self.model_training_parameters.copy()
        freq = train_params.pop("freq", None)
        if freq is None:
            from freqtrade.exchange import timeframe_to_minutes

            freq = f"{timeframe_to_minutes(dk.config['timeframe'])}min"

        prediction_length = train_params.pop(
            "prediction_length",
            self.freqai_info["feature_parameters"].get("label_period_candles", 1),
        )

        predictor = TimeSeriesPredictor(
            target=dk.label_list[0],
            prediction_length=prediction_length,
            freq=freq,
        )

        predictor = predictor.fit(
            train_data=train_tsd,
            tuning_data=tuning_tsd,
            **train_params,
        )
        return predictor
