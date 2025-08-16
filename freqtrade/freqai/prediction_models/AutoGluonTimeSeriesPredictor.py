import logging
from typing import Any


try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

import numpy as np
import pandas as pd

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class AutoGluonTimeSeriesPredictor(BaseRegressionModel):
    """AutoGluon TimeSeries AutoML regressor.

    This model leverages AutoGluon's :class:`~autogluon.timeseries.TimeSeriesPredictor`
    to automatically train an ensemble of models for time series regression tasks.

    The model requires the optional dependency ``autogluon.timeseries`` to be
    installed.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """Train an AutoGluon TimeSeriesPredictor.

        Any argument accepted by :meth:`autogluon.timeseries.TimeSeriesPredictor.fit`
        can be provided via ``model_training_parameters`` in the FreqAI config.
        """
        try:
            from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "AutoGluon is not installed. Install it with "
                "'pip install autogluon.timeseries' to use AutoGluonTimeSeriesPredictor."
            ) from e

        train_params = self.model_training_parameters.copy()
        freq = train_params.pop("freq", None)
        num_gpus = train_params.pop("num_gpus", 1 if torch and torch.cuda.is_available() else 0)
        if num_gpus > 0 and not (torch and torch.cuda.is_available()):
            logger.warning("num_gpus requested but CUDA is not available. Using CPU")
            num_gpus = 0
        ag_args_fit = train_params.pop("ag_args_fit", {})
        ag_args_fit["num_gpus"] = num_gpus

        train = data_dictionary["train_features"].copy()
        train["target"] = data_dictionary["train_labels"].squeeze()
        train["item_id"] = dk.pair
        train["timestamp"] = (
            data_dictionary["train_dates"]
            .iloc[data_dictionary["train_features"].index]
            .reset_index(drop=True)
        )

        self.train_ts = TimeSeriesDataFrame.from_data_frame(
            train,
            id_column="item_id",
            timestamp_column="timestamp",
        )

        tuning_ts = None
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            test = data_dictionary["test_features"].copy()
            test["target"] = data_dictionary["test_labels"].squeeze()
            test["item_id"] = dk.pair
            test["timestamp"] = (
                data_dictionary["train_dates"]
                .iloc[data_dictionary["test_features"].index]
                .reset_index(drop=True)
            )
            tuning_ts = TimeSeriesDataFrame.from_data_frame(
                test,
                id_column="item_id",
                timestamp_column="timestamp",
            )

        prediction_length = self.freqai_info.get("feature_parameters", {}).get(
            "label_period_candles", 1
        )
        predictor = TimeSeriesPredictor(
            prediction_length=prediction_length,
            target="target",
            freq=freq,
        )
        predictor = predictor.fit(
            self.train_ts,
            tuning_data=tuning_ts,
            ag_args_fit=ag_args_fit,
            **train_params,
        )
        return predictor

    def predict(
        self, unfiltered_df: pd.DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Filter the prediction features data and predict with them."""
        try:
            from autogluon.timeseries import TimeSeriesDataFrame
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "AutoGluon is not installed. Install it with "
                "'pip install autogluon.timeseries' to use AutoGluonTimeSeriesPredictor."
            ) from e

        dk.find_features(unfiltered_df)
        dk.data_dictionary["prediction_features"], _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )

        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True
        )

        df = dk.data_dictionary["prediction_features"].copy()
        df["item_id"] = dk.pair
        df["timestamp"] = unfiltered_df.loc[df.index, "date"].reset_index(drop=True)

        ts = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column="item_id",
            timestamp_column="timestamp",
        )

        forecasts = self.model.predict(
            self.train_ts, known_covariates=ts, prediction_length=len(ts)
        )
        pred_df = forecasts.to_pandas().reset_index(level=0, drop=True)
        if "mean" in pred_df.columns:
            pred_df = pred_df.rename(columns={"mean": dk.label_list[0]})
        else:
            pred_df.columns = dk.label_list

        pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)

        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        return pred_df, dk.do_predict
