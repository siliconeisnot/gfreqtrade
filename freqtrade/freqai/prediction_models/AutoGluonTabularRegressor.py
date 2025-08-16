import logging
from typing import Any

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class AutoGluonTabularRegressor(BaseRegressionModel):
    """AutoGluon Tabular AutoML regressor.

    This model leverages AutoGluon's :class:`~autogluon.tabular.TabularPredictor`
    to automatically train an ensemble of models for regression tasks.

    The model requires the optional dependency ``autogluon.tabular`` to be
    installed.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """Train an AutoGluon TabularPredictor.

        Any argument accepted by :meth:`autogluon.tabular.TabularPredictor.fit`
        can be provided via ``model_training_parameters`` in the FreqAI config.
        Common options include ``time_limit`` (maximum training seconds),
        ``presets`` (predefined training configurations), ``hyperparameters``
        (model search space), ``eval_metric``, ``ag_args_fit`` and bagging
        options like ``num_bag_folds`` and ``num_bag_sets``.  Example::

            "freqai": {
                "model_training_parameters": {
                    "time_limit": 600,
                    "num_bag_folds": 5,
                    "num_bag_sets": 1,
                    "presets": "medium_quality",
                    "hyperparameters": {"GBM": {}, "NN_TORCH": {}}
                }
            }
        """
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "AutoGluon is not installed. Install it with "
                "'pip install autogluon.tabular' to use AutoGluonTabularRegressor."
            ) from e

        train = data_dictionary["train_features"].copy()
        train[dk.label_list[0]] = data_dictionary["train_labels"].squeeze()

        tuning_data = None
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            tuning_data = data_dictionary["test_features"].copy()
            tuning_data[dk.label_list[0]] = data_dictionary["test_labels"].squeeze()

        predictor = TabularPredictor(label=dk.label_list[0], problem_type="regression")

        fit_kwargs = self.model_training_parameters.copy()
        time_limit = fit_kwargs.pop("time_limit", None)
        num_bag_folds = fit_kwargs.pop("num_bag_folds", None)
        num_bag_sets = fit_kwargs.pop("num_bag_sets", None)

        params = {"tuning_data": tuning_data, **fit_kwargs}
        if time_limit is not None:
            params["time_limit"] = time_limit
        if num_bag_folds is not None:
            params["num_bag_folds"] = num_bag_folds
        if num_bag_sets is not None:
            params["num_bag_sets"] = num_bag_sets

        predictor = predictor.fit(train, **params)
        return predictor
