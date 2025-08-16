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
        Common options include ``time_limit`` (seconds to train), ``presets``
        (predefined training configurations), ``hyperparameters`` (model search
        space), ``hyperparameter_tune_kwargs`` (search strategy), ``eval_metric``
        and ``ag_args_fit``.  Example::

            "freqai": {
                "model_training_parameters": {
                    "time_limit": 600,
                    "presets": "medium_quality",
                    "eval_metric": "mean_absolute_error",
                    "hyperparameter_tune_kwargs": {
                        "num_trials": 5,
                        "scheduler": "local",
                        "searcher": "auto"
                    },
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

        train_params = self.model_training_parameters.copy()
        # Pull tuning related settings so they can be forwarded explicitly
        hyperparameter_tune_kwargs = train_params.pop("hyperparameter_tune_kwargs", None)
        presets = train_params.pop("presets", None)
        eval_metric = train_params.pop("eval_metric", None)

        predictor = predictor.fit(
            train,
            tuning_data=tuning_data,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            presets=presets,
            eval_metric=eval_metric,
            **train_params,
        )
        return predictor
