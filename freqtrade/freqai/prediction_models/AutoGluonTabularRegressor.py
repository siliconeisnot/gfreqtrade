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

    def _configure_gpus(self) -> None:
        ag_args_fit = self.model_training_parameters.get("ag_args_fit", {})
        if "ag_args_fit" not in self.model_training_parameters:
            self.model_training_parameters["ag_args_fit"] = ag_args_fit
        if "num_gpus" in ag_args_fit:
            return
        num_gpus = self.model_training_parameters.pop("num_gpus", None)
        if num_gpus is None:
            try:  # pragma: no cover - optional dependency
                import torch
            except ImportError:  # pragma: no cover - optional dependency
                pass
            else:
                if torch.cuda.is_available():
                    num_gpus = 1
        if num_gpus is None:
            try:  # pragma: no cover - optional dependency
                from autogluon.common.utils.resource_utils import get_gpu_count
            except ImportError:  # pragma: no cover - optional dependency
                pass
            else:
                count = get_gpu_count()
                if count > 0:
                    num_gpus = count
        if num_gpus is not None:
            ag_args_fit["num_gpus"] = num_gpus

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """Train an AutoGluon TabularPredictor.

        Any argument accepted by :meth:`autogluon.tabular.TabularPredictor.fit`
        can be provided via ``model_training_parameters`` in the FreqAI config.
        Common options include ``time_limit`` (seconds to train), ``presets``
        (predefined training configurations), ``hyperparameters`` (model
        search space), ``eval_metric`` and ``ag_args_fit``.  Example::

            "freqai": {
                "model_training_parameters": {
                    "time_limit": 600,
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

        # Enable GPU training if available unless configured otherwise
        self._configure_gpus()

        train = data_dictionary["train_features"].copy()
        train[dk.label_list[0]] = data_dictionary["train_labels"].squeeze()

        tuning_data = None
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            tuning_data = data_dictionary["test_features"].copy()
            tuning_data[dk.label_list[0]] = data_dictionary["test_labels"].squeeze()

        predictor = TabularPredictor(label=dk.label_list[0], problem_type="regression")
        predictor = predictor.fit(train, tuning_data=tuning_data, **self.model_training_parameters)
        return predictor
