from copy import deepcopy

import numpy as np

from freqtrade.optimize.backtesting import Backtesting
from tests.conftest import patch_exchange


def test_freqai_autogluon_strategy_backtest(mocker, freqai_conf):
    patch_exchange(mocker)
    freqai_conf["strategy"] = "FreqaiAutoGluonStrategy"
    freqai_conf["freqaimodel"] = "AutoGluonTabularRegressor"
    freqai_conf["timerange"] = "20180110-20180113"
    freqai_conf["freqai"]["train_period_days"] = 1
    freqai_conf["freqai"]["backtest_period_days"] = 1

    fit_called = {}

    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X))

    def dummy_fit(self, data_dictionary, dk, **kwargs):
        fit_called["called"] = True
        return DummyModel()

    mocker.patch(
        "freqtrade.freqai.prediction_models.AutoGluonTabularRegressor."
        "AutoGluonTabularRegressor.fit",
        dummy_fit,
    )

    def dummy_start_backtesting(self, dataframe, metadata, dk, strategy):
        self.model = dummy_fit(self, {}, dk)
        size = len(dataframe)
        dk.return_dataframe = dataframe.assign(
            **{
                "&-s_close": np.where(np.arange(size) % 2 == 0, 0.05, -0.05),
                "&-s_close_mean": 0.0,
                "&-s_close_std": 0.01,
                "do_predict": 1,
            }
        )
        return dk

    mocker.patch(
        "freqtrade.freqai.freqai_interface.IFreqaiModel.start_backtesting",
        dummy_start_backtesting,
    )

    backtesting = Backtesting(deepcopy(freqai_conf))
    data, _ = backtesting.load_bt_data()
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = "ETH/BTC"
    df_ind = backtesting.strategy.advise_indicators(data[pair].copy(), {"pair": pair})
    df = backtesting.strategy.ft_advise_signals(df_ind, {"pair": pair})

    assert fit_called.get("called", False)

    for col in ["enter_long", "exit_long", "enter_short", "exit_short"]:
        assert col in df.columns
        assert df[col].notna().any()


def test_freqai_autogluon_strategy_model_creation(mocker, freqai_conf):
    """Ensure model files are created and signals are generated."""
    patch_exchange(mocker)
    freqai_conf["strategy"] = "FreqaiAutoGluonStrategy"
    freqai_conf["freqaimodel"] = "AutoGluonTabularRegressor"
    freqai_conf["timerange"] = "20180110-20180113"
    freqai_conf["freqai"]["train_period_days"] = 1
    freqai_conf["freqai"]["backtest_period_days"] = 1
    freqai_conf["exchange"]["pair_whitelist"] = ["ETH/BTC"]
    freqai_conf["freqai"]["feature_parameters"]["include_corr_pairlist"] = []

    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X))

    def dummy_fit(self, data_dictionary, dk, **kwargs):
        return DummyModel()

    mocker.patch(
        "freqtrade.freqai.prediction_models.AutoGluonTabularRegressor."
        "AutoGluonTabularRegressor.fit",
        dummy_fit,
    )

    def dummy_start_backtesting(self, dataframe, metadata, dk, strategy):
        pair = metadata["pair"]
        timestamp_model_id = int(dataframe["date"].iloc[-1].timestamp())
        dk.set_paths(pair, timestamp_model_id)
        dk.set_new_model_names(pair, timestamp_model_id)
        dk.training_features_list = ["close"]
        dk.label_list = ["&-s_close"]
        dk.data_dictionary["train_features"] = dataframe[["close"]].copy()
        dk.data_dictionary["train_dates"] = dataframe[["date"]].copy()
        dk.data = {}
        self.model = dummy_fit(self, {}, dk)
        self.dd.save_data(self.model, pair, dk)
        size = len(dataframe)
        dk.return_dataframe = dataframe.assign(
            **{
                "&-s_close": np.where(np.arange(size) % 2 == 0, 0.05, -0.05),
                "&-s_close_mean": 0.0,
                "&-s_close_std": 0.01,
                "do_predict": 1,
            }
        )
        return dk

    mocker.patch(
        "freqtrade.freqai.freqai_interface.IFreqaiModel.start_backtesting",
        dummy_start_backtesting,
    )

    backtesting = Backtesting(deepcopy(freqai_conf))
    data, _ = backtesting.load_bt_data()
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = "ETH/BTC"
    df_ind = backtesting.strategy.advise_indicators(data[pair].copy(), {"pair": pair})
    df = backtesting.strategy.ft_advise_signals(df_ind, {"pair": pair})

    model_files = list(freqai_conf["user_data_dir"].rglob("*_model.joblib"))
    assert model_files

    for col in ["enter_long", "exit_long", "enter_short", "exit_short"]:
        assert col in df.columns
        assert df[col].notna().any()


def test_autogluon_tabular_fi_threshold(mocker, freqai_conf):
    import sys
    import types
    import pandas as pd

    freqai_conf["freqai"]["model_training_parameters"]["fi_threshold"] = 0.05
    freqai_conf["freqai"]["data_split_parameters"]["test_size"] = 0

    fit_calls: list[list[str]] = []

    class DummyPredictor:
        def fit(self, train, tuning_data=None, **kwargs):
            fit_calls.append(list(train.columns))
            return self

        def feature_importance(self, data):
            return pd.DataFrame(
                {"importance": [0.2, 0.01, 0.3]}, index=["f1", "f2", "f3"]
            )

    dummy_module = types.ModuleType("autogluon.tabular")
    dummy_module.TabularPredictor = lambda label, problem_type: DummyPredictor()
    dummy_autogluon = types.ModuleType("autogluon")
    dummy_autogluon.tabular = dummy_module
    mocker.patch.dict(sys.modules, {
        "autogluon": dummy_autogluon,
        "autogluon.tabular": dummy_module,
    })

    from freqtrade.freqai.prediction_models.AutoGluonTabularRegressor import (
        AutoGluonTabularRegressor,
    )
    from types import SimpleNamespace

    model = AutoGluonTabularRegressor(freqai_conf)

    data_dictionary = {
        "train_features": pd.DataFrame(
            {"f1": [1, 2], "f2": [3, 4], "f3": [5, 6]}
        ),
        "train_labels": pd.DataFrame({"label": [1, 0]}),
        "test_features": pd.DataFrame(),
        "test_labels": pd.DataFrame(),
    }
    dk = SimpleNamespace(label_list=["label"], training_features_list=["f1", "f2", "f3"])

    model.fit(data_dictionary, dk)

    assert fit_calls == [["f1", "f2", "f3", "label"], ["f1", "f3", "label"]]
    assert dk.training_features_list == ["f1", "f3"]
