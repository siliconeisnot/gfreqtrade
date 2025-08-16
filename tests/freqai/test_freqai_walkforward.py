import numpy as np
from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from tests.freqai.conftest import get_patched_freqai_strategy
from tests.conftest import get_patched_exchange


def test_walkforward_performance_aggregation(freqai_conf, mocker):
    freqai_conf["timerange"] = "20180115-20180125"
    freqai_conf["freqai"]["train_period_days"] = 2
    freqai_conf["freqai"]["backtest_period_days"] = 1

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = False
    dk = FreqaiDataKitchen(freqai_conf)
    freqai.dk = dk

    timerange = TimeRange.parse_timerange(dk.full_timerange)
    freqai.dd.load_all_pair_histories(timerange, dk)
    corr_df, base_df = freqai.dd.get_base_and_corr_dataframes(timerange, "ADA/BTC", dk)
    dataframe = dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "ADA/BTC")

    mocker.patch.object(freqai, "train", lambda df, pair, dk, **kw: None)

    def _predict(df, dk, **kw):
        dk.DI_values = np.zeros(len(df))
        return df[dk.label_list].copy(), np.ones(len(df), dtype=int)

    mocker.patch.object(freqai, "predict", _predict)

    freqai.start_backtesting(dataframe, {"pair": "ADA/BTC"}, dk, strategy)

    assert len(dk.training_timeranges) >= 2
    assert len(dk.walkforward_performance) == len(dk.backtesting_timeranges) >= 2
    assert dk.aggregated_performance == 0.0
