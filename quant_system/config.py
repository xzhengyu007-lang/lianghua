"""Configuration objects shared by data, research, and execution code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORT_DIR = ROOT / "reports"


@dataclass(frozen=True)
class CostConfig:
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_bps: float = 5.0
    # Square-root market impact, quoted as basis points at 100% of 20-day ADV:
    #     impact_bps = coefficient * sqrt(shares / ADV)
    # 0 disables the term, which is the default so published results do not
    # silently change. A value around 40-60 reproduces published A-share
    # large-cap impact estimates at typical participation rates.
    impact_coefficient_bps: float = 0.0

    def validate(self) -> None:
        if min(self.commission_rate, self.stamp_duty_rate, self.transfer_fee_rate) < 0:
            raise ValueError("cost rates cannot be negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage cannot be negative")
        if self.impact_coefficient_bps < 0:
            raise ValueError("impact_coefficient_bps cannot be negative")


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    lot_size: int = 100
    max_position_weight: float = 0.25
    max_gross_exposure: float = 0.95
    min_order_value: float = 1_000.0
    execution_price: str = "open"  # next-day open, never the signal close
    benchmark: str = "sh000300"
    costs: CostConfig = field(default_factory=CostConfig)
    # Orders are capped at a share of trailing average daily volume.
    max_adv_participation: float = 0.10
    adv_window: int = 20
    # ADV must use only bars that closed before the order is sent. Keeping the
    # old inclusive window available reproduces pre-fix capacity reports; it
    # must never be the default because it sizes an open order with the volume
    # that order is about to consume.
    adv_includes_today: bool = False
    # Optional portfolio drawdown guard. ``None`` disables it (the default), so
    # existing results are unchanged. When set, a drawdown of this size scales
    # gross exposure to ``drawdown_de_risk`` until the drawdown recovers to
    # ``drawdown_recover * drawdown_stop``. The decision uses only the equity
    # realised up to the previous close, so there is no look-ahead.
    drawdown_stop: float | None = None
    drawdown_de_risk: float = 0.5
    drawdown_recover: float = 0.5

    def validate(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0, 1]")
        if not 0 < self.max_gross_exposure <= 1:
            raise ValueError("max_gross_exposure must be in (0, 1]")
        if self.execution_price != "open":
            raise ValueError("only next-day open execution is supported")
        if not 0 < self.max_adv_participation <= 1:
            raise ValueError("max_adv_participation must be in (0, 1]")
        if self.adv_window < 1:
            raise ValueError("adv_window must be positive")
        if self.drawdown_stop is not None:
            if not 0 < self.drawdown_stop < 1:
                raise ValueError("drawdown_stop must be in (0, 1)")
            if not 0 < self.drawdown_de_risk <= 1:
                raise ValueError("drawdown_de_risk must be in (0, 1]")
            if not 0 < self.drawdown_recover < 1:
                raise ValueError("drawdown_recover must be in (0, 1)")
        self.costs.validate()


@dataclass(frozen=True)
class RiskConfig:
    target_annual_vol: float = 0.18
    vol_lookback: int = 40
    max_drawdown_stop: float = 0.25
    max_daily_turnover: float = 0.35
    cash_buffer: float = 0.02
    # Optional market-breadth de-risking (RESEARCH_LOG 23.58). Off by default:
    # it is a market-timing layer, it only has 78 months of evidence, and its
    # tercile rule was chosen by the researcher. ``breadth_de_risk`` is the
    # scale applied in a lower-tercile month.
    use_breadth_timing: bool = False
    breadth_lookback: int = 120
    breadth_history_months: int = 24
    breadth_de_risk: float = 0.25

    def validate(self) -> None:
        if self.target_annual_vol <= 0:
            raise ValueError("target_annual_vol must be positive")
        if not 0 < self.max_drawdown_stop < 1:
            raise ValueError("max_drawdown_stop must be in (0, 1)")
        if self.breadth_lookback < 20:
            raise ValueError("breadth_lookback must be at least 20")
        if self.breadth_history_months < 2:
            raise ValueError("breadth_history_months must be at least 2")
        if not 0 <= self.breadth_de_risk <= 1:
            raise ValueError("breadth_de_risk must be in [0, 1]")


@dataclass(frozen=True)
class ResearchConfig:
    start: str | None = None
    end: str | None = None
    train_days: int = 252
    test_days: int = 63
    embargo_days: int = 5
    seed: int = 20261105

    def validate(self) -> None:
        if self.train_days < 60 or self.test_days < 10:
            raise ValueError("walk-forward windows are too short")
        if self.embargo_days < 0:
            raise ValueError("embargo_days cannot be negative")


DEFAULT_UNIVERSE: dict[str, str] = {
    "600519": "贵州茅台",
    "000858": "五粮液",
    "601318": "中国平安",
    "601398": "工商银行",
    "601939": "建设银行",
    "600036": "招商银行",
    "600030": "中信证券",
    "601166": "兴业银行",
    "600276": "恒瑞医药",
    "603259": "药明康德",
    "300760": "迈瑞医疗",
    "000333": "美的集团",
    "000651": "格力电器",
    "000568": "泸州老窖",
    "000596": "古井贡酒",
    "600887": "伊利股份",
    "002714": "牧原股份",
    "601888": "中国中免",
    "300750": "宁德时代",
    "002594": "比亚迪",
    "601012": "隆基绿能",
    "600900": "长江电力",
    "601899": "紫金矿业",
    "600309": "万华化学",
    "600028": "中国石化",
    "601857": "中国石油",
    "601668": "中国建筑",
    "600050": "中国联通",
    "002415": "海康威视",
    "000725": "京东方A",
    "601088": "中国神华",
    "600585": "海螺水泥",
    "601919": "中远海控",
    "600690": "海尔智家",
    "000063": "中兴通讯",
    "601138": "工业富联",
}


def config_dict(obj: Any) -> dict[str, Any]:
    """Recursively convert dataclasses to a JSON-serializable dictionary."""
    return asdict(obj)



# Broader pool used for robustness checks. It intentionally includes both
# large and mid-cap names across sectors; it is not a point-in-time index.
BROAD_UNIVERSE: dict[str, str] = {
    **DEFAULT_UNIVERSE,
    "600016": "民生银行", "601288": "农业银行", "601988": "中国银行", "601328": "交通银行",
    "600015": "华夏银行", "600000": "浦发银行", "000001": "平安银行", "000002": "万科A",
    "600048": "保利发展", "600383": "金地集团", "000069": "华侨城A", "600606": "绿地控股",
    "601601": "中国太保", "601628": "中国人寿", "601336": "新华保险", "600837": "海通证券",
    "000776": "广发证券", "601211": "国泰海通", "600999": "招商证券", "601377": "兴业证券",
    "600196": "复星医药", "000538": "云南白药", "000963": "华东医药", "600085": "同仁堂",
    "300015": "爱尔眼科", "300347": "泰格医药", "002821": "凯莱英", "600436": "片仔癀",
    "000895": "双汇发展", "002304": "洋河股份", "600809": "山西汾酒", "603288": "海天味业",
    "600298": "安琪酵母", "002311": "海大集团", "000876": "新希望", "002385": "大北农",
    "600104": "上汽集团", "601633": "长城汽车", "000625": "长安汽车", "601238": "广汽集团",
    "002050": "三花智控", "000100": "TCL科技", "002241": "歌尔股份", "002475": "立讯精密",
    "300124": "汇川技术", "300274": "阳光电源", "002129": "TCL中环", "600438": "通威股份",
    "601877": "正泰电器", "600089": "特变电工", "601985": "中国核电", "600011": "华能国际",
    "600795": "国电电力", "601225": "陕西煤业", "600188": "兖矿能源", "601898": "中煤能源",
    "600346": "恒力石化", "600426": "华鲁恒升", "600143": "金发科技", "002493": "荣盛石化",
    "600019": "宝钢股份", "000898": "鞍钢股份", "600801": "华新水泥", "000786": "北新建材",
    "601390": "中国中铁", "601186": "中国铁建", "601800": "中国交建", "601618": "中国中冶",
    "600031": "三一重工", "000157": "中联重科", "601766": "中国中车", "600150": "中国船舶",
    "600760": "中航沈飞", "000768": "中航西飞", "002179": "中航光电", "600893": "航发动力",
    "600941": "中国移动", "601728": "中国电信", "600406": "国电南瑞", "002230": "科大讯飞",
    "002236": "大华股份", "300308": "中际旭创", "300502": "新易盛", "002371": "北方华创",
    "603501": "韦尔股份", "688981": "中芯国际", "688012": "中微公司", "688111": "金山办公",
    "601995": "中金公司", "601066": "中信建投", "600958": "东方证券", "601788": "光大证券",
}
