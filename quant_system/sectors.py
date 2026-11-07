"""Static sector map used for concentration controls.

This is a research aid, not point-in-time classification. For production it
must be replaced by a dated industry membership table.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import DATA_DIR
from .data import normalize_symbol


SECTOR_MAP: dict[str, str] = {
    # Banks and financials
    "600000": "金融", "600015": "金融", "600016": "金融", "600036": "金融",
    "601166": "金融", "601288": "金融", "601328": "金融", "601398": "金融",
    "601939": "金融", "601988": "金融", "000001": "金融", "601318": "金融",
    "601601": "金融", "601628": "金融", "601336": "金融", "600030": "金融",
    "600837": "金融", "000776": "金融", "601211": "金融", "600999": "金融",
    "601377": "金融", "601995": "金融", "601066": "金融", "600958": "金融",
    "601788": "金融",
    # Consumer
    "600519": "消费", "000858": "消费", "000568": "消费", "000596": "消费",
    "000333": "消费", "000651": "消费", "600690": "消费", "600887": "消费",
    "000895": "消费", "002304": "消费", "600809": "消费", "603288": "消费",
    "600298": "消费", "601888": "消费",
    # Healthcare
    "600276": "医药", "603259": "医药", "300760": "医药", "600196": "医药",
    "000538": "医药", "000963": "医药", "600085": "医药", "300015": "医药",
    "300347": "医药", "002821": "医药", "600436": "医药",
    # Technology
    "000063": "科技", "601138": "科技", "600050": "科技", "300308": "科技",
    "300502": "科技", "002371": "科技", "603501": "科技", "688981": "科技",
    "688012": "科技", "688111": "科技", "000725": "科技", "002415": "科技",
    "002230": "科技", "002236": "科技", "002475": "科技", "002241": "科技",
    "000100": "科技", "600941": "科技", "601728": "科技", "600406": "科技",
    "300124": "科技", "002050": "科技",
    # Industrials
    "601668": "工业", "600031": "工业", "000157": "工业", "601766": "工业",
    "600150": "工业", "600760": "工业", "000768": "工业", "002179": "工业",
    "600893": "工业", "601390": "工业", "601186": "工业", "601800": "工业",
    "601618": "工业", "600585": "工业", "000786": "工业", "600801": "工业",
    "600019": "工业", "000898": "工业", "600104": "工业", "601633": "工业",
    "000625": "工业", "601238": "工业",
    # Energy and materials
    "600028": "能源", "601857": "能源", "601088": "能源", "600900": "能源",
    "601985": "能源", "600011": "能源", "600795": "能源", "601225": "能源",
    "600188": "能源", "601898": "能源", "600346": "能源", "600426": "能源",
    "600143": "能源", "002493": "能源", "600309": "能源", "601899": "能源",
    "600438": "能源", "601877": "能源", "600089": "能源", "300274": "能源",
    "002129": "能源", "300750": "能源", "601012": "能源", "002594": "工业",
    # Other
    "601919": "运输", "600048": "地产", "000002": "地产", "600383": "地产",
    "000069": "地产", "600606": "地产", "002714": "农业", "002311": "农业",
    "000876": "农业", "002385": "农业",
}


# Delisted names are absent from the hand-maintained map above. Their sector
# assignments are stored separately so the point-in-time experiment can apply
# the same concentration cap to both pools. The file is static (not a dated
# membership table); see reports/point_in_time/README.md.
_DELISTED_PATH = Path(__file__).resolve().parent.parent / "data" / "delisted_sectors.json"


def _load_delisted_map() -> dict[str, str]:
    if not _DELISTED_PATH.exists():
        return {}
    payload = json.loads(_DELISTED_PATH.read_text(encoding="utf-8"))
    return {normalize_symbol(symbol): info["sector"] for symbol, info in payload.items()}


DELISTED_SECTOR_MAP: dict[str, str] = _load_delisted_map()


def sector_of(symbol: str) -> str:
    sym = normalize_symbol(symbol)
    if sym in DELISTED_SECTOR_MAP:
        return DELISTED_SECTOR_MAP[sym]
    return SECTOR_MAP.get(sym[2:], "其他")


def sector_map(symbols: list[str]) -> dict[str, str]:
    return {s: sector_of(s) for s in symbols}
