from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import OHLCV_COLUMNS, infer_price_limit


PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class AdjustmentSegment:
    """One interval on which ``qfq = slope * raw + intercept``."""

    start: int
    end: int
    slope: float
    intercept: float
    sse: float

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class AdjustmentEvent:
    """A corporate action implied by a breakpoint between two segments."""

    date: pd.Timestamp
    split: float
    cash: float
    raw_price: float
    factor_ratio: float
    segment_before: int
    segment_after: int

    @property
    def cash_yield(self) -> float:
        return self.cash / self.raw_price if self.raw_price else float("nan")


@dataclass
class AdjustmentFit:
    segments: list[AdjustmentSegment]
    events: list[AdjustmentEvent]
    factors: pd.Series
    split_factors: pd.Series
    residuals: pd.Series
    max_abs_residual: float
    max_relative_residual: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def problems(self) -> list[str]:
        return list(self.diagnostics.get("problems", []))

    @property
    def ok(self) -> bool:
        return not self.problems


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(values, dtype=float)])


def _interval_fit(
    s: int,
    e: int,
    sx: np.ndarray,
    sy: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: np.ndarray,
) -> tuple[float, float, float]:
    """Least-squares affine fit and SSE for the half-open interval [s, e)."""
    n = e - s
    if n <= 0:
        return 1.0, 0.0, 0.0
    x_sum = sx[e] - sx[s]
    y_sum = sy[e] - sy[s]
    xx_sum = sxx[e] - sxx[s]
    xy_sum = sxy[e] - sxy[s]
    yy_sum = syy[e] - syy[s]
    denominator = n * xx_sum - x_sum * x_sum
    if denominator <= 1e-12 * max(1.0, abs(n * xx_sum)):
        slope = 1.0
        intercept = y_sum / n - slope * x_sum / n
    else:
        slope = (n * xy_sum - x_sum * y_sum) / denominator
        intercept = (y_sum - slope * x_sum) / n
    sse = (
        yy_sum
        - 2.0 * slope * xy_sum
        - 2.0 * intercept * y_sum
        + slope * slope * xx_sum
        + 2.0 * slope * intercept * x_sum
        + n * intercept * intercept
    )
    return float(slope), float(intercept), float(max(0.0, sse))


def _merge_short_spans(
    spans: list[tuple[int, int]],
    sx: np.ndarray,
    sy: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: np.ndarray,
    *,
    min_span: int,
) -> list[tuple[int, int]]:
    """Merge spans too short to be a real corporate-action interval.

    Greedy splitting can carve out a two- or three-bar sliver around the exact
    event date.  Such a sliver has no economic meaning, and it blocks the
    boundary refinement from snapping to the true date.  Merge it into the
    neighbour whose combined SSE increases least.
    """
    spans = list(spans)
    while len(spans) > 1:
        short = [i for i, (start, end) in enumerate(spans) if end - start < min_span]
        if not short:
            break
        position = min(short, key=lambda i: spans[i][1] - spans[i][0])
        candidates: list[tuple[float, int]] = []
        if position > 0:
            combined = _interval_fit(
                spans[position - 1][0], spans[position][1], sx, sy, sxx, sxy, syy
            )[2]
            left = _interval_fit(
                *spans[position - 1], sx, sy, sxx, sxy, syy
            )[2]
            right = _interval_fit(*spans[position], sx, sy, sxx, sxy, syy)[2]
            candidates.append((combined - left - right, position - 1))
        if position < len(spans) - 1:
            combined = _interval_fit(
                spans[position][0], spans[position + 1][1], sx, sy, sxx, sxy, syy
            )[2]
            left = _interval_fit(*spans[position], sx, sy, sxx, sxy, syy)[2]
            right = _interval_fit(
                *spans[position + 1], sx, sy, sxx, sxy, syy
            )[2]
            candidates.append((combined - left - right, position))
        if not candidates:
            break
        _, merge_at = min(candidates)
        spans = _merge_span(spans, merge_at)
    return spans


def _refine_breakpoints(
    spans: list[tuple[int, int]],
    x: np.ndarray,
    y: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: np.ndarray,
    min_segment: int,
    window: int = 40,
) -> list[tuple[int, int]]:
    """Move each internal boundary to the SSE minimum in a local window.

    A greedy split only needs to be near the true event date; the exact bar
    matters because a one-bar offset shifts the dividend into the wrong day
    and creates a large residual on the day before.  Searching locally (with
    the neighbouring boundaries fixed) is both fast and immune to the
    degenerate solution where a boundary drifts across the whole series.
    """
    if len(spans) < 2:
        return spans
    boundaries = [end for _, end in spans[:-1]]
    n = len(x)
    for _ in range(3):
        moved = False
        for position, boundary in enumerate(boundaries):
            left = 0 if position == 0 else boundaries[position - 1]
            right = n if position == len(boundaries) - 1 else boundaries[position + 1]
            low = max(left + min_segment, boundary - window)
            high = min(right - min_segment, boundary + window)
            if high <= low:
                continue
            best_at = boundary
            best_sse = float("inf")
            for candidate in range(low, high + 1):
                _, _, left_sse = _interval_fit(left, candidate, sx, sy, sxx, sxy, syy)
                _, _, right_sse = _interval_fit(
                    candidate, right, sx, sy, sxx, sxy, syy
                )
                total = left_sse + right_sse
                if total < best_sse - 1e-12:
                    best_sse = total
                    best_at = candidate
            if best_at != boundary:
                boundaries[position] = best_at
                moved = True
        if not moved:
            break
    edges = [0] + boundaries + [n]
    out: list[tuple[int, int]] = []
    for start_at, end_at in zip(edges[:-1], edges[1:]):
        if end_at - start_at < min_segment:
            continue
        out.append((start_at, end_at))
    if not out:
        return [(0, n)]
    out[0] = (0, out[0][1])
    out[-1] = (out[-1][0], n)
    return out


def _prune_spans(
    spans: list[tuple[int, int]],
    sx: np.ndarray,
    sy: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: np.ndarray,
    *,
    threshold: float,
    min_segment: int,
) -> list[tuple[int, int]]:
    """Remove statistically meaningless breakpoints, keeping genuine events."""
    spans = list(spans)
    while len(spans) > 1:
        fits = [
            _interval_fit(start, end, sx, sy, sxx, sxy, syy) for start, end in spans
        ]
        best_gain = float("inf")
        best_position = -1
        for position in range(len(spans) - 1):
            combined_sse = _interval_fit(
                spans[position][0],
                spans[position + 1][1],
                sx,
                sy,
                sxx,
                sxy,
                syy,
            )[2]
            gain = combined_sse - fits[position][2] - fits[position + 1][2]
            if gain < best_gain:
                best_gain = gain
                best_position = position
        if best_gain >= 0.5 * threshold:
            break
        position = best_position
        left, right = spans[position], spans[position + 1]
        if right[1] - left[0] < min_segment:
            spans[position : position + 2] = [(left[0], right[1])]
        else:
            spans[position : position + 2] = [(left[0], right[1])]
    return spans



def _merge_span(spans: list[tuple[int, int]], position: int) -> list[tuple[int, int]]:
    left, right = spans[position], spans[position + 1]
    merged = (left[0], right[1])
    return spans[:position] + [merged] + spans[position + 2 :]


def _clean_spans(
    spans: list[tuple[int, int]],
    x: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: np.ndarray,
    *,
    min_segment: int,
    diagnostics: dict,
) -> list[tuple[int, int]]:
    """Drop boundaries that cannot be real corporate actions.

    Two failure modes survive a plain SSE search: (a) the provider's page
    splice creates a fake level shift in the first few bars, and (b) a spurious
    extra breakpoint makes one segment's slope negative.  Both would inject a
    huge fake return, so they are merged away rather than silently used.
    """
    spans = list(spans)
    n = len(x)
    ignored_early = 0
    ignored_implausible = 0

    def stats(position: int) -> tuple[float, float, float, float]:
        left = _interval_fit(spans[position][0], spans[position][1], sx, sy, sxx, sxy, syy)
        right = _interval_fit(
            spans[position + 1][0], spans[position + 1][1], sx, sy, sxx, sxy, syy
        )
        split = right[0] / left[0] if left[0] else float("nan")
        cash = split * (right[1] - left[1]) if np.isfinite(split) else float("nan")
        raw_price = float(x[spans[position + 1][0]])
        return split, cash, raw_price, left[2] + right[2]

    def plausible(split: float, cash: float, raw_price: float) -> bool:
        if not np.isfinite(split) or not np.isfinite(cash) or raw_price <= 0:
            return False
        if split < 0.2 or split > 5.0:
            return False
        if abs(cash) / raw_price > 0.5:
            return False
        return True

    changed = True
    while changed and len(spans) > 1:
        changed = False
        for position in range(len(spans) - 1):
            if spans[position][1] <= min(3, n // 6):
                spans = _merge_span(spans, position)
                ignored_early += 1
                changed = True
                break
        if changed:
            continue
        worst = None
        for position in range(len(spans) - 1):
            split, cash, raw_price, _ = stats(position)
            if not plausible(split, cash, raw_price):
                combined = _interval_fit(
                    spans[position][0], spans[position + 1][1], sx, sy, sxx, sxy, syy
                )[2]
                left_sse = _interval_fit(
                    spans[position][0], spans[position][1], sx, sy, sxx, sxy, syy
                )[2]
                right_sse = _interval_fit(
                    spans[position + 1][0], spans[position + 1][1], sx, sy, sxx, sxy, syy
                )[2]
                gain = combined - left_sse - right_sse
                if worst is None or gain < worst[0]:
                    worst = (gain, position)
        if worst is not None:
            spans = _merge_span(spans, worst[1])
            ignored_implausible += 1
            changed = True

    diagnostics["ignored_early_boundaries"] = ignored_early
    diagnostics["ignored_implausible_boundaries"] = ignored_implausible
    return spans


def fit_piecewise_affine(
    raw: pd.Series,
    qfq: pd.Series,
    *,
    min_segment: int = 3,
    max_segments: int = 80,
    gain_floor: float | None = None,
) -> AdjustmentFit:
    """Fit ``qfq = a * raw + b`` with piecewise-constant ``(a, b)``.

    Tencent's qfq series is an additive adjustment applied after share-split
    rescaling: within a corporate-action segment it is exactly affine in the
    raw traded price.  Breakpoints are share splits/bonus issues (``a`` moves)
    and cash dividends or rights issues (``b`` moves).  A dynamic-programming
    split search with prefix sums keeps this O(n * segments) per iteration.
    """
    raw = raw.astype(float).sort_index()
    qfq = qfq.astype(float).sort_index()
    common = raw.index.intersection(qfq.index)
    min_segment = max(1, min(min_segment, len(common) // 4))
    # A 1800-bar history has at most a couple of dozen corporate actions; a
    # short fixture cannot support one breakpoint per 20 bars without simply
    # interpolating noise.
    max_segments = max(2, min(max_segments, len(common) // 20))
    if len(common) < 2 * min_segment:
        raise ValueError(
            f"need at least {2 * min_segment} aligned bars, got {len(common)}"
        )
    x = raw.loc[common].to_numpy(dtype=float)
    y = qfq.loc[common].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if valid.sum() < 2 * min_segment:
        raise ValueError("too few valid aligned prices")
    index = common[valid]
    x = x[valid]
    y = y[valid]
    n = len(x)

    sx = _prefix(x)
    sy = _prefix(y)
    sxx = _prefix(x * x)
    sxy = _prefix(x * y)
    syy = _prefix(y * y)

    scale = max(float(np.median(np.abs(y))), 1e-6)
    # Provider prices are rounded to cents, so a genuine breakpoint only has to
    # beat the rounding noise.  Using a residual-driven threshold adapts to
    # both short test fixtures and 1800-bar histories.
    noise = max(2e-4 * scale, 0.002)
    threshold = (
        float(gain_floor)
        if gain_floor is not None
        else max(1e-12, 2.0 * n * noise * noise)
    )

    spans: list[tuple[int, int]] = [(0, n)]
    while len(spans) < max_segments:
        best_gain = 0.0
        best: tuple[int, int] | None = None
        for position, (start, end) in enumerate(spans):
            if end - start < 2 * min_segment:
                continue
            _, _, parent_sse = _interval_fit(start, end, sx, sy, sxx, sxy, syy)
            for split_at in range(start + min_segment, end - min_segment + 1):
                _, _, left_sse = _interval_fit(
                    start, split_at, sx, sy, sxx, sxy, syy
                )
                _, _, right_sse = _interval_fit(
                    split_at, end, sx, sy, sxx, sxy, syy
                )
                gain = parent_sse - left_sse - right_sse
                if gain > best_gain:
                    best_gain = gain
                    best = (position, split_at)
        if best is None or best_gain < threshold:
            break
        position, split_at = best
        start, end = spans[position]
        spans[position : position + 1] = [(start, split_at), (split_at, end)]

    spans.sort()
    spans = _merge_short_spans(
        spans,
        sx,
        sy,
        sxx,
        sxy,
        syy,
        min_span=min(max(2 * min_segment, 6), max(2, n // 6)),
    )
    spans = _refine_breakpoints(spans, x, y, sx, sy, sxx, sxy, syy, min_segment)
    spans = _prune_spans(
        spans, sx, sy, sxx, sxy, syy, threshold=threshold, min_segment=min_segment
    )
    spans = _refine_breakpoints(spans, x, y, sx, sy, sxx, sxy, syy, min_segment)
    early: dict = {}
    spans = _clean_spans(
        spans,
        x,
        sx,
        sy,
        sxx,
        sxy,
        syy,
        min_segment=min_segment,
        diagnostics=early,
    )
    segments = [
        AdjustmentSegment(
            start=start,
            end=end,
            slope=slope,
            intercept=intercept,
            sse=sse,
        )
        for start, end in spans
        for slope, intercept, sse in [
            _interval_fit(start, end, sx, sy, sxx, sxy, syy)
        ]
    ]

    events: list[AdjustmentEvent] = []
    for position in range(1, len(segments)):
        before = segments[position - 1]
        after = segments[position]
        split = after.slope / before.slope if before.slope else float("nan")
        # ``qfq = a * raw + b`` is anchored at the latest bar, so moving
        # forward in time the split ratio S applies to the *new* raw price and
        # the cash term is S * (b_new - b_old).  The total-return factor for
        # the older segment is raw / (S * raw + cash) evaluated at the event's
        # own raw price.  A bonus issue (S > 1) therefore *reduces* the older
        # factor, which is the opposite direction from a cash dividend.
        cash = split * (after.intercept - before.intercept)
        raw_price = float(x[after.start])
        denominator = split * raw_price + cash
        ratio = raw_price / denominator if denominator > 0 else float("nan")
        events.append(
            AdjustmentEvent(
                date=index[after.start],
                split=float(split),
                cash=float(cash),
                raw_price=raw_price,
                factor_ratio=float(ratio),
                segment_before=position - 1,
                segment_after=position,
            )
        )

    factors = np.ones(n, dtype=float)
    split_factors = np.ones(n, dtype=float)
    running = 1.0
    for position in range(len(segments) - 1, -1, -1):
        segment = segments[position]
        factors[segment.start : segment.end] = running
        split_factors[segment.start : segment.end] = (
            1.0 / segment.slope if segment.slope else float("nan")
        )
        if position > 0:
            running *= events[position - 1].factor_ratio

    residuals = np.empty(n, dtype=float)
    for segment in segments:
        fitted = segment.slope * x[segment.start : segment.end] + segment.intercept
        residuals[segment.start : segment.end] = (
            y[segment.start : segment.end] - fitted
        )

    max_abs = float(np.nanmax(np.abs(residuals))) if n else 0.0
    # The first bar or two can differ between the qfq and raw downloads when
    # their page anchors were fetched in different sessions.  Those bars are
    # still traded prices, so they are kept; the residual is reported
    # separately instead of failing the whole symbol.
    edge = min(5, n)
    interior = np.abs(residuals[edge:]) if n > edge else np.array([])
    max_abs_interior = float(np.nanmax(interior)) if interior.size else 0.0
    max_rel = max_abs_interior / scale if scale else float("inf")
    diagnostics_extra = {
        "max_abs_residual_edge": max_abs,
        "max_abs_residual_interior": max_abs_interior,
    }
    problems: list[str] = []
    if not np.isfinite(max_abs):
        problems.append("non-finite adjustment residual")
    if max_rel > 0.005:
        problems.append(
            f"piecewise-affine fit is poor: max relative residual {max_rel:.4%}"
        )
    for event in events:
        if not np.isfinite(event.split) or event.split <= 0:
            problems.append(f"{event.date.date()}: invalid split {event.split!r}")
            continue
        if event.split < 0.1 or event.split > 10.0:
            problems.append(
                f"{event.date.date()}: implausible split factor {event.split:.4f}"
            )
        if not np.isfinite(event.cash):
            problems.append(f"{event.date.date()}: non-finite cash flow")
            continue
        if abs(event.cash) / event.raw_price > 0.5:
            problems.append(
                f"{event.date.date()}: implausible cash flow "
                f"{event.cash:.4f} vs price {event.raw_price:.4f}"
            )
        if not np.isfinite(event.factor_ratio) or event.factor_ratio <= 0:
            problems.append(
                f"{event.date.date()}: invalid total-return factor ratio"
            )

    diagnostics = {
        **early,
        **diagnostics_extra,
        "bars": int(n),
        "segments": len(segments),
        "events": len(events),
        "gain_threshold": threshold,
        "max_abs_residual": max_abs,
        "max_relative_residual": max_rel,
        "problems": problems,
    }
    return AdjustmentFit(
        segments=segments,
        events=events,
        factors=pd.Series(factors, index=index, name="factor"),
        split_factors=pd.Series(split_factors, index=index, name="split_factor"),
        residuals=pd.Series(residuals, index=index, name="residual"),
        max_abs_residual=max_abs,
        max_relative_residual=max_rel,
        diagnostics=diagnostics,
    )


def rebuild_frame(
    raw: pd.DataFrame,
    qfq: pd.DataFrame,
    *,
    symbol: str | None = None,
    min_segment: int = 3,
    max_segments: int = 80,
) -> tuple[pd.DataFrame, AdjustmentFit]:
    """Rebuild a multiplicative total-return OHLCV frame from raw and qfq.

    The returned frame keeps the raw index and raw volume units, but prices are
    multiplied by the cumulative total-return factor and volume is rescaled to
    current-share units.  The last bar is the anchor (factor = 1).
    """
    raw = raw.sort_index()
    qfq = qfq.sort_index()
    if "close" not in raw or "close" not in qfq:
        raise ValueError("both frames need a close column")
    fit = fit_piecewise_affine(
        raw["close"],
        qfq["close"],
        min_segment=min_segment,
        max_segments=max_segments,
    )
    # ``fit`` is computed on the intersection of both indices.  Bars that exist
    # only in ``raw`` are usually the newest session (the qfq download lagged by
    # one day); carry the nearest known factor across them.  A leading gap
    # cannot be filled causally, so it is an explicit error.
    union = raw.index.union(fit.factors.index)
    factors = fit.factors.reindex(union).ffill()
    split_factors = fit.split_factors.reindex(union).ffill()
    factors = factors.reindex(raw.index)
    split_factors = split_factors.reindex(raw.index)
    leading_gap = int((factors.isna() | split_factors.isna()).sum())
    leading_problem: str | None = None
    if leading_gap:
        # A handful of delisted names have a raw bar one session before the
        # provider's qfq history starts.  The factor is unknowable there, so
        # the bar is dropped instead of aborting the whole universe; the count
        # is recorded in diagnostics so the loss is auditable.
        keep = factors.notna() & split_factors.notna()
        raw = raw.loc[keep]
        factors = factors.loc[keep]
        split_factors = split_factors.loc[keep]
        fit.diagnostics["leading_bars_dropped"] = leading_gap
        leading_problem = (
            f"{leading_gap} leading raw bars dropped: "
            "no adjustment factor before the first qfq bar"
        )

    adjusted = raw.loc[:, [c for c in OHLCV_COLUMNS if c in raw]].copy()
    for column in PRICE_COLUMNS:
        if column in adjusted:
            adjusted[column] = adjusted[column] * factors
    if "volume" in adjusted:
        adjusted["volume"] = adjusted["volume"] * split_factors

    problems = list(fit.problems)
    if leading_problem:
        problems.append(leading_problem)
    if (adjusted.loc[:, [c for c in PRICE_COLUMNS if c in adjusted]] <= 0).any().any():
        problems.append("rebuilt prices contain non-positive values")
    returns = adjusted["close"].pct_change()
    limit = infer_price_limit(symbol) if symbol else 0.10
    # Dividends and rights issues can legitimately push the total return past
    # the exchange price limit, so allow a generous event cushion.
    breaches = returns.abs() > limit + 0.10
    if int(breaches.sum()):
        worst = float(returns.abs().max())
        problems.append(
            f"{int(breaches.sum())} rebuilt daily returns exceed "
            f"{limit + 0.10:.0%} (max {worst:.2%})"
        )
    fit.diagnostics["problems"] = problems
    fit.diagnostics["limit_breaches"] = int(breaches.sum())
    fit.diagnostics["max_abs_return"] = (
        float(returns.abs().max()) if returns.notna().any() else float("nan")
    )
    return adjusted, fit


def rebuild_panel(
    symbols,
    cache_dir,
    raw_dir,
    *,
    min_segment: int = 3,
    max_segments: int = 80,
) -> tuple[dict[str, pd.DataFrame], dict[str, AdjustmentFit]]:
    """Rebuild every symbol that has both a qfq and a raw parquet file."""
    from pathlib import Path

    from .data import normalize_symbol

    cache_dir = Path(cache_dir)
    raw_dir = Path(raw_dir)
    panel: dict[str, pd.DataFrame] = {}
    fits: dict[str, AdjustmentFit] = {}
    for symbol in symbols:
        sym = normalize_symbol(symbol)
        raw_path = raw_dir / f"{sym}_raw.parquet"
        qfq_path = cache_dir / f"{sym}_qfq.parquet"
        if not raw_path.exists() or not qfq_path.exists():
            continue
        raw = pd.read_parquet(raw_path)
        qfq = pd.read_parquet(qfq_path)
        raw.index = pd.to_datetime(raw.index)
        qfq.index = pd.to_datetime(qfq.index)
        adjusted, fit = rebuild_frame(
            raw, qfq, symbol=sym, min_segment=min_segment, max_segments=max_segments
        )
        panel[sym] = adjusted
        fits[sym] = fit
    return panel, fits
