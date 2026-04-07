from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from .geometry import CharacteristicCoordinatesResult
from .canonical import canonicalize_coordinate_chart
from .diagnostics import local_chart_conditions_from_coords, score_chart_candidate


@dataclass
class ChartKD:
    chart: CharacteristicCoordinatesResult
    local_conditions: tuple[sp.Expr, ...]
    score: tuple
    source: str


@dataclass
class ChartAtlasKD:
    charts: tuple[ChartKD, ...]

    def best(self) -> ChartKD | None:
        return self.charts[0] if self.charts else None


def make_chart(vars: Sequence[sp.Symbol], invariants: Sequence[sp.Expr], transverse: Sequence[sp.Expr], method: str, validity_conditions: Sequence[sp.Expr] = ()) -> ChartKD:
    chart = CharacteristicCoordinatesResult(tuple(invariants), tuple(transverse), sp.Integer(0), method, tuple(validity_conditions))
    chart = canonicalize_coordinate_chart(chart, vars)
    local = tuple(chart.validity_conditions) + local_chart_conditions_from_coords(vars, chart.invariants + chart.transverse)
    local = tuple(dict.fromkeys(sp.simplify(c) for c in local))
    score = score_chart_candidate(method, chart.invariants, chart.transverse, chart.jacobian, local)
    return ChartKD(chart=chart, local_conditions=local, score=score, source=method)


def build_chart_atlas(vars: Sequence[sp.Symbol], candidates: Sequence[CharacteristicCoordinatesResult]) -> ChartAtlasKD:
    charts = []
    for cand in candidates:
        charts.append(make_chart(vars, cand.invariants, cand.transverse, cand.method, cand.validity_conditions))
    charts.sort(key=lambda c: c.score)
    return ChartAtlasKD(tuple(charts))
