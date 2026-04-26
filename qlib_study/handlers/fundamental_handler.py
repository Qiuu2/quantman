"""
Alpha158Enhanced — extend Qlib's Alpha158 with fundamental + sentiment + macro factors.

Learning goal: show how Qlib's expression-based feature engine composes with
custom fields. Alpha158 covers 158 price/volume factors; we add 18 more from
enhanced_factors (roe/margin/eps_surprise/term_spread/...).

Prerequisite: the custom fields must already exist in the qlib bin directory
(see qlib_study/scripts/export_fundamentals_to_qlib.py). Qlib expressions
reference bin fields with a leading $, same as $close/$volume.
"""

from __future__ import annotations

from qlib.contrib.data.handler import Alpha158, check_transform_proc
from qlib.data.dataset.handler import DataHandlerLP


_DEFAULT_INFER_PROCESSORS = [
    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
]

_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
]


def _fundamental_fields() -> tuple[list[str], list[str]]:
    """
    Fundamental + sentiment + macro expressions.

    Each entry maps a Qlib expression → short factor name. Alpha158 will merge
    these into the feature matrix alongside its 158 built-ins, and downstream
    processors (RobustZScoreNorm, CSRankNorm) treat them uniformly.
    """
    fields, names = [], []

    # --- raw fundamentals (levels) ---
    for f in ["roe", "roa", "gross_margin", "operating_margin",
              "net_margin", "debt_to_equity", "current_ratio"]:
        fields.append(f"${f}")
        names.append(f.upper())

    # --- valuation (inverse → higher is cheaper, matches your value factor) ---
    fields.append("1 / ($pe + 1e-12)")
    names.append("EP")
    fields.append("1 / ($pb + 1e-12)")
    names.append("BP")
    fields.append("1 / ($ps + 1e-12)")
    names.append("SP")

    # --- trends (fundamental momentum) ---
    # Ref($x, 63) ≈ ~one quarter ago on daily bars; diff captures QoQ dynamics
    fields.append("$roe - Ref($roe, 63)")
    names.append("ROE_TREND")
    fields.append("$gross_margin - Ref($gross_margin, 63)")
    names.append("GM_TREND")
    fields.append("$eps / (Ref($eps, 63) + 1e-12) - 1")
    names.append("EPS_GROWTH_Q")

    # --- analyst signals ---
    fields.append("$eps_surprise")
    names.append("EPS_SURPRISE")
    fields.append("$analyst_consensus")
    names.append("CONSENSUS")
    fields.append("$target_upside")
    names.append("TARGET_UPSIDE")

    # --- macro (same value across instruments; GBDT still finds regime splits) ---
    fields.append("$term_spread")
    names.append("TERM_SPREAD")
    fields.append("$macro_regime")
    names.append("MACRO_REGIME")

    return fields, names


class Alpha158Enhanced(Alpha158):
    """
    Alpha158 + fundamental/sentiment/macro factors from enhanced_factors.

    Yaml usage:
        handler:
            class: Alpha158Enhanced
            module_path: qlib_study.handlers.fundamental_handler
            kwargs:
                start_time: 2018-01-01
                end_time: 2024-12-31
                fit_start_time: 2018-01-01
                fit_end_time: 2022-12-31
                instruments: sp500
    """

    def __init__(
        self,
        instruments: str = "sp500",
        start_time: str | None = None,
        end_time: str | None = None,
        freq: str = "day",
        infer_processors: list | None = None,
        learn_processors: list | None = None,
        fit_start_time: str | None = None,
        fit_end_time: str | None = None,
        process_type: str = DataHandlerLP.PTYPE_A,
        **kwargs,
    ):
        infer_processors = check_transform_proc(
            infer_processors or _DEFAULT_INFER_PROCESSORS,
            fit_start_time, fit_end_time,
        )
        learn_processors = check_transform_proc(
            learn_processors or _DEFAULT_LEARN_PROCESSORS,
            fit_start_time, fit_end_time,
        )

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self) -> tuple[list[str], list[str]]:
        """Alpha158's 158 + our fundamental expressions, merged."""
        base_fields, base_names = super().get_feature_config()
        extra_fields, extra_names = _fundamental_fields()
        return base_fields + extra_fields, base_names + extra_names
