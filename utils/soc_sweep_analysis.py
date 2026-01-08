from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pickle
import numpy as np


def get_coulombic_efficiency(
    dir_processed: Path,
    channel_id: int,
    cell_id: int,
    soc: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Calculates coulombic efficiency for a given cell over SOC sweep.
    Returns dict: {"SOC": array, "CE": array, "u_CE": array}
    """
    if soc is None:
        socs = np.arange(10, 95, 5)
    else:
        socs = [int(soc)]

    ret = {
        "SOC": np.array(socs, dtype=float),
        "CE": np.zeros(len(socs), dtype=float),
        "u_CE": np.zeros(len(socs), dtype=float),
    }

    pkl_path = Path(dir_processed) / f"socSweep_Channel_{channel_id}_Cell_{cell_id:02d}.pkl"
    df_sweep = pickle.load(open(pkl_path, "rb"))

    for i, s in enumerate(ret["SOC"]):
        # Charge
        df_chg = df_sweep.loc[
            (df_sweep["Protocol Segment"] == "sweep")
            & (df_sweep["Sweep Segment"] == "charge")
            & (df_sweep["Sweep SOC"] == s)
        ]
        if df_chg.empty:
            continue

        t_chg = df_chg["Time (s)"].values[-1] - df_chg["Time (s)"].values[0]
        i_chg = df_chg["Current (A)"].mean()
        q_chg = df_chg["Capacity (Ah)"].max()

        # Discharge
        df_dchg = df_sweep.loc[
            (df_sweep["Protocol Segment"] == "sweep")
            & (df_sweep["Sweep Segment"] == "discharge")
            & (df_sweep["Sweep SOC"] == s)
        ]
        if df_dchg.empty:
            continue

        t_dchg = df_dchg["Time (s)"].values[-1] - df_dchg["Time (s)"].values[0]
        i_dchg = abs(df_dchg["Current (A)"].mean())
        q_dchg = df_dchg["Capacity (Ah)"].max()

        # Uncertainty (faithful to your code: std from df_chg + dt/10)
        u_i_dchg = u_i_chg = np.std(df_chg["Current (A)"].values)
        u_t_dchg = u_t_chg = np.mean(np.diff(df_chg["Time (s)"].values)) / 10

        # Sensitivity + CE uncertainty
        theta_i_dchg = t_dchg / (i_chg * t_chg)
        theta_t_dchg = i_dchg / (i_chg * t_chg)
        theta_i_chg = -i_dchg * t_dchg * t_chg / (i_chg * t_chg) ** 2
        theta_t_chg = -i_dchg * t_dchg * i_chg / (i_chg * t_chg) ** 2

        u_ce = np.sqrt(
            (theta_i_dchg * u_i_dchg) ** 2
            + (theta_i_chg * u_i_chg) ** 2
            + (theta_t_dchg * u_t_dchg) ** 2
            + (theta_t_chg * u_t_chg) ** 2
        )

        ce = q_dchg / q_chg
        ret["CE"][i] = ce
        ret["u_CE"][i] = u_ce

    return ret


def shift_ce_to_100(ce_values: np.ndarray, u_ce_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shifts CE curve so the first positive CE becomes 100%.
    Keeps u_CE unchanged (faithful to your behavior).
    """
    first_valid_ce = None
    for ce in ce_values:
        if ce > 0:
            first_valid_ce = ce
            break
    if first_valid_ce is None:
        return ce_values, u_ce_values
    shift_amount = 100.0 - first_valid_ce
    return ce_values + shift_amount, u_ce_values


def get_default_groups() -> List[Dict[str, Any]]:
    """Groups config (same as you provided)."""
    return [
        {
            "group": "G1",
            "legend_labels": {
                "5-1 and 5-2": "1.5C",
                "5-3 and 5-4": "2C",
                "5-5 and 5-6": "2.5C",
                "5-7 and 5-8": "3C",
                "8-1 and 8-2": "3.5C",
            },
            "channels_cells": {5: [1, 2, 3, 4, 5, 6, 7, 8], 8: [1, 2]},
            "cell_to_legend_label": {
                (5, 1): "5-1 and 5-2", (5, 2): "5-1 and 5-2",
                (5, 3): "5-3 and 5-4", (5, 4): "5-3 and 5-4",
                (5, 5): "5-5 and 5-6", (5, 6): "5-5 and 5-6",
                (5, 7): "5-7 and 5-8", (5, 8): "5-7 and 5-8",
                (8, 1): "8-1 and 8-2", (8, 2): "8-1 and 8-2",
            },
            "title": "Group 1 (SOH = 100%)",
            "SOH": 100,
        },
        {
            "group": "G2",
            "legend_labels": {
                "1-1 and 1-2": "1.5C",
                "1-3 and 1-4": "2C",
                "1-5 and 1-6": "2.5C",
                "1-7 and 1-8": "3C",
                "4-7 and 4-8": "3.5C",
            },
            "channels_cells": {1: [1, 2, 3, 4, 5, 6, 7, 8], 4: [7, 8]},
            "cell_to_legend_label": {
                (1, 1): "1-1 and 1-2", (1, 2): "1-1 and 1-2",
                (1, 3): "1-3 and 1-4", (1, 4): "1-3 and 1-4",
                (1, 5): "1-5 and 1-6", (1, 6): "1-5 and 1-6",
                (1, 7): "1-7 and 1-8", (1, 8): "1-7 and 1-8",
                (4, 7): "4-7 and 4-8", (4, 8): "4-7 and 4-8",
            },
            "title": "Group 2 (SOH = 90%)",
            "SOH": 90,
        },
        {
            "group": "G3",
            "legend_labels": {
                "1-5 and 1-6": "1.5C",
                "1-7 and 1-8": "2C",
                "2-6 and 2-7": "2.5C",
                "3-3 and 3-4": "3C",
                "3-6 and 3-7": "3.5C",
            },
            "channels_cells": {1: [5, 6, 7, 8], 2: [6, 7], 3: [3, 4, 6, 7]},
            "cell_to_legend_label": {
                (1, 5): "1-5 and 1-6", (1, 6): "1-5 and 1-6",
                (1, 7): "1-7 and 1-8", (1, 8): "1-7 and 1-8",
                (2, 6): "2-6 and 2-7", (2, 7): "2-6 and 2-7",
                (3, 3): "3-3 and 3-4", (3, 4): "3-3 and 3-4",
                (3, 6): "3-6 and 3-7", (3, 7): "3-6 and 3-7",
            },
            "title": "Group 3 (SOH = 80%)",
            "SOH": 80,
        },
        {
            "group": "G4",
            "legend_labels": {
                "2-1 and 2-2": "1.5C",
                "2-3 and 2-4": "2C",
                "2-6 and 3-5": "2.5C",
                "4-3 and 4-7": "3C",
                "3-8 and 5-8": "3.5C",
            },
            "channels_cells": {2: [1, 2, 3, 4, 6], 3: [5, 8], 4: [3, 7], 5: [8]},
            "cell_to_legend_label": {
                (2, 1): "2-1 and 2-2", (2, 2): "2-1 and 2-2",
                (2, 3): "2-3 and 2-4", (2, 4): "2-3 and 2-4",
                (2, 6): "2-6 and 3-5", (3, 5): "2-6 and 3-5",
                (4, 3): "4-3 and 4-7", (4, 7): "4-3 and 4-7",
                (3, 8): "3-8 and 5-8", (5, 8): "3-8 and 5-8",
            },
            "title": "Group 4 (SOH = 70%)",
            "SOH": 70,
        },
    ]
