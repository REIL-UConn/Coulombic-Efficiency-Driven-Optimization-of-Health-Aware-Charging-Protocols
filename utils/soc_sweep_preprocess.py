from __future__ import annotations

from pathlib import Path
import re
import pickle
import warnings

import numpy as np
import pandas as pd


REPO_NAME_DEFAULT = "Coulombic-Efficiency-Driven-Optimization-of-Health-Aware-Charging-Protocols"


# ---------------------------- generic helpers ---------------------------- #

def find_repo_root(repo_name: str = REPO_NAME_DEFAULT, start: Path | None = None) -> Path:
    """Walk upward from `start` (or CWD) until a folder named `repo_name` is found."""
    start = Path.cwd() if start is None else Path(start).resolve()
    for p in [start, *start.parents]:
        if p.name == repo_name and p.is_dir():
            return p
    raise FileNotFoundError(f"Could not locate repo root '{repo_name}' from {start}")


def get_neware_data_header_keys(rpt_data: pd.DataFrame) -> tuple[str, float, str, float, str, float]:
    """Detect voltage/current/capacity column headers and unit modifiers in Neware details sheet."""
    keys = list(rpt_data.keys())

    v_key = keys[np.argwhere(np.char.find(keys, "Voltage") == 0)[0][0]]
    v_modifier = 1.0 if (v_key.rfind("mV") == -1) else (1.0 / 1000.0)

    i_key = keys[np.argwhere(np.char.find(keys, "Current") == 0)[0][0]]
    i_modifier = 1.0 if (i_key.rfind("mA") == -1) else (1.0 / 1000.0)

    q_key = keys[np.argwhere(np.char.find(keys, "Capacity") == 0)[0][0]]
    q_modifier = 1.0 if (q_key.rfind("mAh") == -1) else (1.0 / 1000.0)

    return v_key, v_modifier, i_key, i_modifier, q_key, q_modifier


def convert_rpt_rel_time_to_float(ts) -> np.ndarray:
    """Convert 'Relative Time(h:min:s.ms)' strings to continuous time (seconds)."""
    t_seconds = np.zeros_like(ts, dtype=float)
    t_deltas = np.zeros_like(ts, dtype=float)
    t_continuous = np.zeros_like(ts, dtype=float)

    for i, time_str in enumerate(ts):
        temp = sum(s * float(t) for s, t in zip([1, 60, 3600], reversed(time_str.split(":"))))
        t_seconds[i] = temp
        if i == 0:
            t_deltas[i] = temp
        else:
            t_deltas[i] = temp if (temp < t_seconds[i - 1]) else (temp - t_seconds[i - 1])

    for i in range(len(t_deltas)):
        t_continuous[i] = t_deltas[i] if i == 0 else (t_continuous[i - 1] + t_deltas[i])

    return t_continuous


# ---------------------------- SOC sweep mapping ---------------------------- #

def get_socSweep_mapping() -> pd.DataFrame:
    """Map original protocol steps to {'Protocol Segment', 'Sweep SOC', 'Sweep Segment'} keys."""
    df_mapping = pd.DataFrame(
        columns=["Step", "Protocol Segment", "Sweep SOC", "Sweep Segment"],
        index=np.arange(0, 17.5, 1),
    )

    df_mapping["Step"] = np.arange(1, 18.5, 1).astype(int)
    df_mapping["Protocol Segment"] = "-"
    df_mapping["Sweep SOC"] = np.nan
    df_mapping["Sweep Segment"] = "-"

    df_mapping.loc[df_mapping["Step"].isin(np.arange(3, 6.5, 1)), "Protocol Segment"] = "reference"
    df_mapping.loc[df_mapping["Step"].isin(np.arange(7, 11.5, 1)), "Protocol Segment"] = "conditioning"
    df_mapping.loc[df_mapping["Step"].isin(np.arange(12, 16.5, 1)), "Protocol Segment"] = "sweep"

    df_mapping.loc[df_mapping["Protocol Segment"] == "conditioning", "Sweep SOC"] = 10
    df_mapping.loc[(df_mapping["Protocol Segment"] == "conditioning") & (df_mapping["Step"] == 7), "Sweep Segment"] = "charge"
    df_mapping.loc[(df_mapping["Protocol Segment"] == "conditioning") & (df_mapping["Step"] == 8), "Sweep Segment"] = "rest1"
    df_mapping.loc[(df_mapping["Protocol Segment"] == "conditioning") & (df_mapping["Step"] == 9), "Sweep Segment"] = "discharge"
    df_mapping.loc[(df_mapping["Protocol Segment"] == "conditioning") & (df_mapping["Step"] == 10), "Sweep Segment"] = "rest2"

    df_mapping.loc[df_mapping["Protocol Segment"] == "sweep", "Sweep SOC"] = np.nan
    df_mapping.loc[(df_mapping["Protocol Segment"] == "sweep") & (df_mapping["Step"] == 12), "Sweep Segment"] = "charge"
    df_mapping.loc[(df_mapping["Protocol Segment"] == "sweep") & (df_mapping["Step"] == 13), "Sweep Segment"] = "rest1"
    df_mapping.loc[(df_mapping["Protocol Segment"] == "sweep") & (df_mapping["Step"] == 14), "Sweep Segment"] = "discharge"
    df_mapping.loc[(df_mapping["Protocol Segment"] == "sweep") & (df_mapping["Step"] == 15), "Sweep Segment"] = "rest2"

    return df_mapping


# ---------------------------- main preprocessing ---------------------------- #

def process_raw_socSweep_data(
    dir_raw: Path,
    dir_save: Path,
    overwrite_existing: bool = False,
    *,
    socs_tested: np.ndarray | None = None,
    n_conditioning_cycles: int = 3,
) -> None:
    """
    Processes raw SOC sweep .xlsx files into one dataframe per cell, saved as pickle.

    Expects:
      dir_raw/Group*/<...>-<channel_id>-<cell_id>-<...>.xlsx

    Saves:
      socSweep_Channel_{channel_id}_Cell_{cell_id:02d}.pkl
    """
    dir_raw = Path(dir_raw)
    dir_save = Path(dir_save)
    dir_save.mkdir(exist_ok=True, parents=True)

    if socs_tested is None:
        socs_tested = np.arange(10, 91, 5)

    sweep_mapping = get_socSweep_mapping()

    for dir_group in sorted(dir_raw.glob("Group*")):
        if not dir_group.is_dir():
            continue

        group_id = int(str(dir_group.name)[str(dir_group.name).rindex(" ") + 1 :])

        for file_cell_sweep in sorted(dir_group.glob("*.xlsx")):
            if not file_cell_sweep.is_file():
                continue

            match = re.match(r".*-(\d+)-(\d+)-.*", file_cell_sweep.stem)
            if not match:
                print(f"Skipping file {file_cell_sweep.name}: filename pattern mismatch.")
                continue

            channel_id = int(match.group(1))
            cell_id = int(match.group(2))

            filename = dir_save / f"socSweep_Channel_{channel_id}_Cell_{cell_id:02d}.pkl"
            if (not overwrite_existing) and filename.exists():
                print(f"File exists for Channel {channel_id}, Cell {cell_id}. Skipping...")
                continue

            print(f"Processing SOC sweep | Group {group_id} | Channel {channel_id} | Cell {cell_id}")

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    module=re.escape("openpyxl.styles.stylesheet"),
                )
                sheets = pd.read_excel(file_cell_sweep, sheet_name=[0, 2, 3], engine="openpyxl")

            df_stats = sheets[2]
            df_details = sheets[3]

            v_key, v_mod, i_key, i_mod, q_key, q_mod = get_neware_data_header_keys(df_details)

            df_sweep = pd.DataFrame(
                columns=[
                    "Date (yyyy.mm.dd hh.mm.ss)",
                    "Protocol Segment",
                    "Sweep SOC",
                    "Sweep Segment",
                    "Record Number",
                    "State",
                    "Voltage (V)",
                    "Current (A)",
                    "Capacity (Ah)",
                    "Time (s)",
                ],
                index=np.arange(0, len(df_details), 1),
            )

            df_sweep["Date (yyyy.mm.dd hh.mm.ss)"] = pd.to_datetime(
                df_details["Date(h:min:s.ms)"].values,
                format="%Y-%m-%d %H:%M:%S",
            )
            df_sweep["Protocol Segment"] = "-"
            df_sweep["Sweep SOC"] = np.nan
            df_sweep["Sweep Segment"] = "-"

            df_sweep["Record Number"] = df_details["Record number"]
            df_sweep["State"] = df_details["State"]
            df_sweep["Voltage (V)"] = df_details[v_key] * v_mod
            df_sweep["Current (A)"] = df_details[i_key] * i_mod
            df_sweep["Capacity (Ah)"] = df_details[q_key] * q_mod
            df_sweep["Time (s)"] = convert_rpt_rel_time_to_float(df_details["Relative Time(h:min:s.ms)"].values)

            # Protocol Segment
            for protocol_segment_key in sweep_mapping["Protocol Segment"].unique():
                orig_step_nums = sweep_mapping.loc[sweep_mapping["Protocol Segment"] == protocol_segment_key, "Step"].values
                true_step_nums = df_stats.loc[df_stats["Original step"].isin(orig_step_nums), "Steps"].values
                record_nums = df_details.loc[df_details["Steps"].isin(true_step_nums), "Record number"].values
                df_sweep.loc[df_sweep["Record Number"].isin(record_nums), "Protocol Segment"] = protocol_segment_key

            # Sweep SOC: conditioning cycles
            orig_step_nums = sweep_mapping.loc[sweep_mapping["Protocol Segment"] == "conditioning", "Step"]
            true_step_nums = df_stats.loc[
                (df_stats["Original step"].isin(orig_step_nums)) & (df_stats["Cycle"] <= n_conditioning_cycles),
                "Steps",
            ].values
            record_nums = df_details.loc[df_details["Steps"].isin(true_step_nums), "Record number"].values
            df_sweep.loc[df_sweep["Record Number"].isin(record_nums), "Sweep SOC"] = (
                sweep_mapping.loc[sweep_mapping["Protocol Segment"] == "conditioning", "Sweep SOC"].unique()[0]
            )

            # Sweep SOC: main sweeps
            for i, soc in enumerate(socs_tested):
                orig_step_nums = sweep_mapping.loc[sweep_mapping["Protocol Segment"] == "sweep", "Step"]
                cycle = i + n_conditioning_cycles
                true_step_nums = df_stats.loc[
                    (df_stats["Original step"].isin(orig_step_nums)) & (df_stats["Cycle"] == cycle),
                    "Steps",
                ].values
                record_nums = df_details.loc[df_details["Steps"].isin(true_step_nums), "Record number"].values
                df_sweep.loc[df_sweep["Record Number"].isin(record_nums), "Sweep SOC"] = soc

            # Sweep Segment
            for sweep_segment_key in sweep_mapping["Sweep Segment"].unique():
                for protocol_segment_key in ["sweep", "conditioning"]:
                    orig_step_nums = sweep_mapping.loc[
                        (sweep_mapping["Protocol Segment"] == protocol_segment_key)
                        & (sweep_mapping["Sweep Segment"] == sweep_segment_key),
                        "Step",
                    ].values
                    true_step_nums = df_stats.loc[df_stats["Original step"].isin(orig_step_nums), "Steps"].values
                    record_nums = df_details.loc[df_details["Steps"].isin(true_step_nums), "Record number"].values
                    df_sweep.loc[df_sweep["Record Number"].isin(record_nums), "Sweep Segment"] = sweep_segment_key

            pickle.dump(df_sweep, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def process_soc_sweep_for_soh_levels(
    raw_root: Path,
    save_root: Path,
    sohs: tuple[int, ...] = (70, 80, 90, 100),
    *,
    overwrite_existing: bool = False,
) -> None:
    """
    Run SOC sweep preprocessing for the assigned SOH levels.

    Expects raw folders:
      raw_root / f"SOC_Sweep_SOH{soh}" / Group*/...xlsx

    Saves to:
      save_root / f"SOH{soh}" / *.pkl
    """
    raw_root = Path(raw_root)
    save_root = Path(save_root)

    for soh in sohs:
        dir_raw = raw_root / f"SOC_Sweep_SOH{soh}"
        dir_save = save_root / f"SOH{soh}"
        dir_save.mkdir(parents=True, exist_ok=True)

        if not dir_raw.exists():
            print(f"[WARN] Missing raw folder: {dir_raw} (skipping SOH={soh})")
            continue

        process_raw_socSweep_data(
            dir_raw=dir_raw,
            dir_save=dir_save,
            overwrite_existing=overwrite_existing,
        )
