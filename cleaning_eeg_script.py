#!/usr/bin/env -S  python  #
# -*- coding: utf-8 -*-
# ===============================================================================
# Author: Dr. Samuel Louviot, PhD
#         Dr. Alp Erkent, MD, MA
# Institution: Nathan Kline Institute
#              Child Mind Institute
# Address: 140 Old Orangeburg Rd, Orangeburg, NY 10962, USA
#          215 E 50th St, New York, NY 10022
# Date: 2024-02-27
# email: samuel DOT louviot AT nki DOT rfmh DOT org
#        alp DOT erkent AT childmind DOT org
# ===============================================================================
# LICENCE GNU GPLv3:
# Copyright (C) 2024  Dr. Samuel Louviot, PhD
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ===============================================================================

"""
GENERAL DOCUMENTATION HERE
"""

# standard-library imports
import argparse
import os
from pathlib import Path
from copy import copy


if __name__ == "__main__":
    import argparse

    class HelpFormatter(argparse.RawDescriptionHelpFormatter):
        pass

    # class HelpFormatter( argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter ): pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)

    # Add the arguments
    parser.add_argument("--root", help="Root folder", default=".")

    parser.add_argument("--subject", help="Subject number", default="04")

    parser.add_argument("--session", help="Session number", default="01")

    parser.add_argument("--extension", help="Extension", default=".vhdr")

    parser.add_argument("--task", help="Task name", default="rest")

    parser.add_argument("--run", help="Run", default="01")

    parser.add_argument("--suffix", help="Suffix", default="eeg")
    
    parser.add_argument("--description", help="Description", default="step1")

    parser.add_argument("--datatype", help="Data type", default="eeg")

    parser.add_argument(
        "--datafolder",
        help="Which folder the data are located can only be: 'source', 'rawdata' or 'derivatives'",
        default="eeg",
    )

    # Parse the arguments
    opts = parser.parse_args()

# third-party imports (and comments indicating how to install them)
import mne  # python -m conda install -c conda-forge mne         or    python -m pip install mne
import mne_bids  # python -m conda install -c conda-forge mne-bids     or    python -m pip install mne-bids
from functions import *
from GradientRemover import GradientRemover

__all__ = [
    "bcg_eog_cleaner",
    "map_channel_type",
]

class EndUserError(Exception):
    pass

def map_channel_type(raw):
    """find and map into MNE type the ECG and EOG channels

    Args:
        raw (mne.io.Raw): MNE raw object

    Returns:
        dict: dictionary of channel type to map into `raw.set_channel_types` method
    """
    channels_mapping = dict()
    for ch_type in ["ecg", "eog"]:
        ch_names_raw = [
            ch_name for ch_name in raw.ch_names if ch_type in ch_name.lower()
        ]
        if ch_names_raw:
            channels_mapping.update(
                {ch_name_raw: ch_type for ch_name_raw in ch_names_raw}
            )
        else:
            print(f"No {ch_type.upper()} channel found.")
            if ch_type == "eog":
                print("Fp1 and Fp2 will be used for EOG signal detection")
                channels_mapping.update({"eog": "will be Fp1 and Fp2"})

    return channels_mapping


def gradient_cleaner(raw, bids_path):
    local_bids_path = copy(bids_path)
    raw.filter(1, None)
    gradient_trigger = mne.events_from_annotations(raw, event_id={"Stimulus/R128": 0})
    g_remover = GradientRemover(raw.get_data(), gradient_trigger[0])
    raw_corrected = g_remover.correct()
    raw = mne.io.RawArray(raw_corrected, raw.info)

    root_path = Path(local_bids_path.root)
    saving_root = root_path.parent / "pre_cleaning"
    
    local_bids_path.update(
        root=saving_root, description="GradientRemoved", suffix="eeg", extension=".fif"
    )
    local_bids_path.mkdir()
    raw.save(f"{local_bids_path.fpath}", overwrite=True)

    return raw


def bcg_eog_cleaner(raw, bids_path):
    """Pre-clean EEG data from BCG and EOG artifacts"""
    local_bids_path = copy(bids_path)
    # ================================================================================
    # READING RAW DATA
    # ================================================================================

    raw.filter(1, 50).resample(250)

    # ===============================================================================
    # ECG AND EOG CHANNELS DETECTION
    # ===============================================================================
    channel_map = map_channel_type(raw)
    if channel_map:
        raw.set_channel_types(channel_map)
    raw.set_montage("standard_1005", on_missing="warn")

    # ===============================================================================
    # COMPUTE SSP ON RAW
    # ================================================================================
    raw_projs = mne.compute_proj_raw(raw, n_eeg=3)
    raw.add_proj(raw_projs)
    root_path = Path(local_bids_path.root)
    saving_root = root_path.parent / "pre_cleaning"
    local_bids_path.update(root=saving_root, description="SSP1", suffix=None)
    local_bids_path.mkdir()
    # I put suffix = None above so I have to modify the filename manually bellow
    # Because mne_bids is a little bit too narrow minded
    mne.write_proj(f"{local_bids_path.fpath}.fif", raw_projs, overwrite=True)
    raw.apply_proj()

    # ================================================================================
    # COMPUTE SSP ON ECG AND EOG
    # ================================================================================

    for i, channel_type in enumerate(channel_map.values()):
        if channel_type == "ecg":
            number_of_vectors = 2
            channels = "ECG"
        elif channel_type == "eog":
            number_of_vectors = 1
            channels = ["Fp1", "Fp2"]
        projections, _ = getattr(mne.preprocessing, f"compute_proj_{channel_type}")(
            raw, ch_name=channels, n_eeg=number_of_vectors, reject=None
        )
        raw.add_proj(projections)
        local_bids_path.update(description=f"SSP{i+2}{channel_type}", suffix=None)
        mne.write_proj(f"{local_bids_path.fpath}.fif", projections, overwrite=True)
        raw.apply_proj()

    if not channel_map:
        print(
            "No electrophysiological projections computed due to lack of ECG and EOG channels."
        )
    return raw


def Main(**kwargs):
    kwargs.update({key: None for key in kwargs.keys() if kwargs.get(key).lower() == "none"})
    reading_root = Path(kwargs["root"], kwargs["datafolder"])
    print(f"Reading root: {reading_root}")
    print(kwargs.items())
    try:
        existing_subject_numbers = set(numerical_explorer(reading_root, "sub"))
        print(f"Existing subject numbers: {existing_subject_numbers}")
    except:
        raise Exception

    try:
        desired_subject_numbers = set(
            input_interpreter(
                kwargs["subject"], "sub", max_value=max(existing_subject_numbers)
            )
        )
    except:
        raise Exception

    subject_numbers = list(
        existing_subject_numbers.intersection(desired_subject_numbers)
    )
    subject_list = [f"{subject_number:02d}" for subject_number in subject_numbers]

    for subject in subject_list:
        subject_path = reading_root / f"sub-{subject}"
        print(f"Subject path: {subject_path}")
        try:
            existing_session_numbers = set(numerical_explorer(subject_path, "ses"))
            print(f"Existing session numbers: {existing_session_numbers}")
        except:
            raise Exception

        try:
            desired_session_numbers = set(
                input_interpreter(
                    kwargs["session"], "ses", max_value=max(existing_session_numbers)
                )
            )
        except:
            raise Exception

        session_numbers = list(
            existing_session_numbers.intersection(desired_session_numbers)
        )
        session_list = [f"{session_number:02d}" for session_number in session_numbers]

        for session in session_list:
            session_path = subject_path / f"ses-{session}" / kwargs["datatype"]
            print(f"Session path: {session_path}")
            try:
                existing_run_numbers = set(numerical_explorer(session_path, "run"))
                print(f"Existing run numbers: {existing_run_numbers}")
            except:
                raise Exception

            try:
                desired_run_numbers = set(
                    input_interpreter(
                        kwargs["run"], "run", max_value=max(existing_run_numbers)
                    )
                )
            except:
                raise Exception

            run_numbers = list(
                existing_run_numbers.intersection(desired_run_numbers)
            )
            run_list = [f"{run_number:02d}" for run_number in run_numbers]

            for run in run_list:
                bids_path = mne_bids.BIDSPath(
                    subject=subject,
                    suffix=kwargs.get("suffix"),
                    extension=kwargs.get("extension"),
                    root=reading_root,
                    session=session,
                    task=kwargs.get("task"),
                    run=run,
                    datatype=kwargs.get("datatype"),
                    description=kwargs.get("description"),
                )

                raw = read_raw_eeg(str(bids_path.fpath), preload=True)
                # raw = gradient_cleaner(raw, bids_path)
                raw = bcg_eog_cleaner(raw, bids_path)
                bids_path.update(description="precleaning", suffix="eeg", extension=".fif")
                raw.save(str(bids_path.fpath), overwrite=True)

# "Main" block for this file (if this was a package rather than a single-file module, we would just cut this code out and put it in WhateverThePackageIsCalled/__main__.py (without the if statement)
if __name__ == "__main__":
    print(opts)
    try:
        # ENABLE POSSIBLITY OF LOOPING THROUGH MULTIPLE SUBJECTS
        Main(**opts.__dict__)

    except EndUserError as error:
        raise SystemExit(error)
