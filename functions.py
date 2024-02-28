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


if __name__ == "__main__":
    import argparse

    class HelpFormatter(argparse.RawDescriptionHelpFormatter):
        pass

    # class HelpFormatter( argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter ): pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)

    # Add the arguments
    parser.add_argument(
        "--root", help="Root folder", default="./derivatives/gradient_removal"
    )

    parser.add_argument("--subject", help="Subject number", default="04")

    parser.add_argument("--session", help="Session number", default="01")

    parser.add_argument("--task", help="Task name", default="rest")

    parser.add_argument("--run", help="Run", default="01")

    parser.add_argument("--description", help="Description", default="step1")

    parser.add_argument("--datatype", help="Data type", default="eeg")

    # Parse the arguments
    opts = parser.parse_args()

# third-party imports (and comments indicating how to install them)
import mne          # python -m conda install -c conda-forge mne       or    python -m pip install mne
import mne_bids     # python -m conda install -c conda-forge mne-bids  or    python -m pip install mne-bids
import numpy as np  # python -m conda install -c conda-forge numpy     or    python -m pip install numpy
import scipy        # python -m conda install -c conda-forge scipy     or    python -m pip install scipy

__all__ = [
    "input_interpreter",
    "read_raw_eeg",
    "numerical_explorer",
]


class NoSubjectFoundError(Exception):
    pass


class NoSessionFoundError(Exception):
    pass


class NoDataTypeError(Exception):
    pass


class ReadingFileError(Exception):
    pass


def input_interpreter(input_string, input_param, max_value=1000):
    """Interpret input string as a list of integers.
    The input string can contain:
        - a list of integers separated by commas (e.g. "1,3,5,7")
        - a range of integers separated by a hyphen (e.g. "1-5") which will be
          interpreted as "1,2,3,4,5"
        - a range of integers separated by a hyphen with an asterisk
          (e.g. "1-*") which will be interpreted as "1 to the maximum value"
          which can be specified by the max_value argument (max number of
          subjects for example)

    The input can be a combination of the above (e.g. "1,3-5,7-*") which will
    be interpreted as "1,3,4,5,7 to the maximum value"


    Args:
        input_string (str): input string

    Returns:
        list: list of integers
    """
    elements = input_string.split(",")
    desired_subject_numbers = []
    for element in elements:
        if "-" in element:
            start, stop = element.split("-")
            start = start.replace("*", "0")
            stop = stop.replace("*", str(max_value))
            start = start.strip()
            stop = stop.strip()
            if start.isnumeric() and stop.isnumeric():
                desired_subject_numbers.extend(range(int(start), int(stop) + 1))
            else:
                print(f"Please make sure that '{input_param}'='{input_string}' is correctly formatted. See help for more information.")
                break
        else:
            if element.strip().isnumeric():
                desired_subject_numbers.append(int(element))
            elif element.strip() == "*":
                desired_subject_numbers.extend(range(1, max_value + 1))
            else:
                print(f"Please make sure that '{input_param}'='{input_string}' is correctly formatted. See help for more information.")
                break
    return desired_subject_numbers


def read_raw_eeg(filename, preload=False):
    """read_raw_eeg.
    Wrapper function around mne.io.read_raw_* functions to chose the right reading method
    as a function of the extension of the file.

    Format allowed are:
    - egi (.mff, .RAW)
    - bdf (.bdf)
    - edf (.edf)
    - fif (.fif)
    - eeglab (.set)
    - brainvision (.eeg)

    Args:
        filename (str): path to the file
        preload (bool, optional): If True, the data will be preloaded into memory. Defaults to False.

    Returns:
        raw: mne.Raw object
        status: (str) 'ok' if the file is read correctly, 'corrupted' if the file is corrupted
    """
    if os.path.exists(filename):
        extension = os.path.splitext(filename)[1]
        if extension == ".mff" or extension == ".RAW":
            method = "read_raw_egi"
        elif extension == ".bdf":
            method = "read_raw_bdf"
        elif extension == ".edf":
            method = "read_raw_edf"
        elif extension == ".fif":
            method = "read_raw_fif"
        elif extension == ".set":
            method = "read_raw_eeglab"
        elif extension == ".vhdr":
            method = "read_raw_brainvision"

        reader = getattr(mne.io, method)
        try:
            raw = reader(filename, preload=preload)
            return raw

        except ReadingFileError:
            print(
                f"File {filename} is corrupted or extension {extension} is not recognized"
            )

    else:
        raise FileNotFoundError(f"File {filename} does not exist")


def numerical_explorer(directory, prefix):
    """Give the existing numerical elements based on the prefix

    Args:
        directory (str or pathlike): The directory to explore
        prefix (str): The prefix to filter the elements (e.g., "sub", "ses", "run")

    Returns:
        list: List of existing numerical elements based on the prefix
    """
    if os.path.isdir(directory):
        if prefix == "run":
            elements = (
                int(element.split("_run-")[-1][:2])
                for element in os.listdir(directory)
                if prefix in element
            )
        else:
            elements = (
                int(element.split("-")[-1])
                for element in os.listdir(directory)
                if prefix in element
            )
        if elements:
            return elements
        else:
            raise ValueError(f"No element with prefix '{prefix}' found in {directory}")
    else:
        raise NotADirectoryError(f"{directory} is not a directory")


def avg_fft_calculation(signal):
    avg_fft = []
    for channel in range(signal.shape[0]):
        fft = abs(scipy.fft.fft(np.squeeze(signal[channel, :])))
        fft = fft[: len(fft) // 2]
        avg_fft.append(np.mean(fft))

    return avg_fft


def rms_calculation(signal):
    rms = []
    for channel in range(signal.shape[0]):
        rms.append(np.sqrt(np.mean(signal[channel, :] ** 2)))
    return rms


def max_gradient_calculation(signal, sampling_rate=1000):
    """Calculate the maximum gradient of the signal
    Largest difference between samples within the signal
    Threshold is usually 10uV/ms

    Args:
        signal (_type_): _description_

    Returns:
        list of bool: _d
    """
    if sampling_rate <= 1000:
        samples = 1
    else:
        samples = int(sampling_rate / 1000)
    max_gradient = []
    for channel in range(signal.shape[0]):
        max_gradient.append(np.max(np.diff(signal[channel, :], n=samples)))
    return max_gradient


def kurtosis_calculation(signal):
    """
    Calculate the kurtosis of each channel in the EEG signal.

    Kurtosis is a statistical measure that describes the shape of a distribution.
    In the context of EEG signal quality assessment, kurtosis is important because it provides information
    about the presence of outliers or extreme values in the signal. High kurtosis values indicate heavy tails
    and a higher likelihood of extreme values, which can be indicative of artifacts or abnormal brain activity.
    Therefore, calculating the kurtosis of each channel in the EEG signal can help in identifying channels
    with potential quality issues or abnormalities.

    Parameters:
    signal (numpy.ndarray): The EEG signal with shape (num_channels, num_samples).

    Returns:
    list: A list of kurtosis values, where each value corresponds to a channel in the signal.

    References:
    - scipy.stats.kurtosis: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html
    """
    kurtosis = []
    for channel in range(signal.shape[0]):
        kurtosis.append(scipy.stats.kurtosis(signal[channel, :]))
    return kurtosis


def zero_crossing_calculation(signal):
    """
    Calculates the number of zero crossings in each channel of the given EEG signal.

    Zero crossing refers to the point at which the signal changes its polarity, i.e., when it crosses the zero axis.
    In EEG analysis, zero crossing calculation can provide insights into the frequency content and dynamics of the signal.
    It is often used as a feature to characterize the temporal properties of the EEG waveform.

    Parameters:
    signal (numpy.ndarray): The input EEG signal with shape (num_channels, num_samples).

    Returns:
    list: A list containing the number of zero crossings for each channel in the EEG signal.
    """
    zero_crossing = []
    for channel in range(signal.shape[0]):
        zero_crossing.append(np.sum(np.abs(np.diff(np.sign(signal[channel, :]))) == 2))
    return zero_crossing


def hjorth_parameters(signal):
    """
    Calculate Hjorth parameters.

    This function calculates the Hjorth parameters for a given signal.
    The Hjorth parameters are measures of activity, mobility, and complexity
    of the signal.

    Args:
        signal (numpy.ndarray): The input signal for which to calculate the Hjorth parameters.
            The signal should be a 2D array, where each row represents a different channel
            and each column represents a different time point.

    Returns:
        tuple: A tuple containing the calculated Hjorth parameters.
            - activity (numpy.ndarray): The activity parameter for each channel.
            - mobility (numpy.ndarray): The mobility parameter for each channel.
            - complexity (numpy.ndarray): The complexity parameter for each channel.
    """
    activity = np.var(signal, axis=1)
    mobility = np.sqrt(np.var(np.diff(signal, axis=1), axis=1) / activity)
    complexity = np.sqrt(
        np.var(np.diff(np.diff(signal, axis=1), axis=1), axis=1)
        / np.var(np.diff(signal, axis=1), axis=1)
    )
    return activity, mobility, complexity
