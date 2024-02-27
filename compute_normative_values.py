
#!/usr/bin/env -S  python  #
# -*- coding: utf-8 -*-
# ===============================================================================
# Author: Dr. Samuel Louviot, PhD
# Institution: Nathan Kline Institute
# Address: 140 Old Orangeburg Rd, Orangeburg, NY 10962, USA
# Date: 2024-01-19
# Modified: 2024-02-08
# email: samuel DOT louviot AT nki DOT rfmh DOT org
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
import pickle
from tqdm import tqdm         # python -m conda install -c conda-forge tqdm      or    python -m pip install tqdm
from functions import (
    avg_fft_calculation,
    rms_calculation,
    max_gradient_calculation,
    kurtosis_calculation,
    zero_crossing_calculation,
    hjorth_parameters,
)

def run_qa(raw):
    sampling_rate = raw.info['sfreq']
    n_samples = raw.n_times
    n_chans_original = raw.info['nchan']
    eeg_data = raw.get_data()
    win_length_secs = 1

    win_size = int(win_length_secs * sampling_rate)
    win_offsets = np.arange(1, (n_samples - win_size), win_size)
    win_count = len(win_offsets)
    
    # Initialize per-window arrays for each type of noise info calculated below
    avg_fft =           np.zeros((win_count, n_chans_original))
    rms =               np.zeros((win_count, n_chans_original))
    max_gradient =      np.zeros((win_count, n_chans_original))
    kurtosis =          np.zeros((win_count, n_chans_original))
    zero_crossing =     np.zeros((win_count, n_chans_original))
    hjorth_activity =   np.zeros((win_count, n_chans_original))
    hjorth_mobility =   np.zeros((win_count, n_chans_original))
    hjorth_complexity = np.zeros((win_count, n_chans_original))

    # Go through each window and calculate the noise info
    for w in tqdm(range(win_count)):
        
        # Get both filtered and unfiltered data for the current window
        start, end = (w * win_size, (w + 1) * win_size)
        eeg_raw = eeg_data[:, start:end]

        avg_fft[w, :] = avg_fft_calculation(eeg_raw)
        rms[w, :] = rms_calculation(eeg_raw)
        max_gradient[w, :] = max_gradient_calculation(eeg_raw)
        kurtosis[w, :] = kurtosis_calculation(eeg_raw)
        zero_crossing[w, :] = zero_crossing_calculation(eeg_raw)
        hjorth_activity[w,:], hjorth_mobility[w,:], hjorth_complexity[w,:] = hjorth_parameters(eeg_raw)
    
    a_avg_fft =           np.mean(avg_fft, axis=0)
    a_rms =               np.mean(rms, axis=0)
    a_max_gradient =      np.mean(max_gradient, axis=0)
    a_kurtosis =          np.mean(kurtosis, axis=0)
    a_zero_crossing =     np.mean(zero_crossing, axis=0)
    a_hjorth_activity =   np.mean(hjorth_activity, axis=0)
    a_hjorth_mobility =   np.mean(hjorth_mobility, axis=0)
    a_hjorth_complexity = np.mean(hjorth_complexity, axis=0)
    
    sd_avg_fft = np.std(avg_fft, axis=0)
    sd_rms = np.std(rms, axis=0)
    sd_max_gradient = np.std(max_gradient, axis=0)
    sd_kurtosis = np.std(kurtosis, axis=0)
    sd_zero_crossing = np.std(zero_crossing, axis=0)
    sd_hjorth_activity = np.std(hjorth_activity, axis=0)
    sd_hjorth_mobility = np.std(hjorth_mobility, axis=0)
    sd_hjorth_complexity = np.std(hjorth_complexity, axis=0)
    
    d = {
        'average_fft': {
            'mean': a_avg_fft,
            'std': sd_avg_fft
        },
        'rms': {
            'mean': a_rms,
            'std': sd_rms
        },
        'max_gradient': {
            'mean': a_max_gradient,
            'std': sd_max_gradient
        },
        'kurtosis': {
            'mean': a_kurtosis,
            'std': sd_kurtosis
        },
        'zero_crossing': {
            'mean': a_zero_crossing,
            'std': sd_zero_crossing
        },
        'hjorth_activity': {
            'mean': a_hjorth_activity,
            'std': sd_hjorth_activity
        },
        'hjorth_mobility': {
            'mean': a_hjorth_mobility,
            'std': sd_hjorth_mobility
        },
        'hjorth_complexity': {
            'mean': a_hjorth_complexity,
            'std': sd_hjorth_complexity
        },
        'channel_names': raw.info['ch_names'],
        
    }
    
    return d

subjects = ['HC008',
            'HC009',
            'HC070']
big_raw = []
for subject in subjects:
    bids_path = mne_bids.BIDSPath(subject=subject, 
                                session='01', 
                                task='ant', 
                                suffix='eeg',
                                datatype='eeg',
                                description='reviewed',
                                root='/Volumes/portable_data/proj-adult_TBI_attention/derivatives/reviewed')
    raw = mne_bids.read_raw_bids(bids_path)
    raw.load_data()
    ica = mne.preprocessing.ICA(random_state=97, method='fastica')
    ica.fit(raw)
    ica.plot_sources(raw, block=True)
    ica.apply(raw)
    raw.interpolate_bads(reset_bads=True)
    statistics = run_qa(raw)
    print(statistics)

    with open(f'statistics{subject}.pickle', 'wb') as f:
        pickle.dump(statistics, f)