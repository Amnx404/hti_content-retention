# hr_analysis.py

"""
HR Analysis Module

This module provides functionalities to extract and analyze heart rate (HR) data
from a nested JSON structure. It includes a function to extract various HR features
based on defined start and end times and granularity.

Functions:
    - extract_hr_features(start_timestamp, end_timestamp, granularity, file_path='./Heart Rate.json'):
        Filters HR data within the specified time range and extracts features based on granularity.
"""

import json
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import linregress
import os


def _extract_data_between_timestamps(data, start_timestamp, end_timestamp):
    """
    Internal helper function to extract heart rate data entries from nested JSON data between two timestamps.

    Args:
        data (dict): Dictionary containing nested HR data entries under 'metrics'.
        start_timestamp (str): Start timestamp in the format '%Y-%m-%d %I:%M:%S %p %z'.
        end_timestamp (str): End timestamp in the format '%Y-%m-%d %I:%M:%S %p %z'.

    Returns:
        list: Filtered list of HR data entries between the given timestamps.
    """
    # Convert start and end timestamps to datetime objects
    try:
        start_dt = datetime.strptime(start_timestamp, '%Y-%m-%d %I:%M:%S %p %z')
        end_dt = datetime.strptime(end_timestamp, '%Y-%m-%d %I:%M:%S %p %z')
    except ValueError as ve:
        raise ValueError(f"Incorrect timestamp format: {ve}")

    filtered_data = []

    # Access 'metrics' within 'data'
    metrics = data.get('metrics', [])
    if not metrics:
        print("Warning: No 'metrics' key found in the data.")
        return filtered_data

    # Iterate over each metric
    for metric_index, metric in enumerate(metrics):
        metric_data = metric.get('data', [])
        if not metric_data:
            print(f"Warning: No 'data' found for metric at index {metric_index}.")
            continue

        # Iterate over each entry in the metric's data
        for entry_index, entry in enumerate(metric_data):
            # Ensure the entry is a dictionary
            if not isinstance(entry, dict):
                print(f"Warning: Entry at metrics[{metric_index}]['data'][{entry_index}] is not a dictionary. Skipping.")
                continue

            # Extract and parse the 'date' field
            date_str = entry.get("date")
            if not date_str:
                print(f"Warning: Entry at metrics[{metric_index}]['data'][{entry_index}] missing 'date'. Skipping.")
                continue

            try:
                # Normalize the date string by replacing non-breaking spaces and other Unicode spaces
                date_str_normalized = ' '.join(date_str.split())
                entry_dt = datetime.strptime(date_str_normalized, '%Y-%m-%d %I:%M:%S %p %z')
            except ValueError as ve:
                print(f"Warning: Invalid date format in entry at metrics[{metric_index}]['data'][{entry_index}]: {ve}. Skipping.")
                continue

            # Check if the entry's date is within the specified range
            if start_dt <= entry_dt <= end_dt:
                filtered_data.append(entry)

    return filtered_data


def extract_hr_features(start_timestamp, end_timestamp, granularity, file_path='./Heart Rate.json'):
    """
    Extracts relevant heart rate features from HR data within a specified time range and granularity.

    Args:
        start_timestamp (str): Start timestamp in the format '%Y-%m-%d %I:%M:%S %p %z'.
        end_timestamp (str): End timestamp in the format '%Y-%m-%d %I:%M:%S %p %z'.
        granularity (int): Segment duration in seconds.
        file_path (str, optional): Path to the JSON file containing heart rate data. Defaults to './Heart Rate.json'.

    Returns:
        pandas.DataFrame: DataFrame containing extracted features for each segment.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' was not found.")

    # Load JSON data from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            json_data = json.load(file)
        except json.JSONDecodeError as jde:
            raise ValueError(f"Error decoding JSON: {jde}")

    # Extract data within the time range using the internal helper function
    hr_entries = _extract_data_between_timestamps(json_data.get('data', {}), start_timestamp, end_timestamp)

    if not hr_entries:
        print("No HR data found within the specified time range.")
        return pd.DataFrame()  # Return empty DataFrame

    # Convert hr_entries to DataFrame
    df = pd.DataFrame(hr_entries)

    if df.empty:
        return None

    # Parse 'date' to datetime using pandas for better handling of various formats
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows with invalid dates
    df = df.dropna(subset=['date'])

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    if df.empty:
        print("No valid HR data available after parsing dates.")
        return pd.DataFrame()  # Return empty DataFrame

    # Set 'date' as index
    df.set_index('date', inplace=True)

    # Resample data based on granularity
    resample_rule = f'{granularity}S'
    grouped = df.resample(resample_rule)

    # Initialize lists to store features
    feature_list = []

    # Iterate over each segment
    for segment_time, group in grouped:
        if group.empty:
            # Handle empty segments by assigning NaN
            feature = {
                'Segment Start': segment_time,
                'Mean HR': np.nan,
                'HR Slope': np.nan,
                'HR Variability': np.nan,
                'Peak HR': np.nan,
                'HR Change': np.nan,
                'Moving Average HR': np.nan,
                'HR Transitions': np.nan
            }
        else:
            # Extract HR values
            hr_values = group['Avg'].astype(float).values
            hr_max = group['Max'].astype(float).values

            # Mean HR
            mean_hr = np.mean(hr_values)

            # HR Slope using linear regression (time in seconds since segment start)
            times = (group.index - segment_time).total_seconds()
            if len(times) > 1:
                slope, intercept, r_value, p_value, std_err = linregress(times, hr_values)
            else:
                slope = 0  # Not enough data points to determine slope

            # HR Variability (Standard Deviation)
            hr_variability = np.std(hr_values)

            # Peak HR
            peak_hr = np.max(hr_max)

            # Moving Average HR (over the segment)
            moving_avg = mean_hr  # Equivalent to mean_hr in this context

            # HR Transitions: Number of times HR changes by more than 10 bpm between consecutive entries
            if len(hr_values) > 1:
                transitions = np.sum(np.abs(np.diff(hr_values)) > 10)
            else:
                transitions = 0  # Not enough data points to determine transitions

            feature = {
                'Segment Start': segment_time,
                'Mean HR': mean_hr,
                'HR Slope': slope,
                'HR Variability': hr_variability,
                'Peak HR': peak_hr,
                'HR Change': np.nan,  # To be filled later
                'Moving Average HR': moving_avg,
                'HR Transitions': transitions
            }

        feature_list.append(feature)

    # Create DataFrame from features
    features_df = pd.DataFrame(feature_list)

    # Calculate HR Change (difference in Mean HR from previous segment)
    features_df['HR Change'] = features_df['Mean HR'].diff()

    # Fill NaN for the first segment's HR Change with 0
    features_df['HR Change'].fillna(0, inplace=True)

    # Reset index to have 'Segment Start' as a column
    features_df.reset_index(drop=True, inplace=True)

    return features_df
