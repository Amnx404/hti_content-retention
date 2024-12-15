
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# start_time = '2024-12-01 22-55-00'
# end_time = '2024-12-01 23-02-00'
def get_au_data(start_time, end_time, granularity):
    import pandas as pd
    import numpy as np
    video_duration_list = pd.read_csv("videoduration.csv")
    video_duration_list.columns
    #make a dictionary of name : duration columns
    video_duration_dict = dict(zip(video_duration_list['name'], video_duration_list['duration']))

    import json
    from datetime import datetime, timedelta
    import os
    import pandas as pd

    def extract_video_data_between_timestamps(video_files, start_timestamp, end_timestamp):
        """
        Extract data entries from a series of video files between two timestamps.

        :param video_files: Dictionary of video file names and their durations in seconds
                        {"video_name.csv": duration_in_seconds}
        :param start_timestamp: Start timestamp as a string in the format '%Y-%m-%d %H-%M-%S'
        :param end_timestamp: End timestamp as a string in the format '%Y-%m-%d %H-%M-%S'
        :return: Dictionary with video file names and corresponding filtered DataFrames
        """
        start_dt = datetime.strptime(start_timestamp, '%Y-%m-%d %H-%M-%S')
        end_dt = datetime.strptime(end_timestamp, '%Y-%m-%d %H-%M-%S')

        filtered_data = {}

        for video_name, duration in video_files.items():
            # Extract the start time from the video name
            video_start_str = video_name.split('.')[0]  # Remove file extension
            video_start_dt = datetime.strptime(video_start_str, '%Y-%m-%d%H-%M-%S')
            video_end_dt = video_start_dt + timedelta(seconds=duration)

            # Check if the video overlaps with the desired time range
            if video_end_dt < start_dt or video_start_dt > end_dt:
                continue  # Skip videos outside the range

            # Load the video data
            video_path = f'./Out/{video_name}'
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"The video file '{video_name}' was not found.")

            video_data = pd.read_csv(video_path)

            # If timestamp is in seconds relative to video start, use it directly
            video_data['absolute_timestamp'] = video_start_dt + pd.to_timedelta(video_data[' timestamp'], unit='s')

            # Filter rows within the desired time range
            filtered_video_data = video_data[(video_data['absolute_timestamp'] >= start_dt) &
                                            (video_data['absolute_timestamp'] <= end_dt)]

            filtered_data[video_name] = filtered_video_data

        return filtered_data

    # Example usage:
    video_files = video_duration_dict


    filtered_videos = extract_video_data_between_timestamps(video_files, start_time, end_time)

    # for video_name, data in filtered_videos.items():
        # print(f"Filtered data for {video_name}:")
        # print(data.head())
        # print(data.tail())
    if not filtered_videos:
        return None

    # Concat all data
    all_au_data = pd.concat(filtered_videos.values())

    if all_au_data.empty: # No data available
        return None

    au_features_df = all_au_data

    

    # Assume `merged_df` is the dataframe after merging AU and HR data and aggregating filtered_videos on seconds absolute timestamp
    au_features_df.set_index("absolute_timestamp", inplace=True)  
    selected_aus = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r',
                ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r',
                ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r',
                ' AU26_r', ' AU45_r', ' AU01_c', ' AU02_c',
                ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c',
                ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c',
                ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c',
                ' AU45_c', ' pose_Rx', ' pose_Ry', ' pose_Rz', ' gaze_angle_x', ' gaze_angle_y']
    
    au_aggregated = au_features_df[selected_aus].resample(granularity).agg(['mean', 'max', 'std'])

    au_aggregated.columns = ['_'.join(col).strip() for col in au_aggregated.columns.values]
    au_aggregated.reset_index(inplace=True)
    return au_aggregated