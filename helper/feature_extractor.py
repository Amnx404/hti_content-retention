from .au_extract import get_au_data
from .hr_analysis import extract_hr_features


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def extract_features(start_time, end_time):
    import pandas as pd
    from datetime import datetime, timezone, timedelta

    # Original timestamps
    

    # Define the timezone offset (+05:30)
    tz_offset = timezone(timedelta(hours=5, minutes=30))

    # Parse the input strings into datetime objects
    dt_start = datetime.strptime(start_time, '%Y-%m-%d %H-%M-%S')
    dt_end = datetime.strptime(end_time, '%Y-%m-%d %H-%M-%S')

    # Assign the timezone to the datetime objects
    dt_start = dt_start.replace(tzinfo=tz_offset)
    dt_end = dt_end.replace(tzinfo=tz_offset)

    # Format the datetime objects to the desired string format
    formatted_start = dt_start.strftime('%Y-%m-%d %I:%M:%S %p %z')
    formatted_end = dt_end.strftime('%Y-%m-%d %I:%M:%S %p %z')

# Display the formatted timestamps
    # print(f"start_time = '{formatted_start}'")
    # print(f"end_time = '{formatted_end}'")

    # Get AU data
    merged_df = get_au_data(start_time, end_time, '10S')
    if merged_df is None:
        return None

    merged_df['timestamp'] = merged_df['absolute_timestamp'].dt.strftime('%Y-%m-%d %H-%M-%S')

    granularity = 15

    # Call the extract_hr_features function
    features = extract_hr_features(formatted_start, formatted_end, granularity)

    if features is None or features.empty:
        return None
    # features


    # Convert timestamps to datetime format
    features['Segment Start'] = pd.to_datetime(features['Segment Start'], errors='coerce').dt.tz_convert('UTC')
    merged_df['absolute_timestamp'] = pd.to_datetime(merged_df['absolute_timestamp'], errors='coerce').dt.tz_localize('UTC')

    # Add 5 hours and 30 minutes to the 'Segment Start' column for alignment
    features['Segment Start'] = features['Segment Start'] + pd.Timedelta(hours=5, minutes=30)
    # convert 5:30 to UTC format
    features['Segment Start'] = features['Segment Start']

    # Merge the dataframes on the adjusted timestamps
# if empty df return None

    if features.empty or merged_df.empty:
        return None
    

    merged_data = pd.merge(
        features,
        merged_df,
        how='right',
        left_on='Segment Start',
        right_on='absolute_timestamp',
    )

    adjusted_merged_data = merged_data.copy()
    adjusted_merged_data.sort_values(by='absolute_timestamp', inplace=True)

    # Set 'absolute_timestamp' as the index for interpolation
    adjusted_merged_data.set_index('absolute_timestamp', inplace=True)

    # Interpolate missing data with a time limit of 2.5 seconds
    adjusted_merged_data = adjusted_merged_data.interpolate(method='linear', limit=7, limit_direction='both')

    # Reset the index back to default
    adjusted_merged_data.reset_index(inplace=True)


    # drop based on MeanHR
    adjusted_merged_data = adjusted_merged_data.dropna(subset=['Mean HR'])
    adjusted_merged_data




    merged_df = adjusted_merged_data.copy()
    # 1. Composite Emotional Scores
    merged_df['Positive_Engagement'] = (merged_df['AU06_r_mean'] + merged_df['AU12_r_mean'] + merged_df['AU14_r_mean']) - (merged_df['AU04_r_mean'] + merged_df['AU07_r_mean'])
    merged_df['Stress_Score'] = (merged_df['AU04_r_mean'] + merged_df['AU07_r_mean'] + merged_df['AU23_r_mean']) - (merged_df['AU06_r_mean'] + merged_df['AU12_r_mean'])

    # 2. Happiness-to-Stress Ratio
    merged_df['Happiness_Stress_Ratio'] = (merged_df['AU06_r_mean'] + merged_df['AU12_r_mean']) / (merged_df['AU04_r_mean'] + merged_df['AU07_r_mean'] + merged_df['AU23_r_mean'] + 1e-5)


    # 4. Engagement Score
    merged_df['Engagement_Score'] = merged_df['HR Variability'] * merged_df['AU12_r_mean']

    # 5. Head Movement Intensity
    merged_df['Head_Movement_Intensity'] = np.sqrt(merged_df['pose_Rx_std']**2 + merged_df['pose_Ry_std']**2 + merged_df['pose_Rz_std']**2)

    # 6. Gaze Consistency
    merged_df['Gaze_Consistency'] = merged_df[['gaze_angle_x_mean', 'gaze_angle_y_mean']].var(axis=1)

    # 7. Valence and Arousal
    merged_df['Valence'] = (merged_df['AU06_r_mean'] + merged_df['AU12_r_mean'] + merged_df['AU14_r_mean']) / 3
    merged_df['Arousal'] = (merged_df['AU04_r_mean'] + merged_df['AU07_r_mean'] + merged_df['AU09_r_mean'] +
                            merged_df['AU10_r_mean'] + merged_df['AU20_r_mean'] + merged_df['AU23_r_mean'] +
                            merged_df['AU25_r_mean'] + merged_df['AU26_r_mean']) / 8

    # 8. Facial Activity Variability (AU Peak Ratio)
    merged_df['AU_Peak_Ratio'] = (merged_df['AU06_r_mean'] + merged_df['AU12_r_mean']) / (merged_df['AU04_r_mean'] + merged_df['AU07_r_mean'] + 1e-5)

    # 9. PCA on Selected AUs
    # pca_features = ['AU01_r_mean', 'AU02_r_mean', 'AU04_r_mean', 'AU05_r_mean', 'AU06_r_mean',
    #                'AU07_r_mean', 'AU09_r_mean', 'AU10_r_mean', 'AU12_r_mean', 'AU14_r_mean',
    #                'AU15_r_mean', 'AU17_r_mean', 'AU20_r_mean', 'AU23_r_mean', 'AU25_r_mean',
    #                'AU26_r_mean', 'AU45_r_mean']

    # scaler = StandardScaler()
    # scaled_pca_features = scaler.fit_transform(merged_df[pca_features])

    # pca = PCA(n_components=5)
    # principal_components = pca.fit_transform(scaled_pca_features)

    # for i in range(1, 6):
    #     merged_df[f'PC{i}'] = principal_components[:, i-1]
    # 3. Cognitive Load Index
    merged_df['Cognitive_Load'] = (
        0.10 * merged_df['Mean HR'] +
        0.10 * merged_df['HR Variability'] +
        0.15 * merged_df['AU04_r_mean'] +
        0.15 * merged_df['AU07_r_mean'] +
        0.10 * merged_df['Stress_Score'] +
        0.05 * merged_df['Arousal'] +
        0.05 * merged_df['Valence'] +
        0.05 * merged_df['Head_Movement_Intensity'] +
        0.05 * merged_df['Gaze_Consistency']
    )


    # 10. Final Feature Set
    final_features = [
        'Positive_Engagement', 'Stress_Score', 'Happiness_Stress_Ratio',
        'Cognitive_Load', 'Engagement_Score', 'Head_Movement_Intensity',
        'Gaze_Consistency', 'Valence', 'Arousal', 'AU_Peak_Ratio'
        # 'PC1', 'PC2', 'PC3', 'PC4', 'PC5'
    ]


    # Optionally, drop original AU columns to reduce dimensionality
    # merged_df = merged_df.drop(columns=pca_features)

    # Now, `merged_df` contains all the engineered features ready for modeling
    return merged_df