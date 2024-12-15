# main_script.py

from hr_analysis import extract_hr_features

# Define the path to your JSON file
file_path = './Heart Rate.json'

# Define start and end times in the specified format
start_time = '2024-11-10 12:00:00 AM +0530'
end_time = '2024-12-14 12:30:00 AM +0530'

# Define granularity in seconds (e.g., 60 for 1-minute segments)
granularity = 60

# Call the extract_hr_features function
features = extract_hr_features(start_time, end_time, granularity, file_path)

# Check if any features were extracted
if features.empty:
    print("No heart rate data available for the specified time range.")
else:
    # Display the extracted features
    print(features)
