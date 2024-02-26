import pandas as pd
import numpy as np

# Functions from plot_behaviors_from_sensors.py
# Written by Hien
def euler_to_rotation_matrix(roll_rad, pitch_rad, yaw_rad):
    # Rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])

    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])

    # Total rotation matrix
    R_total = np.dot(R_z, np.dot(R_y, R_x))

    return R_total

def rotate_vector(vector, roll, pitch, yaw):
    R_total = euler_to_rotation_matrix(roll, pitch, yaw)
    result_vector = np.dot(R_total, vector)
    return result_vector

# Necessary values for behavior calculations
north_vector = np.array([0, 9, 0]).astype(float)
# From uwb_localization/utils/pen_model.py
pen_min_x, pen_max_x = -8.79, 10.42
pen_min_y, pen_max_y = -6.46, 5.33
trough_x = 0.6
trough_y = 1.8
trough_z = 0.6

behaviors = ["mineral", "feeding", "drinking", "milking"]


# Load in data for one cow for one day
tag = 1
date = '0725'
full_filepath = f'combined_cow_data/T0{tag}/cow_data_T0{tag}_{date}.csv'

# Load in data to a df
df = pd.read_csv(full_filepath)
# Add behavior columns
for behavior in behaviors:
    df[behavior] = 0

# Loop through all rows to label behavior
# TODO: Find someway to vectorize this
for i, row in df.iterrows():
    # Get all useful values from row
    loc_x, loc_y, loc_z =  row['coord_x_cm'], row['coord_y_cm'], row['coord_y_cm']
    roll, pitch, yaw = row['roll_deg'], row['pitch_deg'], row['yaw_deg']
    relative_angle = row['relative_angle_deg']

    # All logic from plot_behaviors_from_sensors.py
    if np.isnan(loc_x) == False:
        if np.abs(loc_y) < 250 and loc_x < 630 and loc_x > -400:
            # resting += 1
            pass
        if loc_y < -640:
            if loc_x > 950:
                # mineral += 1
                df.at[i, "mineral"] = 1
            else:
                # feeding += 1
                df.at[i, "feeding"] = 1

        if relative_angle > 12:
            phi = np.deg2rad(roll)
            theta = np.deg2rad(pitch)
            psi = np.deg2rad(yaw)

            drink = False
            vector = rotate_vector(north_vector, theta, phi, -psi) * 0.5
            
            v_x, v_y, v_z = vector
            if v_x < (pen_min_x - trough_x/2) or v_x > (pen_max_x - trough_x):
                if np.abs(v_y) < trough_y/2:
                    if pitch < -30:
                        drink = True

            if loc_x < (pen_min_x + trough_x) or loc_x > (pen_max_x - trough_x*1.5):
                if np.abs(loc_y) < trough_y/2:
                    if pitch < -30:
                        drink = True
            
            if drink == True:
                # drinking += 1
                df.at[i, "drinking"] = 1

    else:
        # milking += 1
        df.at[i, "milking"] = 1

# Drop all values except labeled timestamp and behavior
df = df.loc[:, df.columns.intersection(["timestamp"] + behaviors)]

# Save df
df.to_csv("behaviors.csv")