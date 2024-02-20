

import datetime as dt
import os

import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time
from datetime import datetime

def search_files(folder_path, search_text):
    file_names = []
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        # print(file_name)
        # Check if the file is a text file
        if file_name.endswith(".csv"):
            # Check if the search text is present in the file name
            if search_text in file_name:
                file_names.append(file_name)
    return sorted(file_names)

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

# ===============================================
""" Main program from here """
if __name__ == '__main__':
    current_dir = os.path.join(os.path.dirname(__file__))  # Folder
    project_dir = os.path.dirname(current_dir)   # Get the parent directory (one level up)

    tag_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tag_list = [2]

    import sys
    sys.path.append(project_dir)
    from CPS_cow_data_v1.environmental_data.plot_avg_THI import *
    from CPS_cow_data_v1.cow_data.plot_ankle_resting_by_day import *
    from uwb_localization.utils.pen_model import *

    THI_timestamps, daily_THI = get_avg_THI()

    # fig, ax1 = plt.subplots(figsize=(10, 6))

    for tag_id in tag_list:

        tag_name = f'T{tag_id:02d}' 
        cow_name = f'C{tag_id:02d}' 
        print(tag_name)

        north_vector = np.array([0, 9, 0]).astype(float)

        input_dir = project_dir + '/CPS_cow_data_v1/cow_data'
        ankle_standing_timestamps, daily_ankle_resting = get_ankle_resting(input_dir, tag_id)

        input_path = project_dir + "/uwb_localization/data/combined_cow_data/" + tag_name + "/"

        search_text = 'cow_data'
        matched_file_names = search_files(input_path, search_text)
        # print(f"\tMatching csv files with '{search_text}' in the name:")

        # For each day
        if len(matched_file_names) > 0:
            
            time_list = []
            resting_list = []
            feeding_list = []
            standing_list = []
            milking_list = []
            mineral_list = []
            drinking_list = []
            stand_n_feed_list = []

            if len(matched_file_names[1:-1]) != len(daily_ankle_resting):
                print("Number of dates mismatched")
                exit()

            for input_file_name, ankle_resting in zip(matched_file_names[1:-1], daily_ankle_resting):
                print('File: ' + input_file_name)
                file_path = input_path + "/" + input_file_name

                df = pd.read_csv(file_path) # skip the firt row, otherwise: header = True
                sensor_data = df.to_numpy()

                n_samples = len(sensor_data[:,0])
                # print(f"n_samples {n_samples}")

                resting = 0
                feeding = 0
                milking = 0
                mineral = 0
                drinking = 0

                for row in sensor_data:
                    loc_x, loc_y, loc_z = row[1:4]
                    roll, pitch, yaw = row[4:7]
                    relative_angle  = row[7]

                    if np.isnan(loc_x) == False:
                        if np.abs(loc_y) < 250 and loc_x < 630 and loc_x > -400:
                            # resting += 1
                            pass
                        if loc_y < -640:
                            if loc_x > 950:
                                mineral += 1
                            else:
                                feeding += 1

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
                                drinking += 1

                    else:
                        milking += 1
                
                resting = ankle_resting / 24 * n_samples

                # scale = 24
                # standing = n_samples - resting - feeding - milking # by number of uwb samples

                scale = 100
                standing = n_samples - resting - feeding - milking - mineral # by number of uwb samples
                
                resting_duration = (resting / n_samples) * scale 
                feeding_duration = (feeding / n_samples) * scale
                standing_duration = (standing / n_samples) * scale
                milking_duration = (milking / n_samples) * scale
                mineral_duration = (mineral / n_samples) * scale
                drinking_duration = (drinking / n_samples) * scale
                # print(f"resting_duration {resting_duration:.1f}")
                # print(f"feeding_duration {feeding_duration:.1f}")

                time_list.append(dt.datetime.fromtimestamp(sensor_data[0,0]).date())
                resting_list.append(resting_duration)
                feeding_list.append(feeding_duration)
                standing_list.append(standing_duration)
                milking_list.append(milking_duration)
                mineral_list.append(mineral_duration)
                drinking_list.append(drinking_duration)
                stand_n_feed_list.append(standing_duration + feeding_duration)

                # datet = [dt.datetime.fromtimestamp(ts) for ts in timestamps]

            # print(pd.DatetimeIndex(time_list))

            # ## First axis ------------------------------------------------------
            # fig, ax1 = plt.subplots(figsize=(10, 6))
            # ax1.set_title(cow_name)
            # ax1.grid(color='gray', linestyle=':', linewidth=0.5)

            # # ax1.set_ylim([6, 18])

            # ax1.bar(time_list, standing_list, label = 'standing')
            # ax1.set_ylim([0, 14])

            # # ax1.bar(time_list, mineral_list, label = 'licking mineral')
            # # ax1.set_ylim([0, 1])

            # # ax1.bar(time_list, drinking_list, label = 'drinking')
            # # ax1.set_ylim([0, 0.11])

            # # plt.plot(time_list, milking_list, label = 'milking')
            # # plt.ylim([0, 2])

            # # plt.bar(time_list, stand_n_feed_list, label = 'stand n feed')
            # # ax1.set_ylim([4, 16])
            # # plt.bar(time_list, stand_n_feed_list, label = 'stand n feed')
            # # ax1.set_ylim([0, 100])

            # # plt.plot(time_list, feeding_list, label = 'feeding')
            # # ax1.set_ylim([4, 14])
                
            # ax1.set_ylabel("Hours")

            # # plt.ylim([0, 2])
            
            ## Combined graph --------------------------------------------------
            ## First axis ------------------------------------------------------
            fig, ax1 = plt.subplots(figsize=(6, 4))
            # ax1.set_title(cow_name)
            ax1.grid(color='gray', linestyle=':', linewidth=0.5)

            # ax1.set_ylim([6, 18])

            ax1.bar(time_list, standing_list, label = 'standing', color='tab:orange')
            bottom = np.asarray(standing_list)
            ax1.bar(time_list, feeding_list, bottom=bottom, label = 'feeding', color='tab:blue')
            bottom += np.asarray(feeding_list)
            ax1.bar(time_list, mineral_list, bottom=bottom, label = 'licking', color='darkred')
            bottom += np.asarray(mineral_list)
            ax1.bar(time_list, milking_list, bottom=bottom, label = 'milking', color='y')
            bottom += np.asarray(milking_list)
            ax1.bar(time_list, resting_list, bottom=bottom, label = 'resting', color='tab:green')
            
            ax1.set_ylabel("Percentage (%)")
            # ax1.set_ylim([0, 14])

            ## Second axis -----------------------------------------------------
            ax2 = ax1.twinx()
            ax2.plot(THI_timestamps, daily_THI, color='red', linestyle='-', linewidth=3, label='avg THI')
            ax2.set_ylim([55,95])
            # ax1.set_ylabel("deg C")
            # plt.legend()

            # plt.legend()
            ax1.set_xlabel("Date")
            ax2.set_ylabel("THI")
            # plt.tight_layout() 

            # Combine legends for both y-axes
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc="upper right", ncol=2)

            # ax2.set_xticks(['a','a','a','a','a','a','a'])
            plt.xticks(rotation=30)  # Rotate x-axis labels for better readability
            plt.gca().xaxis.set_major_formatter(md.DateFormatter('%m/%d'))
            plt.gca().xaxis.set_major_locator(md.DayLocator(interval=2))
            # plt.grid(color='gray', linestyle=':', linewidth=0.5)

            output_path = current_dir + '/combined_behaviors.pdf'
            plt.tight_layout()
            plt.savefig(output_path)
            


    #     plt.plot(time_list, mineral_list, label=cow_name)

    # plt.gca().xaxis.set_major_formatter(md.DateFormatter('%m/%d'))
    # plt.gca().xaxis.set_major_locator(md.AutoDateLocator())
    # # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    # plt.grid(color='gray', linestyle=':', linewidth=0.5)
    # plt.title(cow_name)
    # plt.legend()
    # plt.xlabel("Day")
    # plt.ylabel("Hour")
    # plt.tight_layout() 


    print("\nDone\n")
    plt.show()
