import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Disable tensorflow debug info

from numpy import mean
from numpy import std
from numpy import asarray
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from pandas import merge
import pandas as pd
import datetime
import numpy as np

from utils.load_data_utils import load_to_df, create_rolling_window_data
from utils.features_utils import add_svm_feature
from models import CNN, CNN_LSTM
from tensorflow.keras.utils import to_categorical


# Copied from internet to get rid of error
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Train and evaluate a model
def build_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    n_batches = trainX.shape[0]

    model = CNN(n_timesteps, n_features, n_outputs)
    # model = CNN_LSTM(n_timesteps, n_features, n_outputs)
    # fit network
    model.fit(trainX, trainy, epochs=epochs,
              batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(
        testX, testy, batch_size=batch_size, verbose=0)
    
    return accuracy

def graph_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    n_batches = trainX.shape[0]

    model = CNN(n_timesteps, n_features, n_outputs)
    # model = CNN_LSTM(n_timesteps, n_features, n_outputs)
    # fit network
    model.fit(trainX, trainy, epochs=epochs,
              batch_size=batch_size, verbose=verbose)
    # evaluate model
    # _, accuracy = model.evaluate(
    #     testX, testy, batch_size=batch_size, verbose=0)
    
    predictions = model.predict(testX)

    testy = [val[0] for val in testy]
    predictions = [val[0] for val in predictions]

    plt.plot(testy, label = "Groundtruth", color='blue', linestyle='solid')
    plt.plot(predictions, label = "Prediction", color='red', linestyle='dashed')

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Standing")
    plt.yticks([0,1], ["Sitting", "Standing"])

    base = pd.to_datetime(1690261200.0, unit='s')
    timestamp_list = [base + datetime.timedelta(minutes=x*5) for x in range(0,200,20)]
    times = [f'{x.hour}:{str(x.minute).zfill(2)}' for x in timestamp_list]

    plt.xticks(range(0,200,20), times)

    plt.title('T07 07/25 UWB')

    plt.savefig('UWB_Accuracies.png')
    # return accuracy

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

    return m

def run_pooling_on_single_tag_single_day(repeats=3):
    accel_data_prefix = 'converted_data/'
    uwb_data_prefix = 'location_data/'

    # The tag numbers we want to train on
    train_tags = [1,2,3,4,5,6,8,9,10]
    # The tag numbers we want to test on
    # test_tags = [1,2,3,4,5,6,7,8,9,10]
    test_tags = [7]

    # Array of all the training data we load in from each tag
    train_inputs = []
    train_groundtruths = []
    # Array of all the testing data we load in from each tag
    test_inputs = []
    test_groundtruths = []


    # Load in all the training
    for tag in train_tags:
        # Build folder path
        accel_data_dir = accel_data_prefix + 'T' + str(tag).zfill(2) + '/'
        uwb_data_dir = uwb_data_prefix + 'T' + str(tag).zfill(2) + '/'
        groundtruth_dir = 'converted_data/' + 'T' + str(tag).zfill(2) + '/'

        # Get all accelerometer sensor data files for this folder
        accel_filepaths = os.listdir(accel_data_dir)
        accel_filepaths = [accel_data_dir + file for file in accel_filepaths if file.startswith('sensor_data') and file.endswith('.csv')]
        accel_filepaths.sort() # Make sure they're in order for processing

        # Get all UWB sensor data files for this folder
        # uwb_filepaths = os.listdir(uwb_data_dir)
        # uwb_filepaths = [uwb_data_dir + file for file in uwb_filepaths if file.startswith('uwb_loc') and file.endswith('.csv')]
        # uwb_filepaths.sort() # Make sure they're in order for processing
        
        # Get groundtruth path
        groundtruth_path = groundtruth_dir + 'T' + str(tag).zfill(2) + '_groundtruths.csv'

        # Load in both sensor data
        accel_input_df, groundtruth_df = load_to_df(accel_filepaths, groundtruth_path)
        # uwb_input_df, groundtruth_df = load_to_df(uwb_filepaths, groundtruth_path)

        # Combine all sensor data together
        # input_df = merge(accel_input_df, uwb_input_df, how='outer', on='timestamp')

        print(f"Loaded in tag {tag}")
        # Create sliding window
        X, y = create_rolling_window_data(accel_input_df, groundtruth_df)
        print(f"Created Sliding window for tag {tag} \n")

        # Add to array
        train_inputs.append(X)
        train_groundtruths.append(y)

        # Manage memory
        del [accel_input_df, groundtruth_df]
        gc.collect()


    # Load in all the testing (just one day)
    for tag in test_tags:
        # Build folder path
        accel_data_dir = accel_data_prefix + 'T' + str(tag).zfill(2) + '/'
        uwb_data_dir = uwb_data_prefix + 'T' + str(tag).zfill(2) + '/'
        groundtruth_dir = 'converted_data/' + 'T' + str(tag).zfill(2) + '/'

        # Get all accelerometer sensor data files for this folder
        accel_filepaths = os.listdir(accel_data_dir)
        accel_filepaths = [accel_data_dir + file for file in accel_filepaths if file.startswith('sensor_data') and file.endswith('0725.csv')]
        accel_filepaths.sort() # Make sure they're in order for processing

        # Get all UWB sensor data files for this folder
        # uwb_filepaths = os.listdir(uwb_data_dir)
        # uwb_filepaths = [uwb_data_dir + file for file in uwb_filepaths if file.startswith('uwb_loc') and file.endswith('0725.csv')]
        # uwb_filepaths.sort() # Make sure they're in order for processing
        
        # Get groundtruth path
        groundtruth_path = groundtruth_dir + 'T' + str(tag).zfill(2) + '_groundtruths.csv'

        # Load in both sensor data
        accel_input_df, groundtruth_df = load_to_df(accel_filepaths, groundtruth_path)
        # uwb_input_df, groundtruth_df = load_to_df(uwb_filepaths, groundtruth_path)

        # Combine all sensor data together
        # input_df = merge(accel_input_df, uwb_input_df, how='outer', on='timestamp')

        print(f"Loaded in tag {tag}")
        # Create sliding window
        X, y = create_rolling_window_data(accel_input_df, groundtruth_df)
        print(f"Created Sliding window for tag {tag} \n")

        # Add to array
        test_inputs.append(X)
        test_groundtruths.append(y)

        # Manage memory
        del [accel_input_df, groundtruth_df]
        gc.collect()


    accuracies = []
    # Loop through every test tag to train on all other data and test on this one
    for test_tag_i in range(len(test_tags)):
        print(f"Testing on tag {test_tags[test_tag_i]}")

        # Prepare training data
        # Combine all training data
        X_train = np.array([])
        y_train = np.array([])

        for i in range(len(train_tags)):
            # Skip if this is the tag we're testing on
            if train_tags[i] == test_tags[test_tag_i]:
                continue

            # X_train += train_inputs[i]
            # y_train += train_groundtruths[i]
            if X_train.shape[0] == 0:
                X_train = np.copy(train_inputs[i])
            else:
                X_train = np.vstack([X_train, train_inputs[i]])
            
            if y_train.shape[0] == 0:
                y_train = np.copy(train_groundtruths[i])
            else:
                y_train = np.concatenate((y_train, train_groundtruths[i]))
                

        # One-hot encoding
        y_train = to_categorical(y_train)

        # Prepare testing data
        X_test = test_inputs[test_tag_i]
        y_test = test_groundtruths[test_tag_i]

        y_test = to_categorical(y_test)

        # Train/Test split for data
        print("Training data shape: (X) (y)")
        print(X_train.shape, y_train.shape)
        print("Testing data shape: (X) (y)")
        print(X_test.shape, y_test.shape)

        print('Data loaded! Ready to train')

        # graph exp
        graph_model(X_train, y_train, X_test, y_test)

        # repeat experiment
        scores = list()
        for r in range(repeats):
            score = build_model(X_train, y_train, X_test, y_test)
            score = score * 100.0
            print('>#%d: %.3f' % (r+1, score))
            scores.append(score)
        # summarize results
        m = summarize_results(scores)
        accuracies.append(m)

    # print(f"OVERALL ACCURACY WAS {mean(accuracies)}")


def run_pooling_on_single_tag_single_day_behavior(repeats=3):
    accel_data_prefix = 'converted_data/'
    groundtruth_data_prefix = 'behavior_analysis/combined_cow_data/'

    # The tag numbers we want to train on
    train_tags = [1,2,3,4,5,6,7,8,9,10]
    # The tag numbers we want to test on
    test_tags = [1,2,3,4,5,6,7,8,9,10]
    # test_tags = [7]

    # Array of all the training data we load in from each tag
    train_inputs = []
    train_groundtruths = []

    # Array of all the testing data we load in from each tag
    test_inputs = []
    test_groundtruths = []

    # Load in all the training
    for tag in train_tags:
        # Build folder path
        accel_data_dir = accel_data_prefix + 'T' + str(tag).zfill(2) + '/'
        groundtruth_dir = groundtruth_data_prefix + 'T' + str(tag).zfill(2) + '/'

        # Get all accelerometer sensor data files for this folder
        accel_filepaths = os.listdir(accel_data_dir)
        accel_filepaths = [accel_data_dir + file for file in accel_filepaths if file.startswith('sensor_data') and file.endswith('.csv')]
        accel_filepaths.sort() # Make sure they're in order for processing
        
        # Get groundtruth path
        groundtruth_path = groundtruth_dir + 'T' + str(tag).zfill(2) + '_groundtruth.csv'

        # Load in both sensor data
        accel_input_df, groundtruth_df = load_to_df(accel_filepaths, groundtruth_path)

        print(f"Loaded in tag {tag}")
        # Create sliding window
        X, y = create_rolling_window_data(accel_input_df, groundtruth_df)
        print(f"Created Sliding window for tag {tag} \n")

        # Add to array
        print(asarray(X).shape)
        train_inputs.append(X)
        train_groundtruths.append(y)

        # Manage memory
        del [accel_input_df, groundtruth_df]
        gc.collect()


    # Load in all the testing (just one day)
    for tag in test_tags:
        # Build folder path
        accel_data_dir = accel_data_prefix + 'T' + str(tag).zfill(2) + '/'
        groundtruth_dir = groundtruth_data_prefix + 'T' + str(tag).zfill(2) + '/'

        # Get all accelerometer sensor data files for this folder
        accel_filepaths = os.listdir(accel_data_dir)
        accel_filepaths = [accel_data_dir + file for file in accel_filepaths if file.startswith('sensor_data') and file.endswith('0725.csv')]
        accel_filepaths.sort() # Make sure they're in order for processing
        
        # Get groundtruth path
        groundtruth_path = groundtruth_dir + 'T' + str(tag).zfill(2) + '_groundtruth.csv'

        # Load in both sensor data
        accel_input_df, groundtruth_df = load_to_df(accel_filepaths, groundtruth_path)

        print(f"Loaded in tag {tag}")
        # Create sliding window
        X, y = create_rolling_window_data(accel_input_df, groundtruth_df)
        print(f"Created Sliding window for tag {tag} \n")

        # Add to array
        test_inputs.append(X)
        test_groundtruths.append(y)

        # Manage memory
        del [accel_input_df, groundtruth_df]
        gc.collect()


    accuracies = []
    # Loop through every test tag to train on all other data and test on this one
    for test_tag_i in range(len(test_tags)):
        print(f"Testing on tag {test_tags[test_tag_i]}")

        # Prepare training data
        # Combine all training data
        X_train = []
        y_train = []

        for i in range(len(train_tags)):
            # Skip if this is the tag we're testing on
            if i == test_tag_i:
                continue

            X_train += train_inputs[i]
            y_train += train_groundtruths[i]
        
        # One-hot encoding
        y_train = to_categorical(y_train, num_classes=5)
        X_train = asarray(X_train)

        # Prepare testing data
        X_test = test_inputs[test_tag_i]
        y_test = test_groundtruths[test_tag_i]

        y_test = to_categorical(y_test, num_classes=5)
        X_test = asarray(X_test)

        # Train/Test split for data
        print("Training data shape: (X) (y)")
        print(X_train.shape, y_train.shape)
        print("Testing data shape: (X) (y)")
        print(X_test.shape, y_test.shape)

        print('Data loaded! Ready to train')

        # graph exp
        # graph_model(X_train, y_train, X_test, y_test)

        # repeat experiment
        scores = list()
        for r in range(repeats):
            score = build_model(X_train, y_train, X_test, y_test)
            score = score * 100.0
            print('>#%d: %.3f' % (r+1, score))
            scores.append(score)
        # summarize results
        m = summarize_results(scores)
        accuracies.append(m)

    print(f"OVERALL ACCURACY WAS {mean(accuracies)}")
    

run_pooling_on_single_tag_single_day()