import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Disable tensorflow debug info

from numpy import mean
from numpy import std
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from pandas import merge

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
    plt.xlabel("Time Frame")
    plt.ylabel("Standing")

    plt.title('CNN Prediction vs Groundtruth on T07 UWB Data on 07/25')
    
    plt.savefig('Accuracies.png')
    # return accuracy

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

    return m


'''
Hyperparameters: sliding window size, sliding window strides
'''
def tune_hyperparameters(repeats=2):
    accel_data_prefix = 'converted_data/'

    window_sizes = [25, 30]
    strides = [5,10]

    for window_size in window_sizes:
        for stride in strides:
            print(f"WINDOW SIZE: {window_size}")
            print(f"STRIDE: {stride} \n")
            # The tag numbers we want to train on
            train_tags = [1,2,3,5,6,8,9,10]
            # The tag numbers we want to test on
            # test_tags = [1,2,3,4,5,6,7,8,9,10]
            test_tags = [7]
            # The tag numbers we want to validate with
            validation_tags = [4]

            # Array of all the training data we load in from each tag
            train_inputs = []
            train_groundtruths = []
            # Array of all the testing data we load in from each tag
            test_inputs = []
            test_groundtruths = []
            # Array of all the testing data we load in from each tag
            validation_inputs = []
            validation_groundtruths = []

            # Load in all the training
            for tag in train_tags:
                # Build folder path
                tag_name = 'T' + str(tag).zfill(2)
                accel_data_dir = accel_data_prefix + tag_name + '/'
                groundtruth_dir = 'converted_data/' + tag_name + '/'

                # Get all accelerometer sensor data files for this folder
                accel_filepaths = os.listdir(accel_data_dir)
                accel_filepaths = [accel_data_dir + file for file in accel_filepaths if file.startswith('sensor_data') and file.endswith('.csv')]
                accel_filepaths.sort() # Make sure they're in order for processing

                # Only train on half the days
                accel_filepaths = accel_filepaths = accel_filepaths[:(int(len(accel_filepaths)/2)+1)]
                
                # Get groundtruth path
                groundtruth_path = groundtruth_dir + 'T' + str(tag).zfill(2) + '_groundtruths.csv'

                # Load in both sensor data
                accel_input_df, groundtruth_df = load_to_df(accel_filepaths, groundtruth_path)

                print(f"Loaded in tag {tag}")
                # Create sliding window
                X, y = create_rolling_window_data(accel_input_df, groundtruth_df, window_size=window_size, stride=stride)
                print(f"Created Sliding window for tag {tag}")

                # Add to array
                train_inputs.append(X)
                train_groundtruths.append(y)

                # Manage memory
                del [accel_input_df, groundtruth_df]
                gc.collect()


            # Load in all the testing (just one day)
            for tag in test_tags:
                break
                # Build folder path
                accel_data_dir = accel_data_prefix + 'T' + str(tag).zfill(2) + '/'
                groundtruth_dir = 'converted_data/' + 'T' + str(tag).zfill(2) + '/'

                # Get all accelerometer sensor data files for this folder
                accel_filepaths = os.listdir(accel_data_dir)
                accel_filepaths = [accel_data_dir + file for file in accel_filepaths if file.startswith('sensor_data') and file.endswith('0725.csv')]
                accel_filepaths.sort() # Make sure they're in order for processing
                
                # Get groundtruth path
                groundtruth_path = groundtruth_dir + 'T' + str(tag).zfill(2) + '_groundtruths.csv'

                # Load in both sensor data
                accel_input_df, groundtruth_df = load_to_df(accel_filepaths, groundtruth_path)

                print(f"Loaded in tag {tag}")
                # Create sliding window
                X, y = create_rolling_window_data(accel_input_df, groundtruth_df, window_size=window_size, stride=stride)
                print(f"Created Sliding window for tag {tag}")

                # Add to array
                test_inputs.append(np.array(X))
                test_groundtruths.append(np.array(y))

                # Manage memory
                del [accel_input_df, groundtruth_df]
                gc.collect()

            
            # Load in all the testing (just one day)
            for tag in validation_tags:
                # Build folder path
                accel_data_dir = accel_data_prefix + 'T' + str(tag).zfill(2) + '/'
                groundtruth_dir = 'converted_data/' + 'T' + str(tag).zfill(2) + '/'

                # Get all accelerometer sensor data files for this folder
                accel_filepaths = os.listdir(accel_data_dir)
                accel_filepaths = [accel_data_dir + file for file in accel_filepaths if file.startswith('sensor_data') and file.endswith('0725.csv')]
                accel_filepaths.sort() # Make sure they're in order for processing
                
                # Get groundtruth path
                groundtruth_path = groundtruth_dir + 'T' + str(tag).zfill(2) + '_groundtruths.csv'

                # Load in both sensor data
                accel_input_df, groundtruth_df = load_to_df(accel_filepaths, groundtruth_path)

                print(f"Loaded in tag {tag}")
                # Create sliding window
                X, y = create_rolling_window_data(accel_input_df, groundtruth_df, window_size=window_size, stride=stride)
                print(f"Created Sliding window for tag {tag}")

                # Add to array
                validation_inputs.append(X)
                validation_groundtruths.append(y)

                # Manage memory
                del [accel_input_df, groundtruth_df]
                gc.collect()


            # Loop through every test tag to train on all other data and test on this one
            for val_tag_i in range(len(validation_tags)):
                print(f"Validating on tag {validation_tags[val_tag_i]}")

                # Prepare training data
                # Combine all training data
                X_train = np.array([])
                y_train = np.array([])

                for i in range(len(train_tags)):
                    # Skip if this is the tag we're testing on
                    if train_tags[i] == validation_tags[val_tag_i]:
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

                # Prepare validation data
                X_val = validation_inputs[val_tag_i]
                y_val = validation_groundtruths[val_tag_i]

                y_val = to_categorical(y_val)

                # Train/Test split for data
                print("Training data shape: (X) (y)")
                print(X_train.shape, y_train.shape)
                print("Testing data shape: (X) (y)")
                print(X_val.shape, y_val.shape)

                print('Data loaded! Ready to train')

                # graph exp
                # graph_model(X_train, y_train, X_test, y_test)

                # repeat experiment
                scores = list()
                for r in range(repeats):
                    score = build_model(X_train, y_train, X_val, y_val)
                    score = score * 100.0
                    print('>#%d: %.3f' % (r+1, score))
                    scores.append(score)
                # summarize results
                m = summarize_results(scores)
                print(m)
                print("\n")

tune_hyperparameters()