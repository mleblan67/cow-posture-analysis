import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Disable tensorflow debug info

from numpy import mean
from numpy import std
from numpy import asarray
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from pandas import merge
import pandas as pd
import datetime
import numpy as np
import pywt
import seaborn as sns

from utils.load_data_utils import load_to_df, create_rolling_window_data
from utils.features_utils import add_svm_feature
from models import CNN, CNN2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# Copied from internet to get rid of error
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Train and evaluate a model
def build_model(trainX, trainy, testX, testy):
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    verbose, epochs, batch_size = 0, 15, 64
    # wv_x, wv_y, wv_channels, num_classes = trainX.shape[1], trainX.shape[2], trainX.shape[3], trainy.shape[1]
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    # model = CNN2D(wv_x, wv_y, wv_channels, num_classes)
    model = CNN(n_timesteps, n_features, n_outputs)

    # fit network
    model.fit(trainX, trainy, epochs=epochs,
              batch_size=batch_size, verbose=verbose,
              validation_split=0.3, callbacks=[early_stop])
    
    # evaluate model
    _, accuracy, f1_score = model.evaluate(
        testX, testy, batch_size=batch_size, verbose=0)
    
    # get_confusion_matrix(model, testX, testy)
    class_acc = get_class_accuracies(model, testX, testy)
    
    return accuracy, f1_score, class_acc

def get_confusion_matrix(model, testX, testy):
    y_pred = model.predict(testX)

    # One hot encoding back to single int
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(testy, axis=1)

    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['1', '2', '3', '4','5','6','7'], yticklabels=['1', '2', '3', '4','5','6','7'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('raw_accel_conf_matrix.png')

def get_class_accuracies(model, testX, testy):
    y_pred = model.predict(testX)

    # One hot encoding back to single int
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(testy, axis=1)

    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

    num_classes = conf_matrix.shape[0]
    acc_per_class = np.zeros(num_classes)
    
    for i in range(num_classes):
        true_positive = conf_matrix[i, i]
        false_negative = np.sum(conf_matrix[i, :]) - true_positive
        acc_per_class[i] = true_positive / (true_positive + false_negative)
    
    # Print accuracy for each class
    # for i, acc in zip(range(num_classes), acc_per_class):
    #     print(f"Class {i+1}: {acc:.2f}")

    return acc_per_class

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

    return m

def run_exp(repeats=3):
    accel_data_prefix = 'converted_data/'
    uwb_data_prefix = 'location_data/'

    # The tag numbers we want to train on
    train_tags = [1,2,3,4,5,8,9,10]
    # The tag numbers we want to test on
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
        groundtruth_dir = 'behavior_analysis/individual_behaviors/'

        # Get all accelerometer sensor data files for this folder
        accel_filepaths = os.listdir(accel_data_dir)
        accel_filepaths = [accel_data_dir + file for file in accel_filepaths if file.startswith('sensor_data') and file.endswith('0725.csv')]
        accel_filepaths.sort() # Make sure they're in order for processing

        # Get all UWB sensor data files for this folder
        uwb_filepaths = os.listdir(uwb_data_dir)
        uwb_filepaths = [uwb_data_dir + file for file in uwb_filepaths if file.startswith('uwb_loc') and file.endswith('0725.csv')]
        uwb_filepaths.sort() # Make sure they're in order for processing
        
        # Get groundtruth path
        groundtruth_path = groundtruth_dir + 'C' + str(tag).zfill(2) + '_0725.csv'

        # Load in both sensor data
        accel_input_df, groundtruth_df = load_to_df(accel_filepaths, groundtruth_path)
        uwb_input_df, _ = load_to_df(uwb_filepaths, groundtruth_path)

        # Combine all sensor data together
        input_df = merge(accel_input_df, uwb_input_df, how='outer', on='timestamp')

        print(f"Loaded in tag {tag}")
        # Create sliding window
        X, y = create_rolling_window_data(uwb_input_df, groundtruth_df, window_size=10, stride=5)
        print(f"Created Sliding window for tag {tag} \n")

        # Add to array
        train_inputs.append(X)
        train_groundtruths.append(y)


    # Load in all the testing (just one day)
    for tag in test_tags:
        # Build folder path
        accel_data_dir = accel_data_prefix + 'T' + str(tag).zfill(2) + '/'
        uwb_data_dir = uwb_data_prefix + 'T' + str(tag).zfill(2) + '/'
        groundtruth_dir = 'behavior_analysis/individual_behaviors/'

        # Get all accelerometer sensor data files for this folder
        accel_filepaths = os.listdir(accel_data_dir)
        accel_filepaths = [accel_data_dir + file for file in accel_filepaths if file.startswith('sensor_data') and file.endswith('0725.csv')]
        accel_filepaths.sort() # Make sure they're in order for processing

        # Get all UWB sensor data files for this folder
        uwb_filepaths = os.listdir(uwb_data_dir)
        uwb_filepaths = [uwb_data_dir + file for file in uwb_filepaths if file.startswith('uwb_loc') and file.endswith('0725.csv')]
        uwb_filepaths.sort() # Make sure they're in order for processing
        
        # Get groundtruth path
        groundtruth_path = groundtruth_dir + 'C' + str(tag).zfill(2) + '_0725.csv'

        # Load in both sensor data
        accel_input_df, groundtruth_df = load_to_df(accel_filepaths, groundtruth_path)
        uwb_input_df, _ = load_to_df(uwb_filepaths, groundtruth_path)

        # Combine all sensor data together
        input_df = merge(accel_input_df, uwb_input_df, how='outer', on='timestamp')

        print(f"Loaded in tag {tag}")
        # Create sliding window
        X, y = create_rolling_window_data(uwb_input_df, groundtruth_df,window_size=10,stride=5)
        print(f"Created Sliding window for tag {tag} \n")

        # Add to array
        test_inputs.append(X)
        test_groundtruths.append(y)


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

        # repeat experiment
        accuracies = list()
        f1_scores = list()
        class_accuracies = np.zeros(7)
        print('      F1    \tAcc')
        for r in range(repeats):
            acc, f1_score, class_acc = build_model(X_train, y_train, X_test, y_test)
            acc = acc * 100.0
            f1_score = f1_score * 100.0
            class_acc = class_acc * 100.0
            print('>#%d:  %.3f\t%.3f' % (r+1, acc, f1_score))
            accuracies.append(acc)
            f1_scores.append(f1_score)
            class_accuracies += class_acc
        # model results
        print('      %.3f\t%.3f\n' % (mean(accuracies), mean(f1_scores)))
        # Class results
        class_accuracies = class_accuracies / repeats
        for class_i, acc in zip(range(7), class_accuracies):
            print('Class %d: %.3f' % (class_i+1, acc))


def run_wv_exp(repeats=3):
    accel_data_prefix = 'converted_data/'
    uwb_data_prefix = 'location_data/'

    # The tag numbers we want to train on
    train_tags = [1,2,3,4,5,8,9,10]
    # The tag numbers we want to test on
    test_tags = [7]

    # Array of all the training data we load in from each tag
    train_inputs = []
    train_groundtruths = []
    # Array of all the testing data we load in from each tag
    test_inputs = []
    test_groundtruths = []

    # Wavelet constants
    waveletname = "mexh"
    scales = range(1,201)


    # Load in all the training
    for tag in train_tags:
        # Build folder path
        accel_data_dir = accel_data_prefix + 'T' + str(tag).zfill(2) + '/'
        uwb_data_dir = uwb_data_prefix + 'T' + str(tag).zfill(2) + '/'
        groundtruth_dir = 'behavior_analysis/individual_behaviors/'

        # Get all accelerometer sensor data files for this folder
        accel_filepaths = os.listdir(accel_data_dir)
        accel_filepaths = [accel_data_dir + file for file in accel_filepaths if file.startswith('sensor_data') and file.endswith('0725.csv')]
        accel_filepaths.sort() # Make sure they're in order for processing

        # Get all UWB sensor data files for this folder
        uwb_filepaths = os.listdir(uwb_data_dir)
        uwb_filepaths = [uwb_data_dir + file for file in uwb_filepaths if file.startswith('uwb_loc') and file.endswith('0725.csv')]
        uwb_filepaths.sort() # Make sure they're in order for processing
        
        # Get groundtruth path
        groundtruth_path = groundtruth_dir + 'C' + str(tag).zfill(2) + '_0725.csv'

        # Load in both sensor data
        accel_input_df, groundtruth_df = load_to_df(accel_filepaths, groundtruth_path)
        uwb_input_df, _ = load_to_df(uwb_filepaths, groundtruth_path)

        # Combine all sensor data together
        input_df = merge(accel_input_df, uwb_input_df, how='outer', on='timestamp')

        print(f"Loaded in tag {tag}")
        # Create sliding window
        X, y = create_rolling_window_data(input_df, groundtruth_df,window_size=20, stride=10)
        print(f"Created Sliding window for tag {tag}")

        channels = X.shape[2]

        # Create wavelets
        wvs = np.ndarray((X.shape[0],200,200,channels))

        for wi in range(len(X)):
            # Transpose into X, Y, and Z channels
            window = X[wi].T
            for si in range(channels):
                signal = window[si]
                coeff, _ = pywt.cwt(signal, scales, waveletname, 1)
                wvs[wi, :, :, si] = coeff
        
        print(f"Created CWT for tag {tag} \n")

        # Add to array
        train_inputs.append(wvs)
        train_groundtruths.append(y)


    # Load in all the testing (just one day)
    for tag in test_tags:
        # Build folder path
        accel_data_dir = accel_data_prefix + 'T' + str(tag).zfill(2) + '/'
        uwb_data_dir = uwb_data_prefix + 'T' + str(tag).zfill(2) + '/'
        groundtruth_dir = 'behavior_analysis/individual_behaviors/'

        # Get all accelerometer sensor data files for this folder
        accel_filepaths = os.listdir(accel_data_dir)
        accel_filepaths = [accel_data_dir + file for file in accel_filepaths if file.startswith('sensor_data') and file.endswith('0725.csv')]
        accel_filepaths.sort() # Make sure they're in order for processing

        # Get all UWB sensor data files for this folder
        uwb_filepaths = os.listdir(uwb_data_dir)
        uwb_filepaths = [uwb_data_dir + file for file in uwb_filepaths if file.startswith('uwb_loc') and file.endswith('0725.csv')]
        uwb_filepaths.sort() # Make sure they're in order for processing
        
        # Get groundtruth path
        groundtruth_path = groundtruth_dir + 'C' + str(tag).zfill(2) + '_0725.csv'

        # Load in both sensor data
        accel_input_df, groundtruth_df = load_to_df(accel_filepaths, groundtruth_path)
        uwb_input_df, _ = load_to_df(uwb_filepaths, groundtruth_path)

        # Combine all sensor data together
        input_df = merge(accel_input_df, uwb_input_df, how='outer', on='timestamp')

        print(f"Loaded in tag {tag}")
        # Create sliding window
        X, y = create_rolling_window_data(input_df, groundtruth_df,window_size=20,stride=10)
        print(f"Created Sliding window for tag {tag}")

        channels = X.shape[2]

        # Create wavelets
        wvs = np.ndarray((X.shape[0],200,200,channels))

        for wi in range(len(X)):
            # Transpose into X, Y, and Z channels
            window = X[wi].T
            for si in range(channels):
                signal = window[si]
                coeff, _ = pywt.cwt(signal, scales, waveletname, 1)
                wvs[wi, :, :, si] = coeff

        print(f"Created CWT for tag {tag} \n")

        # Add to array
        test_inputs.append(wvs)
        test_groundtruths.append(y)

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


run_exp()