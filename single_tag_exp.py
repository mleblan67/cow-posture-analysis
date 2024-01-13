import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Disable tensorflow debug info

import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from utils.load_data_utils import load_to_df, prepare_uwb_data, create_rolling_window_data
from utils.features_utils import add_mean_xyz_feature, add_svm_feature, add_sma_feature, add_pitch_roll_features, add_uwb_features
from models import CNN, CNN_LSTM

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
    # TODO: graph accuracies
    return accuracy

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# Build and evaluate model 10 times to get a better idea of accuracy
def run_experiment(repeats=10):
    # load data
    input_filenames = ['sensor_data_T03S0728.csv', 'sensor_data_T03S0729.csv']
    output_filename = 'T03_groundtruths.csv'

    input_df, groundtruth_df = load_to_df(input_filenames, output_filename, prefix='converted_data/T03')
    # Add features
    # input_df = add_mean_xyz_feature(input_df)
    input_df = add_svm_feature(input_df)
    # input_df = add_sma_feature(input_df)
    # input_df = add_pitch_roll_features(input_df)
    # input_df = add_uwb_features(input_df, uwb_df)

    # X, y = match_groundtruth_to_input(input_df, groundtruth_df)
    X, y = create_rolling_window_data(input_df, groundtruth_df)

    # Final data prep
    # One hot encoding for output labels 
    y = to_categorical(y)
    X = np.asarray(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
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
    summarize_results(scores)