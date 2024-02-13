import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Disable tensorflow debug info

from numpy import mean
from numpy import std
from numpy import asarray
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from utils.load_data_utils import load_to_df, create_rolling_window_data
from utils.features_utils import add_svm_feature
from models import CNN, CNN_LSTM
from tensorflow.keras.utils import to_categorical

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
    
    plt.savefig('Accuracies.png')
    # return accuracy

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

    return m


def run_total_pooling_experiment(prefix='converted_data/', repeats=10):
    # The tag numbers we want to test on
    tags = range(1,11)
    # Array of all the data we load in from each tag
    inputs = []
    groundtruths = []

    # Load in all the data
    for tag in tags:
        # Build folder path
        data_dir = prefix + 'T' + str(tag).zfill(2) + '/'

        # Get all sensor data files for this folder
        filepaths = os.listdir(data_dir)
        filepaths = [data_dir + file for file in filepaths if file.startswith('sensor_data') and file.endswith('.csv')]
        filepaths.sort() # Make sure they're in order for processing

        # Get groundtruth path
        groundtruth_path = data_dir + 'T' + str(tag).zfill(2) + '_groundtruths.csv'

        # Load in the data
        input_df, groundtruth_df = load_to_df(filepaths, groundtruth_path)

        # Add to array
        inputs.append(input_df)
        groundtruths.append(groundtruth_df)

        print("Loaded Tag " + str(tag))

    # Create X and y window data
    Xs = []
    ys = []

    for tag, input_df, groundtruth_df in zip(tags, inputs, groundtruths):
        # Create rolling window
        X, y = create_rolling_window_data(input_df, groundtruth_df)

        # Add to data arrays to combine all data
        Xs += X
        ys += y

        print("Created sliding window for Tag " + str(tag))

    ys = to_categorical(ys)
    Xs = asarray(Xs)

    # Train/Test split for data
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.30)
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

def run_single_day_pooling_experiment(prefix='converted_data/', repeats=10):
    # The tag numbers we want to test on
    tags = range(1,11)
    # Array of all the data we load in from each tag
    inputs = []
    groundtruths = []

    # Load in all the data
    for tag in tags:
        # Build folder path
        data_dir = prefix + 'T' + str(tag).zfill(2) + '/'

        # Get all sensor data files for this folder
        # filepaths = os.listdir(data_dir)
        # filepaths = [data_dir + file for file in filepaths if file.startswith('sensor_data') and file.endswith('.csv')]
        # filepaths.sort() # Make sure they're in order for processing
        filepaths = [data_dir + 'sensor_data_T' + str(tag).zfill(2)+'_0725.csv']

        # Get groundtruth path
        groundtruth_path = data_dir + 'T' + str(tag).zfill(2) + '_groundtruths.csv'

        # Load in the data
        input_df, groundtruth_df = load_to_df(filepaths, groundtruth_path)

        # Add to array
        inputs.append(input_df)
        groundtruths.append(groundtruth_df)

        print("Loaded Tag " + str(tag))



    test_input = inputs.pop(5)
    test_groundtruth = groundtruths.pop()

    X_test, y_test = create_rolling_window_data(test_input, test_groundtruth)
    y_test = to_categorical(y_test)
    X_test = asarray(X_test)



    # Create X and y window data
    X_train = []
    y_train = []

    for tag, input_df, groundtruth_df in zip(tags, inputs, groundtruths):
        # Create rolling window
        X, y = create_rolling_window_data(input_df, groundtruth_df)

        # Add to data arrays to combine all data
        X_train += X
        y_train += y

        print("Created sliding window for Tag " + str(tag))

    y_train = to_categorical(y_train)
    X_train = asarray(X_train)

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
    summarize_results(scores)

def run_pooling_on_single_tag_single_day(prefix='converted_data/', repeats=10, sensor='sensor_data'):
    # The tag numbers we want to train on
    train_tags = [1,2,3,4,5,6,7,8,9,10]
    # The tag numbers we want to test on
    test_tags = [1,2,3,4,5,6,7,8,9,10]

    # Array of all the training data we load in from each tag
    train_inputs = []
    train_groundtruths = []

    # Array of all the testing data we load in from each tag
    test_inputs = []
    test_groundtruths = []

    # Load in all the training
    for tag in train_tags:
        # Build folder path
        data_dir = prefix + 'T' + str(tag).zfill(2) + '/'
        groundtruth_dir = 'converted_data/' + 'T' + str(tag).zfill(2) + '/'

        # Get all sensor data files for this folder
        filepaths = os.listdir(data_dir)
        filepaths = [data_dir + file for file in filepaths if file.startswith(sensor) and file.endswith('.csv')]
        filepaths.sort() # Make sure they're in order for processing
        

        # Get groundtruth path
        groundtruth_path = groundtruth_dir + 'T' + str(tag).zfill(2) + '_groundtruths.csv'

        # Load in the data
        input_df, groundtruth_df = load_to_df(filepaths, groundtruth_path)
        print(f"Loaded in tag {tag}")
        # Create sliding window
        X, y = create_rolling_window_data(input_df, groundtruth_df)
        print(f"Created Sliding window for tag {tag} \n")

        # Add to array
        train_inputs.append(X)
        train_groundtruths.append(y)

    # Load in all the testing (just one day)
    for tag in test_tags:
        # Build folder path
        data_dir = prefix + 'T' + str(tag).zfill(2) + '/'
        groundtruth_dir = 'converted_data/' + 'T' + str(tag).zfill(2) + '/'

        # Get all sensor data files for this folder
        filepaths = os.listdir(data_dir)
        filepaths = [data_dir + file for file in filepaths if file.startswith(sensor) and file.endswith('0725.csv')]
        filepaths.sort() # Make sure they're in order for processing
        

        # Get groundtruth path
        groundtruth_path = groundtruth_dir + 'T' + str(tag).zfill(2) + '_groundtruths.csv'

        # Load in the data
        input_df, groundtruth_df = load_to_df(filepaths, groundtruth_path)
        print(f"Loaded in tag {tag}")
        # Create sliding window
        X, y = create_rolling_window_data(input_df, groundtruth_df)
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
        X_train = []
        y_train = []

        for i in range(len(train_tags)):
            # Skip if this is the tag we're testing on
            if i == test_tag_i:
                continue

            X_train += train_inputs[i]
            y_train += train_groundtruths[i]
        
        # One-hot encoding
        y_train = to_categorical(y_train)
        X_train = asarray(X_train)

        # Prepare testing data
        X_test = test_inputs[test_tag_i]
        y_test = test_groundtruths[test_tag_i]

        y_test = to_categorical(y_test)
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

    

run_pooling_on_single_tag_single_day(prefix='location_data/',repeats=3,sensor='uwb_loc')
# run_pooling_on_single_tag_single_day()