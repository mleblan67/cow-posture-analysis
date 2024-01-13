import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debug info

from numpy import mean
from numpy import std
from numpy import average
from sklearn.model_selection import train_test_split

from utils.load_data_utils import load_to_df, match_groundtruth_to_input
from models import CNN

# Train and evaluate a model
def build_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    model = CNN(n_timesteps, n_features, n_outputs)

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
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# Build and evaluate model 10 times to get a better idea of accuracy
def run_experiment(input_df, groundtruth_df, repeats=10):
    X, y = match_groundtruth_to_input(input_df, groundtruth_df)
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

def run_transfer_experiment(train_df, test_df, groundtruth_df, repeats=10):
    X_train, y_train = match_groundtruth_to_input(train_df, groundtruth_df)
    X_test, y_test = match_groundtruth_to_input(test_df, groundtruth_df)

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

def main():
    # load data
    input_filenames = ['converted_bits_data_T03.csv']
    output_filename = 'T03_groundtruths.csv'
    input_df, groundtruth_df = load_to_df(input_filenames, output_filename, prefix='converted_data/T03/')
    print("Data Loaded!")

    # Get full (16-bit) data only
    float16_df = input_df.loc[:, input_df.columns.intersection(['timestamp','accel_x_mps2', 'accel_y_mps2','accel_z_mps2'])]

    # Get full 14-bit data only
    float14_df = input_df.loc[:, input_df.columns.intersection(['timestamp','accel_x_14_float', 'accel_y_14_float','accel_z_14_float'])]

    # Get full 12-bit data only
    float12_df = input_df.loc[:, input_df.columns.intersection(['timestamp','accel_x_12_float', 'accel_y_12_float','accel_z_12_float'])]

    # Get full 10-bit data only
    float10_df = input_df.loc[:, input_df.columns.intersection(['timestamp','accel_x_10_float', 'accel_y_10_float','accel_z_10_float'])]

    # Get full 8-bit data only
    float8_df = input_df.loc[:, input_df.columns.intersection(['timestamp','accel_x_8_float', 'accel_y_8_float','accel_z_8_float'])]

    # Get full 6-bit data only
    float6_df = input_df.loc[:, input_df.columns.intersection(['timestamp','accel_x_6_float', 'accel_y_6_float','accel_z_6_float'])]


    '''
    Use 16 as train and test it with all other data sets
    '''

    # # 16 on 14
    # print("\n\nStarting 16 on 14 experiment")
    # run_transfer_experiment(float16_df, float14_df, groundtruth_df)

    # # 16 on 12
    # print("\n\nStarting 16 on 12 experiment")
    # run_transfer_experiment(float16_df, float12_df, groundtruth_df)

    # # 16 on 10
    # print("\n\nStarting 16 on 10 experiment")
    # run_transfer_experiment(float16_df, float10_df, groundtruth_df)

    # 16 on 8
    print("\n\nStarting 16 on 8 experiment")
    run_transfer_experiment(float16_df, float8_df, groundtruth_df)

    # 16 on 6
    print("\n\nStarting 16 on 6 experiment")
    run_transfer_experiment(float16_df, float6_df, groundtruth_df)

    '''
    70/30 split test

    # Original (16-bit)
    print("\n\nStarting 16-bit experiment")
    run_experiment(float16_df, groundtruth_df)

    # 14-bit
    print("\n\nStarting 14-bit experiment")
    run_experiment(float14_df, groundtruth_df)

    # 12-bit
    print("\n\nStarting 12-bit experiment")
    run_experiment(float12_df, groundtruth_df)

    # 10-bit
    print("\n\nStarting 10-bit experiment")
    run_experiment(float10_df, groundtruth_df)
    '''


def average_exp():
    # load data
    input_filenames = ['converted_bits_data_T03.csv']
    output_filename = 'Tag_3_ML_binary_4days.csv'
    input_df, groundtruth_df = load_to_df(input_filenames, output_filename, prefix='tag_3/')
    print("Data Loaded!")

    # Get full (16-bit) data only
    float16_df = input_df.loc[:, input_df.columns.intersection(['timestamp','accel_x_mps2', 'accel_y_mps2','accel_z_mps2'])]

    # Get full 14-bit data only
    float14_df = input_df.loc[:, input_df.columns.intersection(['timestamp','accel_x_14_float', 'accel_y_14_float','accel_z_14_float'])]

    # Get full 12-bit data only
    float12_df = input_df.loc[:, input_df.columns.intersection(['timestamp','accel_x_12_float', 'accel_y_12_float','accel_z_12_float'])]

    # Get full 10-bit data only
    float10_df = input_df.loc[:, input_df.columns.intersection(['timestamp','accel_x_10_float', 'accel_y_10_float','accel_z_10_float'])]

    # Get full 8-bit data only
    float8_df = input_df.loc[:, input_df.columns.intersection(['timestamp','accel_x_8_float', 'accel_y_8_float','accel_z_8_float'])]

    # Get full 6-bit data only
    float6_df = input_df.loc[:, input_df.columns.intersection(['timestamp','accel_x_6_float', 'accel_y_6_float','accel_z_6_float'])]
    
    X, y = match_groundtruth_to_input(float8_df, groundtruth_df)

    standing = []
    sitting = []

    for x_window, y_window in zip(X,y):
        if y_window == 1:
            standing.append(average(x_window, axis=0))
        else:
            sitting.append(average(x_window, axis=0))
    
    print("Standing avg: " + str(average(standing, axis=0)))
    print("Sitting avg: " + str(average(sitting, axis=0)))

    X, y = match_groundtruth_to_input(float6_df, groundtruth_df)

    standing = []
    sitting = []

    for x_window, y_window in zip(X,y):
        if y_window == 1:
            standing.append(average(x_window, axis=0))
        else:
            sitting.append(average(x_window, axis=0))
    
    print("Standing avg: " + str(average(standing, axis=0)))
    print("Sitting avg: " + str(average(sitting, axis=0)))


main()
# average_exp()