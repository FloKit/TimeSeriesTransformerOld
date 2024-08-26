import numpy as np
import tensorflow as tf
import gym
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from constants import *
import collections
    
def sample_data (episodes=10000, seed=0):
    env = gym.make ("CartPole-v1")
    env.np_random.seed(0)
    
    ### CREATE EMPTY Pandas dataset
    transitions = []

    ### SAMPLE DATA
    for episode in range (episodes):
        #print ("Start of episode %d" % episode)
        obs = env.reset()
        step = 0
        done = False

        while step < 500 and not done:
            step += 1
            action = env.action_space.sample()

            transitions.append({CART_POS:obs[0], CART_VEL:obs[1], 
                             PEND_POS:obs[2], PEND_VEL:obs[3],
                             EPISODE:episode, STEP:step, ACTION:action})

            obs, reward, done, _ = env.step(action)

        #print ("  --> finished after %d steps" % step)
        
    return pd.DataFrame(transitions)

#############################################
#### HELPER FUNCTIONS FOR PATTERN GENERATION
#############################################
def create_training_data(data, input_col, target_col, window_size=1, training_pattern_percent=0.7):

    data_train = data

    mean_in, std_in = mean_and_std(input_col, data_train)
    mean_out, std_out = mean_and_std(target_col, data_train)
    #data_plot.plot_hist_df(data_train, input_col)
    #data_plot.plot_timeseries_df(data_train, input_col)
    print(f"mean in = {mean_in}" )
    print(f"std in = {std_in}")
    print(f"mean out =  {mean_out}")
    print(f"std out = {std_out}")

    grouped = data_train.groupby(['episode'])

    inputs_all = []
    labels_all = []

    for g in grouped:
        # be sure that data inside a group is not shuffled # not sure if needed
        g = g[1].sort_values(by='step')

        past_history = window_size   # t-3, t-2, t-1, t
        future_target = 0  # t+1
        STEP = 1 # no subsampling of rows in data, e.g. only every i'th row

        # use pandas.DataFrame.values in order to get an numpy array from an pandas.DataFrame object

        inputs, labels = multivariate_data(dataset=g[input_col][:].values, target=g[target_col][:].values,
                                        start_index=0, end_index=g[input_col][:].values.shape[0]-future_target,
                                        history_size=past_history, target_size=future_target, step=STEP,
                                        single_step=True)

        ## Append data to whole set of patterns
        for i in range (0, len(inputs)):
            inputs_all.append(inputs[i])
            labels_all.append(labels[i])
  
    length = len(inputs_all)

    c = list(zip(inputs_all, labels_all))
    np.random.shuffle(c)
    inputs_all, labels_all = zip(*c)

    split = int(training_pattern_percent * length)

    inputs_all = np.array(inputs_all)
    labels_all = np.array(labels_all)

    return ((inputs_all[0:split], labels_all[0:split]), (inputs_all[split:], labels_all[split:])), mean_in, std_in, mean_out, std_out


def mean_and_std(columns, data):
    mean = np.zeros(len(columns))
    std = np.zeros(len(columns))
    index = 0
    for c in columns:
        mean[index], std[index] = get_normalizations(data[c])
        index = index + 1
    return mean, std

def get_normalizations(data):
    mean = data.mean()
    std = data.std()
    return mean, std

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
       end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)



def prepare_data(df, input_col, target_col, window_size, training_batch_size=50, validation_batch_size=50, training_pattern_percent=0.7):
    
    ###################
    ## PREPARE DATASET
    ###################
    ((x_train_multi, y_train_multi), (x_val_multi, y_val_multi)), mean_in, std_in, mean_out, std_out = \
                                    create_training_data(df, input_col, target_col, window_size=window_size,
                                                        training_pattern_percent=training_pattern_percent)

    print('trainData: Single window of past history : {}'.format(x_train_multi[0].shape))
    print('trainData: Single window of future : {}'.format(y_train_multi[1].shape))
    print('valData: Single window of past history : {}'.format(x_val_multi[0].shape))
    print('valData: Single window of future : {}'.format(y_val_multi[1].shape))
    print('trainData: number of trainingsexamples: {}'.format(x_train_multi.shape))
    print('valData: number of trainingsexamples: {}'.format(x_val_multi.shape))

    train_data = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    #train_data = train_data.cache().shuffle(max_training_pattern).batch(training_batch_size).repeat()
    train_data = train_data.shuffle(x_train_multi.shape[0]).batch(training_batch_size).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data = val_data.batch(validation_batch_size).repeat()
    input_shape = x_train_multi[0].shape[-2:]
    return train_data, val_data, input_shape, mean_in, std_in, mean_out, std_out, x_train_multi, y_train_multi


def create_quantiled_buckets(df: pd.DataFrame, quantile_step: float) -> pd.DataFrame:
    quantiles = np.arange(0, 100 + quantile_step, quantile_step)
    quantiles = np.clip(quantiles, 0, 100)
    
    def calculate_quantiles(column):
        return [df[column].quantile(q/100) for q in quantiles]

    def bucketize(value, quantile_values):
        for i, v in enumerate(quantile_values):
            if value <= v:
                return i
        return len(quantile_values)  # Handle edge case for the highest bucket
    
    quantile_values_map = {
        CART_POS: calculate_quantiles(CART_POS),
        CART_VEL: calculate_quantiles(CART_VEL),
        PEND_POS: calculate_quantiles(PEND_POS),
        PEND_VEL: calculate_quantiles(PEND_VEL)
    }

    def apply_bucketization(column):
        return df[column].apply(lambda x: bucketize(x, quantile_values_map[column]))

    bucket_df = df.copy()
    bucket_df[CART_POS] = apply_bucketization(CART_POS)
    bucket_df[CART_VEL] = apply_bucketization(CART_VEL)
    bucket_df[PEND_POS] = apply_bucketization(PEND_POS)
    bucket_df[PEND_VEL] = apply_bucketization(PEND_VEL)

    return bucket_df

def plot_quantiles(df, var_name, bucket_size, title):
    plt.figure(figsize=(12,5))
    plt.hist(df[var_name], bins=50, alpha=0.5, label=var_name)

    quantile = 0

    while quantile <= 100:
        plt.axvline(df[var_name].quantile(quantile/100), color='b', linestyle='dashed', linewidth=1)
        quantile += bucket_size

    plt.grid()
    plt.title(title)
    plt.show()

def create_training_val_data(df, input_cols: List[str], target_cols: List[str], window_size: int, training_pattern_percent: float) -> tuple:

    train_data, val_data, input_shape, mean_in, std_in, mean_out, std_out , x_train_multi, y_train_multi =  \
                prepare_data(df, input_cols, target_cols, window_size=window_size, training_pattern_percent=training_pattern_percent)

    print ("Input-Shape: ", input_shape)

    return train_data, val_data, input_shape, mean_in, std_in, mean_out, std_out, x_train_multi, y_train_multi

def evaluate_transformer(transformer_model, dfEval, window_size, output_min, output_max):
    # FIFO-buffer that keeps the neural state
    stateBuffer = collections.deque(maxlen=window_size)

    # outputs of neural network will be stored here
    transitions = []

    for i in range (len(dfEval)): 
                                
        # estimation of first state
        if i < window_size: 
            state_data = np.float32([dfEval[CART_POS].values[i], dfEval[CART_VEL].values[i],
                                dfEval[PEND_POS].values[i], dfEval[PEND_VEL].values[i],
                                dfEval[ACTION].values[i]])
            stateBuffer.append(state_data)
            #print ("Filling initState: %s" % state_data)
        
        # predict successor state
        else: 
            
            ###########################
            # recall of neural network
            ###########################
            state = np.array([list(stateBuffer)])
            if i==5:
                print (state)
            
            netOutput = transformer_model.predict(np.float32(state))[0]
            
            # clip output to observed data bounds
            netOutput = np.clip(netOutput, output_min, output_max)
            
            # check if value bound was hit
            if np.any(netOutput == output_min) or np.any(netOutput == output_max):
                print ("Bound-hit at step: ", i, " => terminating further evaluation")
                break
            
            # append plotting data
            transitions.append ({
                CART_POS:netOutput[0], CART_VEL:netOutput[1],
                PEND_POS:netOutput[2], PEND_VEL:netOutput[3]
            })
            
            # update RNN state
            stateBuffer.append(np.float32([netOutput[0], netOutput[1], 
                                        netOutput[2], netOutput[3], 
                                        dfEval[ACTION].values[i]]))
            
    dfNet = pd.DataFrame(transitions)

    return dfNet

def plot_evaluation(dfNet, dfEval, window_size):
    fig, axs = plt.subplots (5, 1, figsize=(10,10))

    fields = [CART_POS, CART_VEL, PEND_POS, PEND_VEL]

    for i in range (len(fields)):
        f = fields[i]
        axs[i].plot(range (len(dfNet)), dfEval[f].values[window_size:window_size+len(dfNet)], label=f)
        axs[i].plot(range (len(dfNet)), dfNet[f].values, label="prediction - transformer", ls="--")
        axs[i].grid()
        axs[i].legend(loc="best")
        
    axs[4].plot(range (len(dfNet)), dfEval[ACTION].values[window_size:window_size+len(dfNet)], label=ACTION)
    axs[4].grid()
    axs[4].legend(loc="best")

    plt.show()