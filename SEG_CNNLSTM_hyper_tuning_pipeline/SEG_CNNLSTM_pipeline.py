import tensorflow as tf
from tensorflow import keras
import kerastuner
import numpy as np
import pandas as pd
import os
import json
import dill

from tensorflow.keras.layers import (
    TimeDistributed, 
    Dense, 
    Conv1D, 
    MaxPooling1D, 
    Bidirectional, 
    LSTM, 
    Dropout
)


## Global/Environmental Variables
## --------------------------------------------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)  # Off when Distributed Environment

dataset_name = "SEG_CNNLSTM"

static_params = dict()

static_params["PAST_HISTORY"] = 16
static_params["FUTURE_TARGET"] = 8
static_params["BATCH_SIZE"] = 1024
static_params["BUFFER_SIZE"] = 1000000
static_params["EPOCHS"] = 250

with open(dir_path + "/static/test_pipeline.pkl", "rb") as p:
    pipeline = dill.load(p)

static_params["VOCAB_SIZE"] = pipeline["sparse_category_encoder"].vocab_size



## Functions/Classes
## --------------------------------------------------------------------

def generate_timeseries(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, n_feature)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        #data.append(dataset[indices])
        labels.append(np.reshape(dataset[i:i+target_size], (target_size, 1)))
        #labels.append(dataset[i:i+target_size])
    return np.array(data), np.array(labels)

def build_model(hp):
    model = keras.Sequential()
    model.add(Conv1D(
        filters=hp.Int("conv1d_filters", min_value=16, max_value=128, step=8), 
        kernel_size=hp.Choice("conv1d_kernel_size", values=[3, 5, 7]), padding='causal', activation='relu'
        ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(hp.Float("conv1d_dropout", min_value=0.1, max_value=0.5, step=0.05)))
    model.add(Bidirectional(LSTM(units=hp.Int("layer_1_units", min_value=32, max_value=256, step=8), return_sequences=True)))
    model.add(Dropout(hp.Float("layer_1_dropout", min_value=0.1, max_value=0.5, step=0.05)))
    model.add(Bidirectional(LSTM(units=hp.Int("layer_2_units", min_value=32, max_value=256, step=8), return_sequences=True)))
    model.add(Dropout(hp.Float("layer_2_dropout", min_value=0.1, max_value=0.5, step=0.05)))
    model.add(TimeDistributed(Dense(static_params["VOCAB_SIZE"], activation="softmax")))
    
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )
    return model

def get_train_val_data():
    train_set = np.genfromtxt(dir_path + "/data/SEG_train_set.csv", delimiter="\n", dtype=np.float32) 
    x_train, y_train = generate_timeseries(train_set, 0, None, static_params["PAST_HISTORY"], static_params["FUTURE_TARGET"])
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().batch(static_params["BATCH_SIZE"]).shuffle(static_params["BUFFER_SIZE"])

    val_set = np.genfromtxt(dir_path + "/data/SEG_val_set.csv", delimiter="\n", dtype=np.float32) 
    x_val, y_val = generate_timeseries(val_set, 0, None, static_params["PAST_HISTORY"], static_params["FUTURE_TARGET"])
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.cache().batch(static_params["BATCH_SIZE"]).shuffle(static_params["BUFFER_SIZE"])

    return train_data, val_data

def get_test_data():
    test_set = np.genfromtxt(dir_path + "/data/SEG_test_set_original.csv", delimiter="\n", dtype=np.float32) 

    transformed_test_set = pipeline.transform(test_set)
    x_test, y_test = generate_timeseries(transformed_test_set, 0, None, static_params["PAST_HISTORY"], static_params["FUTURE_TARGET"])
    
    return x_test, y_test

def main():
    train_data, val_data = get_train_val_data()
    x_test, y_test = get_test_data()

    tuner = kerastuner.tuners.Hyperband(
        hypermodel=build_model,
        objective='val_accuracy',
        max_epochs=50,
        factor=2,
        hyperband_iterations=3,
        distribution_strategy=tf.distribute.MirroredStrategy(),
        tune_new_entries=True,
        directory="results",
        project_name=dataset_name
    )
    tuner.search(
        train_data,
        validation_data=val_data,
        epochs=100,
        callbacks=[keras.callbacks.EarlyStopping('val_accuracy')] #tensorboard_callback("logs/fit/" + timestamp)
    )
    
    with open(dir_path + "/results/tuner.pkl", "wb") as t:
        dill.dump(tuner, t)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    model.fit(train_data, validation_data=val_data, epochs=static_params["EPOCHS"], callbacks=[keras.callbacks.EarlyStopping('val_accuracy')])

    result = model.evaluate(x_test, y_test)

    with open(dir_path + "/results/evaluate.txt", 'w') as r:
        r.write("loss, accuracy\n")
        r.write("{}, {}".format(result[0], result[1]))

    model.save(dir_path + "/results/model.h5")

if __name__ == "__main__":
    main()