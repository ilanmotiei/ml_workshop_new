import pandas as pd
import json
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import Dense, Dropout, GRU
import matplotlib.pyplot as plt

path_in = ""
path_out = "\\Results\\"
VERBOSE = True
K_ARRAY = [i for i in range(1, 14)] + [7*i for i in range(2, 13)]


def train_test_separation(df, stations):
    df = df.drop(
        ["Leq A", "Leq A Day Max", "Leq A Day Min", "Leq A Max", "Leq A Min", "Leq A Night Max", "Leq A Night Min",
         "Leq C", "Leq C Day", "Leq C Day Max", "Leq C Day Min", "Leq C Max", "Leq C Min", "Leq C Night",
         "Leq C Night Max", "Leq C Night Min", "Lpeak", "Lpeak-day", "Lpeak-night"],
        axis=1)  # drop non-relevant columns
    # remaining: station_name, published_dt, Leq A Day, Leq A Night
    train_concat = pd.DataFrame()
    # train_data = dict()
    test_data = dict()
    for station in stations:
        curr = df[df["station_name"] == station]
        curr = curr.drop(["station_name", "published_dt"], axis=1)
        curr = curr.fillna(method="ffill")  # imputation
        curr = curr.dropna(axis=0, how="any", subset=["Leq A Day", "Leq A Night"])
        train_concat = train_concat.append(curr[:4 * len(curr) // 5])
        # train_data[station] = curr[:4 * len(curr) // 5]
        test_data[station] = curr[4 * len(curr) // 5:]
    train_concat = train_concat.fillna(method="ffill")
    #train_concat = train_concat.drop("published_dt", axis=1)
    return train_concat, test_data


def k_vectorize(test_data, k):
    new_test_data = dict()
    for station in test_data.keys():
        train_X = list()
        train_Y = list()
        for i in range(len(test_data[station]) - k):
            v = test_data[station].iloc[i:i + k]
            train_X.append(v)
            train_Y.append(test_data[station].iloc[i + k])
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)
        new_test_data[station] = (train_X, train_Y)
    return new_test_data


def Univar_ARIMA(train_concat, test_data, k):
    model_pair = list()
    for col_name in ["Leq A Day", "Leq A Night"]:
        arima_model = ARIMA(train_concat[col_name], order=(k, 0, 0)) # TODO - what about order (k, 0, k)?
        model = arima_model.fit()
        model_pair.append(model)
    train_mse_day = model_pair[0].mse
    train_mse_night = model_pair[1].mse
    test_mse_day = 0.0
    test_mse_night = 0.0
    total_div = 0
    for station in test_data.keys():
        Y = test_data[station][1]
        new_res = model_pair[0].extend(Y[:, 0])
        test_mse_day += new_res.mse
        new_res = model_pair[0].extend(Y[:, 1])
        test_mse_night += new_res.mse
        total_div += 1
    return k, train_mse_day, train_mse_night, test_mse_day / total_div, test_mse_night / total_div


def Multivar_ARIMA(train_concat, test_data, k):
    VAR_model = VAR(endog=train_concat)
    model = VAR_model.fit(maxlags=k)
    train_mse_day = model.mse(1)[0][0][0]
    train_mse_night = model.mse(1)[0][1][1]
    test_mse_day = 0.0
    test_mse_night = 0.0
    total_div = 0
    for station in test_data.keys():
        for x, y in zip(test_data[station][0], test_data[station][1]):
            sq_err = np.square(model.forecast(x, 1)[0] - y)
            test_mse_day += sq_err[0]
            test_mse_night += sq_err[1]
            total_div += 1

    return k, train_mse_day, train_mse_night, test_mse_day/total_div, test_mse_night/total_div


def GRU_Train(train_concat, test_data, k):  # TODO
    scaler = MinMaxScaler().fit(train_concat)
    scaled_train = scaler.transform(train_concat)

    train_X = list()
    train_Y = list()
    for i in range(len(scaled_train) - k):
        v = scaled_train[i:i + k]
        train_X.append(v)
        train_Y.append(scaled_train[i + k])
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    model = Sequential()
    units = 64*2
    # Input layer
    model.add(GRU(units=units, return_sequences=True,
                  input_shape=[train_X.shape[1], train_X.shape[2]]))
    model.add(Dropout(0.2))
    # Hidden layer
    model.add(GRU(units=units))
    model.add(Dropout(0.2))
    model.add(Dense(units=2))
    # Compile model
    model.compile(optimizer="adam", loss="mse")

    early_stop = keras.callbacks.EarlyStopping(monitor= "val_loss", patience = 10)
    history = model.fit(train_X, train_Y, epochs=30,
                        validation_split=0.2,
                        batch_size=16, shuffle=False,
                        callbacks=[early_stop])

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation Loss for k=' + str(k))
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
    plt.savefig(path_out + 'k=' + str(k) + '.jpg', format='jpg', dpi=1000)

    square_errors = np.square(scaler.inverse_transform(model.predict(train_X)) - scaler.inverse_transform(train_Y))
    train_mse = (square_errors[:, 0].mean(), square_errors[:, 1].mean())

    def GRU_forecast(past_values, step, get_scaler=False): #TODO - implement step
        if get_scaler:
            return scaler
        return scaler.inverse_transform(model.predict(past_values))

    return GRU_forecast, train_mse

"""
def test_eval(forecast_func, test_data, k):  # TODO
    mseDay = 0.0
    mseNight = 0.0
    total_div = 0
    scaler = forecast_func(0, 0, get_scaler=True) # TODO - hotfix
    for station in test_data.keys():
        curr = test_data[station].drop("published_dt", axis=1)
        for i in range(len(curr) - k):
            v = scaler.transform(curr.iloc[i:i + k]) # TODO - the transform is a hotfix, move to GRU forecast
            v = np.expand_dims(v, axis=0)
            mseDay += np.square(forecast_func(v, 1)[0][0] - curr["Leq A Day"].iloc[i + k])
            mseNight += np.square(forecast_func(v, 1)[0][1] - curr["Leq A Night"].iloc[i + k])
        total_div += len(curr) - k
        pass # forecast_func(train_data[station][-k:], k)
        # also an iterative measurement (do 1 step at a time, while progressing in train/test)
    return (mseDay/total_div, mseNight/total_div)
"""

# todo - a "read exp results" function, that gets arguments (like train/test) and return a sequence of the results of
#  the parameters specified for each k based on exp dict

def main():
    df = pd.read_csv(path_in + "noise_data.csv")
    metadata = pd.read_csv(path_in + "station_metadata.csv")
    stations = metadata["station_name"]
    train_concat, test_data = train_test_separation(df, stations)

    experiment = Multivar_ARIMA  # Univar_ARIMA or Multivar_ARIMA or GRU_train
    experimental_results = pd.DataFrame(columns=["k", "train MSE Day", "train MSE Night", "test MSE Day", "test MSE Night"])
    for k in K_ARRAY:
        if VERBOSE:
            print(f"k={k}")
        experimental_results.loc[len(experimental_results.index)] = experiment(train_concat, k_vectorize(test_data, k), k)
        experimental_results["k"] = experimental_results.astype(int)
        if VERBOSE:
            print(experimental_results)
        experimental_results.to_csv("results_temp.csv", index=False)
    print(experimental_results)


if __name__ == '__main__':
    main()
