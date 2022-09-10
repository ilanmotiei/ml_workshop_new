import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import os


import warnings
warnings.filterwarnings("ignore", message="An unsupported index was provided and will be ignored when e.g. forecasting")


path = ""
DEBUG_MODE = True


def ARIMA_preliminary_research(df, station_name, col):
    txt_out = str()
    txt_out += f"Starting {station_name} ({col})\n"
    folder = path + f"ARIMA_preliminary_research\\{station_name}_only\\{col}\\"
    if not os.path.exists(folder):
        os.makedirs(folder)
    series = df[df["station_name"] == station_name][col]
    series = series.dropna()
    for i in range(3):
        plt.plot(series)
        plt.title(f"Derivative Order {i}")
        plt.savefig(folder + f"Derivative Order {i}")
        plt.clf()
        plot_acf(series, lags=30)
        plt.title(f"Autocorrelation Order {i}")
        plt.savefig(folder + f"Autocorrelation Order {i}")
        plt.clf()
        plot_pacf(series, lags=30, method="ywm")
        plt.title(f"Partial Autocorrelation Order {i}")
        plt.savefig(folder + f"Partial Autocorrelation Order {i}")
        plt.clf()
        adfuller_test_res = adfuller(series)
        txt_out += f"Augmented Dickey-Fuller Test {i}: p-value={adfuller_test_res[1]}, test statistic={adfuller_test_res[0]}, lags={adfuller_test_res[2]}\n"
        series = series.diff().dropna()
    with open(folder + "tests.txt", "w") as f:
        f.write(txt_out)
    if DEBUG_MODE:
        print(txt_out)


def sample_stations():
    return ['Bangalore_RVCE', 'Lucknow_Aligunj', 'Lucknow_SGPGI', 'Mumbai_Pepsico Chembur', 'Mumbai_Thane', 'Kolkata_New Market', 'Kolkata_Tollygunge', 'Hyderabad_Gachibowli', 'Hyderabad_Abid', 'Chennai_Anna Nagar', 'Chennai_Sowcarpet', 'Delhi_ITO (Pragati Maidan)', 'Delhi_Dilshad Garden', 'Banglore_BTM']
    metadata_df = pd.read_csv(path + "station_metadata.csv")
    sampled_stations = list()
    sampled_towns = metadata_df["town"].drop_duplicates().dropna() #.sample(5) # 5 out of 7 # TODO - fix the Bangalore/Banglore error in the station metadata
    for town in sampled_towns:
        local_stations = metadata_df[metadata_df["town"] == town]["station_name"]
        if len(local_stations) > 8:  # usually 9 or 10
            samp = local_stations.sample(2)
            sampled_stations.append(samp.iloc[0])
            sampled_stations.append(samp.iloc[1])
        else:  # usually length==5
            sampled_stations.append(local_stations.sample(1).iloc[0])
    return sampled_stations


def station_sampling_research():
    df = pd.read_csv(path + "noise_data.csv")
    sampled_stations = sample_stations()
    for station in sampled_stations:
        ARIMA_preliminary_research(df, station, "Leq A Day")
        ARIMA_preliminary_research(df, station, "Leq A Night")


def build_ARIMAs():
    df = pd.read_csv(path + "noise_data.csv")
    sampled_stations = sample_stations()
    MSE_sum = 0
    for station, i in zip(sampled_stations, range(len(sampled_stations))):
        if DEBUG_MODE:
            print(f"On station {station} ({i+1}/{len(sampled_stations)})")
        s = df[df["station_name"] == station]["Leq A Day"]
        arima_model = ARIMA(s, order=(7, 0, 1))
        model = arima_model.fit()
        if DEBUG_MODE:
            print(model.mse)
        MSE_sum += model.mse
    print(MSE_sum / len(sampled_stations))
    # TODO

def main():
    if DEBUG_MODE:
        print("Starting...")
    #station_sampling_research()

    # conclusions:
    # d is definitely 0 - at derivative 1 or higher, the p-value becomes minuscule, and autocorrelation becomes random
    # autocorrelation is a nice descending curve , whist partail autocorrelation is high only at 1, with a sharp cutoff, and positive up to ~7
    # therefore, the data has an "AR signature"
    # reasonable orders: (1, 0, 0), (1, 0, 1), (7, 0, 0), (7, 0, 1)
    # additional note: we can sometime see a spike at the autocorrelation for lag 7 (such as at Kolkata_Tollygunge (day)), which makes sense (smae day of the week)
    # introduce seasonality (due to COVID)?

    build_ARIMAs()
    # TODO - make "sample_stations" also hold off on (and cut from the predetemined list accordingly), create a "general series" (value each day is the avg of that day at all stations, ensure the data is sorted by date
    # TODO - create a new file for all misc ("data utils"?) - get states, order, create general series sample stations (data cleansing not included - it is better indivudually
    # TODO - try ARIMA on the sampled stations and check optimal preformance, make ARIMA for each station individually, make ARIMA for general series; see "plot predict"
    # TODO - fix the banglore/bangalore error

if __name__ == '__main__':
    main()
