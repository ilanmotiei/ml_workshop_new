import math
import pandas as pd
import _datetime
COL_NAME = "Leq A Night"
DEBUG_MODE = False

def dB_to_pow(val):
    return 10 ** (val/10)


def pow_to_dB(val):
    return 10 * math.log(val, 10)

class predictor:

    def __init__(self, df):
        self.prediction_dict = None

    def predict(self, query):
        return self.prediction_dict[query]


class avg_predictor(predictor):

    def __init__(self, df):
        self.prediction_dict = df[COL_NAME].mean()

    def predict(self, query):
        return self.prediction_dict


class station_predictor(predictor):

    def __init__(self, df):
        self.prediction_dict = {station_name: df[df["station_name"] == station_name][COL_NAME].mean() for station_name in df["station_name"].drop_duplicates()}

    def predict(self, query):
        return self.prediction_dict[query["station_name"]]


class weekday_predictor(predictor):

    def __init__(self, df):
        DAYS_IN_A_WEEK = 7
        self.prediction_dict = [0 for _ in range(DAYS_IN_A_WEEK)]#[df[_datetime.datetime.strptime(df["published_dt"], "%Y-%m-%d").weekday() == i][COL_NAME].mean() for i in range(DAYS_IN_A_WEEK)]
        totals = [0 for _ in range(DAYS_IN_A_WEEK)]
        for _, row in df.iterrows():
            self.prediction_dict[_datetime.datetime.strptime(row["published_dt"], "%Y-%m-%d").weekday()] += row[COL_NAME]
            totals[_datetime.datetime.strptime(row["published_dt"], "%Y-%m-%d").weekday()] += 1
        for i in range(DAYS_IN_A_WEEK):
            self.prediction_dict[i] /= totals[i]

    def predict(self, query):
        return self.prediction_dict[_datetime.datetime.strptime(query["published_dt"], "%Y-%m-%d").weekday()]


class combined_predictor(predictor):

    def __init__(self, df):
        self.prediction_dict = {station_name: weekday_predictor(df[df["station_name"] == station_name]) for station_name in df["station_name"].drop_duplicates()}

    def predict(self, query):
        return self.prediction_dict[query["station_name"]].predict(query)

"""
class prev_day_predictor:

    def __init__(self, df):
        self.prediction_dict = df

    def predict(self, query):
        prd = self.prediction_dict
        return prd[(prd["station_name"] == query["station_name"]) & (prd["published_dt"] == query["published_dt"])][COL_NAME]
"""


def eval_predictor(df, pred):
    err = 0
    for _, record in df.iterrows():
        err += (pred.predict(record) - record[COL_NAME])**2
    return err/len(df)

""""
def eval_prev_predictor(df):
    err = 0
    unskipped = 0
    prev_entry = None
    for _, entry in df.iterrows():
        if prev_entry is None or prev_entry["station_name"] != entry["station_name"]:
            prev_entry = entry
            continue
        err += (prev_entry[COL_NAME] - entry[COL_NAME]) ** 2
        unskipped += 1
        prev_entry = entry
    if DEBUG_MODE:
        print("went over {} out of {}".format(unskipped, len(df)))
    return err/unskipped
"""

def eval_prev_predictor(df, day_num=3):
    err = 0
    unskipped = 0
    prev_records = [None]*day_num
    for _, record in df.iterrows():
        if prev_records[0] is None or prev_records[0]["station_name"] != record["station_name"]:
            prev_records.append(record)
            prev_records = prev_records[1:]
            continue
        prediction = sum([prev_record[COL_NAME] for prev_record in prev_records])/day_num
        err += (prediction - record[COL_NAME]) ** 2
        unskipped += 1
        prev_records.append(record)
        prev_records = prev_records[1:]
    if DEBUG_MODE:
        print("went over {} out of {}".format(unskipped, len(df)))
    return err/unskipped

def main():
    print("Starting...")
    path = ""
    filename = "noise_data.csv"
    df = pd.read_csv(path + filename)
    df = df.dropna(axis=0, how="any", subset=[COL_NAME])
    print(f"AVG: {eval_predictor(df, avg_predictor(df))}")
    print(f"STATION: {eval_predictor(df, station_predictor(df))}")
    print(f"WEEKDAY: {eval_predictor(df, weekday_predictor(df))}")
    print(f"COMBINED: {eval_predictor(df, combined_predictor(df))}")
    evaluation_list = list()
    for i in range(1, 8):
        if DEBUG_MODE:
            print(i)
        evaluation_list.append(eval_prev_predictor(df, day_num=i))
    #print(f"PREV_DAY: {eval_prev_predictor(df, day_num=1)}")
    #print(f"3_DAY_AVG: {eval_prev_predictor(df, day_num=3)}")
    #print(f"PREV_WEEK_AVG: {eval_prev_predictor(df, day_num=7)}")
    print(evaluation_list)
    print("Done")



if __name__ == '__main__':
    main()
