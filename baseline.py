import numpy as np
import pandas as pd
import _datetime
COL_NAMES = ["Leq A Day", "Leq A Night"]
DEBUG_MODE = True


class Predictor:

    def __init__(
            self,
            df: pd.DataFrame
    ):
        super().__init__()
        self.prediction_dict = None

    def predict(self, query):
        return self.prediction_dict[query]


class AvgPredictor(Predictor):

    def __init__(
            self,
            df: pd.DataFrame,
            col_name: str
    ):
        super().__init__(
            df=df
        )
        self.prediction_dict = df[col_name].mean()

    def predict(self, query):
        return self.prediction_dict


class StationPredictor(Predictor):

    def __init__(
            self,
            df: pd.DataFrame,
            col_name: str
    ):
        super().__init__(
            df=df
        )
        self.prediction_dict = {
            station_name: df[df["station_name"] == station_name][col_name].mean()
            for station_name in df["station_name"].drop_duplicates()
        }

    def predict(self, query):
        return self.prediction_dict[query["station_name"]]


class WeekdayPredictor(Predictor):

    def __init__(
            self,
            df: pd.DataFrame,
            col_name: str
    ):
        super().__init__(
            df=df
        )

        DAYS_IN_A_WEEK = 7
        self.prediction_dict = [0 for _ in range(DAYS_IN_A_WEEK)]

        totals = [0 for _ in range(DAYS_IN_A_WEEK)]

        for _, row in df.iterrows():
            weekday = _datetime.datetime.strptime(row["published_dt"], "%Y-%m-%d").weekday()
            self.prediction_dict[weekday] += row[col_name]
            totals[weekday] += 1

        for i in range(DAYS_IN_A_WEEK):
            self.prediction_dict[i] /= totals[i]

    def predict(self, query):
        weekday = _datetime.datetime.strptime(query["published_dt"], "%Y-%m-%d").weekday()
        return self.prediction_dict[weekday]


class CombinedPredictor(Predictor):

    def __init__(
            self,
            df: pd.DataFrame,
            col_name: str
    ):
        super().__init__(
            df=df
        )

        self.prediction_dict = {
            station_name: WeekdayPredictor(
                df=df[df["station_name"] == station_name],
                col_name=col_name
            )
            for station_name in df["station_name"].drop_duplicates()
        }

    def predict(self, query):
        return self.prediction_dict[query["station_name"]].predict(query)


def eval_predictor(
        df: pd.DataFrame,
        pred: Predictor,
        col_name: str
) -> float:
    err = 0

    for _, record in df.iterrows():
        err += (pred.predict(record) - record[col_name]) ** 2

    return err / len(df)


def eval_prev_predictor(
        df: pd.DataFrame,
        col_name: str,
        day_num: int = 3
) -> float:

    err = 0
    unskipped = 0
    prev_records = [None] * day_num

    for _, record in df.iterrows():
        if prev_records[0] is None or prev_records[0]["station_name"] != record["station_name"]:
            prev_records.append(record)
            prev_records = prev_records[1:]
            continue

        prediction = np.nanmean([prev_record[col_name] for prev_record in prev_records])

        loss = (prediction - record[col_name]) ** 2

        if not np.all(np.isnan(loss)):
            err += loss

        unskipped += 1
        prev_records.append(record)
        prev_records = prev_records[1:]
    if DEBUG_MODE:
        print("went over {} out of {}".format(unskipped, len(df)))

    return err / unskipped


def main():
    print("Starting...")
    path = ""
    filename = "noise_data.csv"
    df = pd.read_csv(path + filename)

    for col_name in COL_NAMES:
        print(f"============= {col_name} ============= ")
        non_nan_df = df.dropna(axis=0, how='any', subset=[col_name])

        print(f"AVG: {eval_predictor(non_nan_df, AvgPredictor(non_nan_df, col_name), col_name)}")
        print(f"STATION: {eval_predictor(non_nan_df, StationPredictor(non_nan_df, col_name), col_name)}")
        print(f"WEEKDAY: {eval_predictor(non_nan_df, WeekdayPredictor(non_nan_df, col_name), col_name)}")
        print(f"COMBINED: {eval_predictor(non_nan_df, CombinedPredictor(non_nan_df, col_name), col_name)}")

        for k in list(range(1, 7)):
            print(f"{k}_DAY_AVG: {eval_prev_predictor(df, day_num=k, col_name=col_name)}")

    print("Done!")


if __name__ == '__main__':
    main()
