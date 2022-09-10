import pandas as pd

pivot = True
path = ""
filename = "Noise Data - All stations.csv"
df = pd.read_csv(path + filename)

# drop dead (dataless) columns:
df = df.drop(["altitude", "unidades"], axis=1) #contains df.dropna(axis=1, how="all")
#unidades: Lpeak/Lpeak-day/Lpeak-night => dBC, otherwise dBA

# move station-based data to station data df:
station_df = df.drop_duplicates(subset="station_name")
df = df.drop(["town", "longitude", "latitude", "estacion"], axis=1)
station_df = station_df.drop(["published_dt", "nombre", "valor"], axis=1)

# reformat rows to columns:
df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
if pivot:
    df = df.pivot_table("valor", ["station_name", "published_dt"], "nombre")
    #df.reindex_axis(["", ""], axis=1)
else:
    df = df.dropna(axis=0, how="any", subset=["valor"])

# save:
df.to_csv("noise_data.csv")
station_df.to_csv("station_metadata.csv", index=False)
