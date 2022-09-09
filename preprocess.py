import pandas as pd

pivot = True
path = ""
filename = "Noise Data - All stations.csv"
df = pd.read_csv(path + filename)

# drop dead (dataless) columns:
df = df.drop(
    ["altitude", "unidades"],
    axis=1
)
# ^ : unidades: Lpeak/Lpeak-day/Lpeak-night => dBC, otherwise dBA


# move station-based data to station data df:
station_df = df.drop_duplicates(subset="station_name")
df = df.drop(
    ["town", "longitude", "latitude", "estacion"],
    axis=1
)
station_df = station_df.drop(
    ["published_dt", "nombre", "valor"],
    axis=1
)

# reformat rows to columns:
df["valor"] = pd.to_numeric(
    df["valor"],
    errors="coerce"
)
# ^ : includes:
# noise_data.replace(
#     to_replace=['-8,765,573,120.00', '864,426.38', '-16,000.00'],
#     value=np.nan,
#     inplace=True
# )

if pivot:
    df = df.pivot_table(
        "valor",
        ["station_name", "published_dt"],
        "nombre"
    )
    # ^ : group data in rows with the same station and with the same day to one row with all the nombres
else:
    df = df.dropna(axis=0, how="any", subset=["valor"])  # drop all rows with Nan at the 'valor' field

# save:
df.to_csv(
    "noise_data.csv"
)

station_df.to_csv(
    "station_metadata.csv",
    index=False
)
