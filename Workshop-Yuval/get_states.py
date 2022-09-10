import pandas as pd

pivot = True
path = ""
filename = "station_metadata.csv"
df = pd.read_csv(path + filename)
town_series = df.drop_duplicates(subset="town")['town']
print(town_series)


"""
Town, State (source: google) 
Bangalore, Karnataka
Lucknow, Uttar Pradesh
Mumbai, Maharashtra
Kolkata, West Bengal.
Hyderabad, Telangana
Chennai, Tamil Nadu
Delhi, Delhi
"""