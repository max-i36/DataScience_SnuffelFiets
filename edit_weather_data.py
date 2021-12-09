import pandas as pd

weather_data_raw = pd.read_csv('weerdata_raw.txt', header=50, low_memory=False)

weather_data = weather_data_raw.where(weather_data_raw.YYYYMMDD > 20180101).dropna()

print(weather_data.head())


def convert_to_int(x):
    try:
        return x.astype(int)
    except:
        return x


weather_data.apply(convert_to_int).to_csv('weerdata.csv', index=False)
