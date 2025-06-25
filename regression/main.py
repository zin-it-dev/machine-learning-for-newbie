import pandas as pd
# from ydata_profiling import ProfileReport

df = pd.read_csv('../data/weatherAUS.csv')

# profile = ProfileReport(df, title="Weather AUS Report", explorative=True)
# profile.to_file("./reports/weatherAUS.html")

df['Date'] = pd.to_datetime(df['Date'])

df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Weekday'] = df['Date'].dt.weekday

target = 'Rainfall_next_day'

df = df.sort_values(['Location', 'Date'])
df[target] = df.groupby('Location')['Rainfall'].shift(-1)

df['Rainfall_t-1'] = df.groupby('Location')['Rainfall'].shift(1)
df['Rainfall_t-2'] = df.groupby('Location')['Rainfall'].shift(2)
df['Rainfall_t-3'] = df.groupby('Location')['Rainfall'].shift(3)

