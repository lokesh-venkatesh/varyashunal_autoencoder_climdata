import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df_list = []
for year in range(1970, 2021, 3):
    ds = xr.open_dataset(f'data/raw_data_files/t2m_{year}_{year+2}/data_stream-oper_stepType-instant.nc')
    df = pd.DataFrame(index=ds['valid_time'], data=ds['t2m'][:, 0, 0])
    df.index = pd.to_datetime(df.index)
    df_list.append(df)

df = pd.concat(df_list)
df.columns = ['Observed']

# convert Kelvin to Fahrenheit
df.Observed = (df.Observed - 273.15) * 9/5 + 32

df_mst = df.copy()
df_mst.index = df_mst.index.shift(-7, freq='h')
df_mst = df_mst.iloc[7:]

# Calculate the slope of the temperature trend across yearly averages

df_yearly = df_mst.resample('YS').mean()
slope = np.polyfit(df_yearly.index.year, df_yearly['Observed'], 1)[0]

# Add climate adjusted data to dataframe

df_mst['Climate Adjusted'] = df_mst['Observed'] - 0.07 * (df_mst.index.year - 2024)

df_yearly = df_mst.resample('YS').mean()
df_yearly.plot()
plt.title('Yearly Average Temperature in Phoenix')
plt.ylabel('Temperature (°F)')
plt.xlabel('Year')
plt.gca().get_lines()[1].set_linestyle('--')

# set the color of both lines to orange
plt.gca().get_lines()[0].set_color('orange')
plt.gca().get_lines()[1].set_color('red')

plt.legend()
plt.savefig('results/long_term_trend.png', dpi=300)
plt.close()

final_timeseries = df_mst.drop(columns=['Observed'])
final_timeseries.to_csv('data/final_timeseries.csv')

obsvd_timeseries = df_mst.drop(columns=['Climate Adjusted'])
obsvd_timeseries.to_csv('data/obsvd_timeseries.csv')

offset = round(df_mst['Climate Adjusted'].mean(), 2)
print('offset: ', offset)
scale = round(df_mst['Climate Adjusted'].std(), 2)
print('scale:  ', scale)
dft = (df_mst['Climate Adjusted'] - offset) / scale

# reshape dataframe to have 64*24 columns

k = 64*24
n = dft.shape[0] // k

data = dft.iloc[:n*k].values.reshape(n, k)
index = dft.index[:-k:k]

dft_reshaped = pd.DataFrame(data=data, index=index)

np.random.seed(42)
# shuffle rows
dft_reshaped = dft_reshaped.sample(frac=1)
dft_reshaped.to_csv('data/reshaped_dataset.csv')