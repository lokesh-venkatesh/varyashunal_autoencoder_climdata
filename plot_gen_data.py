import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

# Load your data
df = pd.read_csv("GEN.csv", parse_dates=["time"])
df = df.sort_values("time")

# 1. Seasonal profile across all years
df['month'] = df['time'].dt.month
seasonal_avg = df.groupby('month')['temperature'].mean()

plt.figure(figsize=(10, 5))
plt.plot(seasonal_avg.index, seasonal_avg.values)
plt.xticks(range(1,13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel("Month")
plt.ylabel("Monthly Average (°C)")
plt.title("Seasonal Profile Across All Years")
plt.tight_layout()
plt.savefig("results/plot1_gen_average_seasonal_profile.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Average Diurnal Profile
df['hour'] = df['time'].dt.hour
diurnal_avg = df.groupby('hour')['temperature'].mean()

diurnal_avg_indices = diurnal_avg.index.tolist()
diurnal_avg_vals = diurnal_avg.values.tolist()

plt.figure(figsize=(10, 6))
plt.plot(diurnal_avg_indices, diurnal_avg_vals)
plt.xticks(range(0, 24))
plt.xlabel("Hour")
plt.ylabel("Hourly Average (°C)")
plt.title("Average Diurnal Profile")
plt.tight_layout()
plt.savefig("results/plot2_gen_average_diurnal_profile.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Last two weeks of hourly data
last_four_weeks = df[df['time'] >= df['time'].max() - pd.Timedelta(weeks=4)]

plt.figure(figsize=(12, 5))
plt.plot(last_four_weeks['time'], last_four_weeks['temperature'])
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))
plt.xlabel("Date")
plt.ylabel("Hourly Temperature (°C)")
plt.title("Last Two Weeks of Hourly Data")
plt.tight_layout()
plt.savefig("results/plot3_gen_last_two_weeks_of_hourly_data.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Last year of daily averages
last_year = df[df['time'] >= df['time'].max() - pd.DateOffset(years=1)]
daily_avg = last_year.resample('D', on='time').mean()

plt.figure(figsize=(12, 5))
plt.plot(daily_avg.index, daily_avg['temperature'])
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.xlabel("Month")
plt.ylabel("Daily Average (°C)")
plt.title("Last Year of Daily Averages")
plt.tight_layout()
plt.savefig("results/plot4_gen_last_year_of_daily_averages.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Auto-correlation
def autocorrelation(series, lag):
    return series.autocorr(lag=lag)

hourly_corr = autocorrelation(df['temperature'], lag=1)
daily_df = df.resample('D', on='time').mean() # First resample to DataFrame, then access 'temperature'
daily_corr = autocorrelation(daily_df['temperature'], lag=1)

monthly_df = df.resample('M', on='time').mean()
monthly_corr = autocorrelation(monthly_df['temperature'], lag=1)

corrs = [hourly_corr, daily_corr, monthly_corr]
labels = ['Hourly', 'Daily', 'Monthly']

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, corrs, color='royalblue', width=0.5)
plt.ylim(0, 1)
plt.ylabel("Correlation Coefficient")
plt.xlabel("Variable")
plt.xticks(rotation=90)
plt.title("Auto-Correlation", pad=20)  # Adjusted the padding to move the title higher
for bar, corr in zip(bars, corrs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{corr:.4f}", 
             ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig("results/plot5_gen_autocorrelations.png", dpi=300, bbox_inches='tight')
plt.close()