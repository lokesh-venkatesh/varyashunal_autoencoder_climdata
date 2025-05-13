'''
cc_adjstd_nrmlsd_df['time'] = pd.to_datetime(cc_adjstd_nrmlsd_df['time']) 
# Convert 'valid_time' to datetime if it's not already

cc_adjstd_nrmlsd_df = cc_adjstd_nrmlsd_df[(cc_adjstd_nrmlsd_df['time'].dt.year >= 1970) & (cc_adjstd_nrmlsd_df['time'].dt.year <= 2020)] 
cc_adjstd_df = cc_adjstd_df[(cc_adjstd_df['time'].dt.year >= 1970) & (cc_adjstd_df['time'].dt.year <= 2020)] 
# Filter out data from after 2020 and before 1970
cc_adjstd_nrmlsd_df['year'] = cc_adjstd_nrmlsd_df['time'].dt.year # Add a 'year' column for grouping

print(cc_adjstd_nrmlsd_df.head())
# Calculate the yearly average temperatures
#raw_temps = pd.read_csv(combined_csv_path, index_col=None)
#raw_temps['time'] = pd.to_datetime(raw_temps["time"])
cc_adjstd_df['year'] = cc_adjstd_df['time'].dt.year
yearly_avg_raw_temp = cc_adjstd_df.groupby('year')['temperature'].mean()
yearly_avg_adj_temp = cc_adjstd_nrmlsd_df.groupby('year')['temperature'].mean()

# Plot the yearly average temperatures
plt.figure(figsize=(10, 5))
plt.plot(yearly_avg_raw_temp.index, yearly_avg_raw_temp.values, label='Raw Temperature', color='blue')
plt.plot(yearly_avg_adj_temp.index, yearly_avg_adj_temp.values, label='Adjusted Temperature', color='red')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.title('Yearly Average Temperature Comparison')
plt.xticks(yearly_avg_raw_temp.index, rotation=90)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('results/plot6_impact_of_climate_change.png', dpi=300)
'''