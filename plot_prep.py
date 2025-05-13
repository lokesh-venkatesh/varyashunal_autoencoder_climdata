import pandas as pd

processing_params = pd.read_csv("data/processing_params_deets.csv", index_col=0)
print(processing_params.head())

cc_slope = processing_params.loc['cc_slope', 'value']
norm_offset = processing_params.loc['norm_offset', 'value']
norm_scale = processing_params.loc['norm_scale', 'value']

raw_data = pd.read_csv("data/final_timeseries.csv", index_col=0).squeeze()
gen_data = pd.read_csv("results/gnrtd_timeseries.csv", index_col=0).squeeze()

raw_data = raw_data*norm_scale + norm_offset
gen_data = gen_data*norm_scale + norm_offset

'''
raw_length = len(raw_data)
adjusted_raw_data = raw_data + cc_slope*(raw_length-raw_data.reset_index(drop=True).index-1)
gen_length = len(gen_data)
adjusted_gen_data = gen_data + cc_slope*(gen_length-gen_data.reset_index(drop=True).index-1)
'''

raw_data.to_csv("RAW.csv")
gen_data.to_csv("GEN.csv")