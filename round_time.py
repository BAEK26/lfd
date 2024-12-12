import csv
import pandas as pd


file_path = r"data\sampled_neo_show_scenario.csv"
new_file_path = r"data\relative_sampled_neo_show_scenario.csv"
data = pd.read_csv(file_path)

times = data["timestamp"].values

start_time = times[0]

relative_times = [round((t - start_time)*1000) for t in times]

data["timestamp"] = relative_times

data.to_csv(new_file_path, index=False)