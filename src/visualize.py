# draw plt of csv file.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
origin_file_path = 'scenarios/one_scenario.csv'
interpolated_file_path = 'scenarios/one_scenario_interpolated.csv'
filtered_file_path = 'scenarios/one_scenario_only-sig.csv'

origin_df = pd.read_csv(origin_file_path)
interpolated_df = pd.read_csv(interpolated_file_path)
filtered_df = pd.read_csv(filtered_file_path)
# draw plt from df only points

plt.scatter(origin_df['timestamp'], origin_df['joint6'], label='joint6')
plt.scatter(origin_df['timestamp'], origin_df['joint5'], label='joint5')
plt.scatter(origin_df['timestamp'], origin_df['joint4'], label='joint4')
plt.scatter(origin_df['timestamp'], origin_df['joint3'], label='joint3')
plt.scatter(origin_df['timestamp'], origin_df['joint2'], label='joint2')
plt.scatter(origin_df['timestamp'], origin_df['joint1'], label='joint1')
plt.legend()
plt.savefig('diagrams/post-process_origin.png')
plt.show()


plt.scatter(interpolated_df['timestamp'], interpolated_df['joint6'], label='joint6_interpolated')
plt.scatter(interpolated_df['timestamp'], interpolated_df['joint5'], label='joint5_interpolated')
plt.scatter(interpolated_df['timestamp'], interpolated_df['joint4'], label='joint4_interpolated')
plt.scatter(interpolated_df['timestamp'], interpolated_df['joint3'], label='joint3_interpolated')
plt.scatter(interpolated_df['timestamp'], interpolated_df['joint2'], label='joint2_interpolated')
plt.scatter(interpolated_df['timestamp'], interpolated_df['joint1'], label='joint1_interpolated')
plt.legend()
plt.savefig('diagrams/post-process_interpolated.png')
plt.show()

plt.scatter(filtered_df['timestamp'], filtered_df['joint6'], label='joint6_filtered')
plt.scatter(filtered_df['timestamp'], filtered_df['joint5'], label='joint5_filtered')
plt.scatter(filtered_df['timestamp'], filtered_df['joint4'], label='joint4_filtered')
plt.scatter(filtered_df['timestamp'], filtered_df['joint3'], label='joint3_filtered')
plt.scatter(filtered_df['timestamp'], filtered_df['joint2'], label='joint2_filtered')
plt.scatter(filtered_df['timestamp'], filtered_df['joint1'], label='joint1_filtered')
plt.legend()
plt.savefig('diagrams/post-process_filtered.png')
plt.show()