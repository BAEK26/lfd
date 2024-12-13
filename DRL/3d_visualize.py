# draw plt of csv file.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
origin_file_path = 'scenarios/xyz.csv'
interpolated_file_path = 'scenarios/xyz_interpolated.csv'
filtered_file_path = 'scenarios/xyz_only-sig.csv'

origin_df = pd.read_csv(origin_file_path)
interpolated_df = pd.read_csv(interpolated_file_path)
filtered_df = pd.read_csv(filtered_file_path)
# draw plt from df only points

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(origin_df['x'], origin_df['y'], origin_df['z'], label='origin', c='r')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.legend()
# plt.savefig('diagrams/xyz_origin_plot_plot.png', dpi=300)
# plt.show()

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(interpolated_df['x'], interpolated_df['y'], interpolated_df['z'], label='interpolated', c='g')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.legend()
# plt.savefig('diagrams/xyz_interpolated_plot_plot.png', dpi=300)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(filtered_df['x'], filtered_df['y'], filtered_df['z'], label='filtered', c='b')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.legend()
# plt.savefig('diagrams/xyz_filtered_plot_plot.png', dpi=300)
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(origin_df['x'], origin_df['y'], origin_df['z'], label='origin', c='r')
# ax.plot(interpolated_df['x'], interpolated_df['y'], interpolated_df['z'], label='interpolated', c='g')
# ax.plot(filtered_df['x'], filtered_df['y'], filtered_df['z'], label='filtered', c='b')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.legend()
# plt.savefig('diagrams/xyz_together_plot_plot.png', dpi=300)
# plt.show()