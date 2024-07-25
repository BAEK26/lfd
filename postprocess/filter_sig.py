#remove all but significant points from the scenario file
import pandas as pd
import numpy as np
from time import time

file_path = 'scenarios/xyz.csv'
df = pd.read_csv(file_path)
# print(df)
# Compute significancy by differences
# diff = np.linalg.norm(np.diff(df[['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']], axis=0), axis=1)
# diff = np.linalg.norm(np.diff(df[['joint5']], axis=0), axis=1)


# threshold = 0.5

# significant_mask = np.concatenate(([True], diff >= threshold))
# significant_df = df[significant_mask]



# Comput significancy by gradients
# joint1_diff = df['joint1'].diff().values
# joint2_diff = df['joint2'].diff().values
# joint3_diff = df['joint3'].diff().values
# joint4_diff = df['joint4'].diff().values
# joint5_diff = df['joint5'].diff().values
# joint6_diff = df['joint6'].diff().values
x_diff = df['x'].diff().values
y_diff = df['y'].diff().values
z_diff = df['z'].diff().values

# Apply the significant criterion
significant_indices = np.where(isinstance(np.abs(x_diff), float) )[0]
print(significant_indices)
                               
# significant_indices  = np.where(np.abs(joint1_diff) > 0.5)[0]
# significant_indices = np.where((joint1_diff[:-1] * joint1_diff[1:]) < 0)[0] + 1
# significant_indices = np.union1d(significant_indices, np.where((joint2_diff[:-1] * joint2_diff[1:]) < 0)[0] + 1)
# significant_indices = np.union1d(significant_indices, np.where((joint3_diff[:-1] * joint3_diff[1:]) < 0)[0] + 1)
# significant_indices = np.union1d(significant_indices, np.where((joint4_diff[:-1] * joint4_diff[1:]) < 0)[0] + 1)
# significant_indices = np.union1d(significant_indices, np.where((joint5_diff[:-1] * joint5_diff[1:]) < 0)[0] + 1)
# significant_indices = np.union1d(significant_indices, np.where((joint6_diff[:-1] * joint6_diff[1:]) < 0)[0] + 1)
significant_indices = np.union1d(significant_indices, np.where((x_diff[:-1] * x_diff[1:]) < 0)[0] + 1)
significant_indices = np.union1d(significant_indices, np.where((y_diff[:-1] * y_diff[1:]) < 0)[0] + 1)
significant_indices = np.union1d(significant_indices, np.where((z_diff[:-1] * z_diff[1:]) < 0)[0] + 1)

# Extract significant points
significant_df = df.iloc[significant_indices]
print(significant_df)
print(significant_df.shape)



significant_df.to_csv('scenarios/xyz_only-sig.csv', index=False)

"""
60 -> filtered
all -> 35
+ joint1 threshold 0.5 -> 34
- joint2 -> 31
- joint3 -> 32
- joint4 -> 31
- joint5 -> 32
- joint6 -> 3
"""