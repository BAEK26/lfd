import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

def interpolate_points(points):
    points = np.array(points)
    # print(points)
    interpolated_points = []

    for i in range(1, len(points[0])):
        cs = CubicSpline(points[:, 0], points[:, i], axis=0)
        segment_points = cs(np.linspace(0, len(points), 100))
        if i == 1:
            interpolated_points = np.column_stack((np.linspace(0, len(points), 100), segment_points))
        else:
            segment_points = segment_points.reshape(-1, 1)
            print(interpolated_points.shape, segment_points.shape)
            interpolated_points = np.append(interpolated_points, segment_points, axis=1)

    return interpolated_points
file_path = 'scenarios/one_scenario.csv'
df = pd.read_csv(file_path)
points = df
print(points)
interpolated_points = interpolate_points(points)
df = pd.DataFrame(interpolated_points)
df.columns = ['timestamp', 'x','y','z','roll','pitch','yaw','joint1','joint2','joint3','joint4','joint5','joint6']
print(df)

df.to_csv('scenarios/one_scenario_interpolated.csv', index=False)
