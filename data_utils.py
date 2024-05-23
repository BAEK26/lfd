import csv
import torch
def load_dataset_list() -> list:
    actions_dataset = []
    for i in range(5):
        scenario = []
        actions = []
        positions = []
        with open(f'robot_data_{i}.csv', 'r') as csvfile:
            fieldnames = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            t0 = None
            reader = csv.DictReader(csvfile)
            for row in reader:
                if t0 is None:
                    t0 = float(row['timestamp'])
                positions.append(torch.tensor([float(row['joint1']),
                                             float(row['joint2']),
                                             float(row['joint3']),
                                             float(row["joint4"]),
                                             float(row['joint5']),
                                             float(row['joint6']),
                                             float(row['timestamp'])-t0
                                             ]))
        for i, position in enumerate(positions):
            if i+1 == len(positions):
                actions.append(positions[-1])
            else:
                actions.append(positions[i+1])


        actions_dataset.append([actions, positions])
    return actions_dataset
if __name__ == "__main__":
    actions , positions = load_dataset_list()[0]
    for act, pos in zip(actions, positions):
        print('act: ', act, 'pos: ', pos)