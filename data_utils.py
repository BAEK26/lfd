import os
import csv
import torch
from glob import glob
import pandas as pd
import numpy as np

class XArmDataset(torch.utils.data.Dataset):
    def __init__(self, path, sequence_length = 5):
        super(XArmDataset, self).__init__()
        self.sequence_length = sequence_length
        self.coordinates = []
        self.next_coordinates = []
        self.load_dataset(path)

    def load_dataset(self, path):
        assert os.path.isdir(path)
        csv_paths = glob(os.path.join(path,"*.csv"))
        for csv_path in csv_paths:
            scenario = pd.read_csv(csv_path)

            coordinates = []
            next_coordinates = []
            for i in range(len(scenario) - self.sequence_length):
                coordinates.append(scenario.iloc[i:i+self.sequence_length][['joint1','joint2','joint3','joint4','joint5','joint6', 'timestamp']].values)
                next_coordinates.append(scenario.iloc[i+self.sequence_length][['joint1','joint2','joint3','joint4','joint5','joint6', 'timestamp']].values)


            #timestamp,x,y,z,roll,pitch,yaw,joint1,joint2,joint3,joint4,joint5,joint6
            self.coordinates.extend(np.array(coordinates))
            self.next_coordinates.extend(np.array(next_coordinates))
            #TODO timestamp -> delta
        self.coordinates = torch.tensor(self.coordinates, dtype=torch.float32)
        self.next_coordinates = torch.tensor(self.next_coordinates, dtype=torch.float32)

    def __len__(self):
        return len(self.coordinates) - self.sequence_length
    def __getitem__(self, idx):
        return self.coordinates[idx], self.next_coordinates[idx]

def load_dataset() -> XArmDataset:
    actions_dataset = []
    files = glob('data\*')
    for file in files:
        scenario = []
        actions = []
        positions = []
        data_path = os.path.join('data', file)
        with open(file, 'r') as csvfile:
            fieldnames = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            t0 = None
            reader = csv.DictReader(csvfile)
            for row in reader:
                if t0 is None:
                    t0 = float(row['timestamp'])
                positions.append(torch.tensor([[float(row['joint1']),
                                             float(row['joint2']),
                                             float(row['joint3']),
                                             float(row["joint4"]),
                                             float(row['joint5']),
                                             float(row['joint6']),
                                             float(row['timestamp'])-t0
                                             ]]))
        for i, position in enumerate(positions):
            if i+1 == len(positions):
                actions.append(positions[-1])
            else:
                actions.append(positions[i+1])

        actions_dataset.append([actions, positions])
    return actions_dataset



def load_dataset_list() -> list:
    actions_dataset = []
    files = glob('data\*')
    for file in files:
        scenario = []
        actions = []
        positions = []
        data_path = os.path.join('data', file)
        with open(file, 'r') as csvfile:
            fieldnames = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            t0 = None
            reader = csv.DictReader(csvfile)
            for row in reader:
                if t0 is None:
                    t0 = float(row['timestamp'])
                positions.append(torch.tensor([[float(row['joint1']),
                                             float(row['joint2']),
                                             float(row['joint3']),
                                             float(row["joint4"]),
                                             float(row['joint5']),
                                             float(row['joint6']),
                                             float(row['timestamp'])-t0
                                             ]]))
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