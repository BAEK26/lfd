import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from glob import glob
import os
from time import time
import csv
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--file_name', action='store', type=str, default='test', help="model's path will be generated in this file", required=False) 
parser.add_argument("--model_path", type=str, default=None, help="model path")
parser.add_argument("--train", action='store_true', help="train the model")
parser.add_argument("--visulaize", action='store_true', help="visualize the model")
parser.add_argument("--save_model_at", type=str, help="save the model")
args = parser.parse_args()



df = pd.read_csv('data/robot_data_0.csv')
timeseries = df[["joint1","joint2","joint3","joint4","joint5","joint6"]].values.astype('float32')
scenarios = []
csv_paths = glob(os.path.join('data',"*.csv"))

for csv_path in csv_paths:
    scenario = pd.read_csv(csv_path)
    scenarios.append(scenario[["joint1","joint2","joint3","joint4","joint5","joint6"]].values.astype('float32'))
    timeseries = np.concatenate([timeseries, scenario[["joint1","joint2","joint3","joint4","joint5","joint6"]].values.astype('float32')])


# train_size = int(len(timeseries) * 0.67)
# test_size = len(timeseries) - train_size
# train, test = timeseries[:train_size], timeseries[train_size:]

for s in scenarios:
    # print(s.shape)
    pass
train_size = int(len(scenarios) * 0.67)
test_size = len(scenarios) - train_size
train, test = scenarios[:train_size], scenarios[train_size:]
print([a.shape for a in train], [e.shape for e in test])

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))

lookback = 30
batch_size = 1
# X_train, y_train = create_dataset(train, lookback=lookback)
# X_test, y_test = create_dataset(test, lookback=lookback)

def create_scenario_dataset(scenario_list):
    X, Y = [], []
    lengths = []
    for scenario in scenario_list:
        feature = scenario[:-1]
        target = scenario[1:]
        X.append(torch.tensor(feature))
        Y.append(torch.tensor(target))
        lengths.append(len(feature))
    padded_x , padded_y = pad_sequence(X, batch_first=True), pad_sequence(Y, batch_first=True)
    return padded_x, padded_y, lengths
        
X_train, y_train, lengths_train = create_scenario_dataset(train)
X_test, y_test, lengths_test = create_scenario_dataset(test)

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 6)
    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (hn, cn) = self.lstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.linear(out)
        return out

model = AirModel()
# print the number of model parameters
print(sum(p.numel() for p in model.parameters()))
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

if args.model_path is not None:
    model.load_state_dict(torch.load(args.model_path))
    print("Model loaded from", args.model_path)


class SequenceDataset(data.Dataset):
    def __init__(self, sequences, targets, lengths):
        self.sequences = sequences
        self.targets = targets
        self.lengths = lengths
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        return self.sequences[index], self.targets[index], self.lengths[index]

train_dataset = SequenceDataset(X_train, y_train, lengths_train)
test_dataset = SequenceDataset(X_test, y_test, lengths_test)

train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=1)
test_loader = data.DataLoader(test_dataset, shuffle=False, batch_size=1)

if args.train:
    n_epochs = 1500
    for epoch in tqdm(range(n_epochs)):
        model.train()
        for X_batch, y_batch, lengths in train_loader:
            y_pred = model(X_batch, lengths)

            y_pred = pack_padded_sequence(y_pred, lengths, batch_first=True, enforce_sorted=False).data
            y_batch = pack_padded_sequence(y_batch, lengths, batch_first=True, enforce_sorted=False).data
            
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train, lengths_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))

            y_pred = model(X_test, lengths_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

    if args.save_model_at is not None:
        torch.save(model.state_dict(), args.save_model_at)
        print("Model saved at", args.save_model_at)

if args.visulaize:
    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(timeseries) * np.nan
        # y_pred = model(X_train)
        # y_pred = y_pred[:, -1, :]
        train_plot[lookback:train_size] = model(X_train, lengths_train)[:, -1, :]
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[train_size+lookback:len(timeseries)] = model(X_test, lengths_test)[:, -1, :]
    # plot
    plt.plot(timeseries, c='b')
    plt.plot(train_plot, c='r')
    plt.plot(test_plot, c='g')
    plt.show()
# demonstraiton
# using the model to predict the next 100 steps from t0 and show the plot.
# t0 should be 32type tensor with shape [1, 10, 6]
timesteps = 60
length = 1
t0 = torch.tensor(np.array([93.483335,26.31005,33.65004,-5.537121,-89.355461,-4.383528]), dtype=torch.float32).repeat(length).reshape(1, length, 6)
show_plot = np.ones((timesteps+length, 6)) * np.nan
with torch.no_grad():
    for i in range(timesteps):
        pred = model(t0, [length])
        # take t0 and prediction so that t0 gets bigger.
        t0 = torch.cat([t0, pred[:, -1, :].reshape(1, 1, 6)], dim=1)
        show_plot[length-1, :] = pred[:, -1, :].numpy()
        length += 1

plt.plot(show_plot) # plot the data
plt.show()  # show the plot

print()
# write t0 data to csv file
# t0 6 points refers to joint1, joint2, joint3, joint4, joint5, joint6.
testfile = os.path.join("scenarios", "one_scenario"+'.csv')
with open(testfile, 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(timesteps):
        if np.isnan(show_plot[i, 0]):
            continue
        writer.writerow({'timestamp': i, 'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0, 'joint1': show_plot[i, 0], 'joint2': show_plot[i, 1], 'joint3': show_plot[i, 2], 'joint4': show_plot[i, 3], 'joint5': show_plot[i, 4], 'joint6': show_plot[i, 5]})
