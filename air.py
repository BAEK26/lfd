import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from glob import glob
import os
from time import time
import csv

df = pd.read_csv('data/robot_data_0.csv')
timeseries = df[["joint1","joint2","joint3","joint4","joint5","joint6"]].values.astype('float32')
csv_paths = glob(os.path.join('data',"*.csv"))

for csv_path in csv_paths:
    scenario = pd.read_csv(csv_path)
    timeseries = np.concatenate([timeseries, scenario[["joint1","joint2","joint3","joint4","joint5","joint6"]].values.astype('float32')])

train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

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

lookback = 10
batch_size = 8
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 6)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)

n_epochs = 1000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
test = False
if test:
    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(timeseries) * np.nan
        # y_pred = model(X_train)
        # y_pred = y_pred[:, -1, :]
        train_plot[lookback:train_size] = model(X_train)[:, -1, :]
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
    # plot
    plt.plot(timeseries, c='b')
    plt.plot(train_plot, c='r')
    plt.plot(test_plot, c='g')
    plt.show()
# demonstraiton
# using the model to predict the next 100 steps from t0 and show the plot.
# t0 should be 32type tensor with shape [1, 10, 6]
timesteps = 40
init = 2
t0 = torch.tensor(np.array([93.483335,26.31005,33.65004,-5.537121,-89.355461,-4.383528]), dtype=torch.float32).repeat(init).reshape(1, init, 6)
show_plot = np.ones((timesteps+init, 6)) * np.nan
with torch.no_grad():
    for i in range(timesteps):
        pred = model(t0)
        # take t0 and prediction so that t0 gets bigger.
        t0 = torch.cat([t0, pred[:, -1, :].reshape(1, 1, 6)], dim=1)
        show_plot[init + i, :] = pred[:, -1, :].numpy()
    
plt.plot(show_plot) # plot the data
plt.show()  # show the plot

print()
# write t0 data to csv file
# t0 6 points refers to joint1, joint2, joint3, joint4, joint5, joint6.
testfile = os.path.join("scenarios", "parallel"+'.csv')
with open(testfile, 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(timesteps+init):
        if show_plot[i, 0] is np.nan:
            continue
        writer.writerow({'timestamp': i, 'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0, 'joint1': show_plot[i, 0], 'joint2': show_plot[i, 1], 'joint3': show_plot[i, 2], 'joint4': show_plot[i, 3], 'joint5': show_plot[i, 4], 'joint6': show_plot[i, 5]})
