import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

############################
# Class 1: DMPDataLoader
############################
class DMPDataLoader:
    def __init__(self, data_file, alpha=4.0):
        self.data_file = data_file
        self.alpha = alpha
        self.joint_scaler = MinMaxScaler()
        self.xyz_scaler = MinMaxScaler()
        self.joints_demo_scaled = None
        self.xyz_demo_scaled = None
        self.t = None
        self.s = None
        self.timesteps = None
        print("DMPDataLoader initialized.")

    def load_and_preprocess(self):
        print("Loading and preprocessing data...")
        data = pd.read_csv(self.data_file)
        joint_cols = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        xyz_cols = ['x', 'y', 'z']
        
        # Extract joints and xyz data
        self.joints_demo = data[joint_cols].values
        self.xyz_demo = data[xyz_cols].values
        self.timesteps = len(data)
        
        # Compute time steps
        x_demo, y_demo, z_demo = self.xyz_demo[:, 0], self.xyz_demo[:, 1], self.xyz_demo[:, 2]
        self.t = np.cumsum(np.sqrt(np.diff(x_demo)**2 + np.diff(y_demo)**2 + np.diff(z_demo)**2))
        self.t = np.insert(self.t, 0, 0) / self.t[-1]
        self.s = np.exp(-self.alpha * self.t)

        # Scale the data
        self.joints_demo_scaled = self.joint_scaler.fit_transform(self.joints_demo)
        self.xyz_demo_scaled = self.xyz_scaler.fit_transform(self.xyz_demo)

        print("Data preprocessing complete.")

    def get_training_data(self):
        print("Preparing training data...")
        start_joints = self.joints_demo_scaled[0]
        target_joints = self.joints_demo_scaled[-1]
        start_xyz = self.xyz_demo_scaled[0]
        target_xyz = self.xyz_demo_scaled[-1]

        train_input = []
        for i in range(self.timesteps):
            inp = [self.t[i], self.s[i]] + start_joints.tolist() + start_xyz.tolist() + target_joints.tolist() + target_xyz.tolist()
            train_input.append(inp)
        
        train_output = np.concatenate([self.joints_demo_scaled, self.xyz_demo_scaled], axis=1)
        print("Training data ready.")
        return torch.tensor(train_input, dtype=torch.float32), torch.tensor(train_output, dtype=torch.float32)

############################
# Class 2: DMPTrainer
############################
class DMPTrainer:
    def __init__(self, input_dim=20, output_dim=9, hidden_dim=128, learning_rate=0.0018, epochs=2000):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softsign(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.writer = SummaryWriter(log_dir='./runs/dmp_training')
        print("DMPTrainer initialized.")

    def train(self, train_input, train_output):
        print("Starting training...")
        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()
            predictions = self.model(train_input)
            loss = self.criterion(predictions, train_output)
            loss.backward()
            self.optimizer.step()
            
            mse = torch.mean((predictions - train_output)**2).item()
            self.writer.add_scalar('Loss/train', loss.item(), epoch)
            self.writer.add_scalar('MSE/train', mse, epoch)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss.item():.6f}, MSE: {mse:.6f}")
        print("Training complete.")

    def save_model(self, path='dmp_model.pt'):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}.")

############################
# Main Execution
############################
if __name__ == "__main__":
    # Data loading and preprocessing
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_file_path = os.path.join(base_dir, 'data', 'pumping_interpolated_trajectory.csv')
    
    data_loader = DMPDataLoader(data_file=data_file_path)
    data_loader.load_and_preprocess()
    train_input, train_output = data_loader.get_training_data()

    # Model training
    trainer = DMPTrainer()
    trainer.train(train_input, train_output)
    trainer.save_model()
