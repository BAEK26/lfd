import os
import csv
import torch
import argparse
import datetime
import data_utils
from data_utils import XArmDataset
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

dtype = torch.float
#lstm
class SimpleLearner(nn.Module): 
    def __init__(self, args):
        super(SimpleLearner, self).__init__()
        self.hidden_size = args.hidden_size

        self.lstm = nn.LSTM(input_size=args.input_size, hidden_size=args.hidden_size, dropout=0.2)
        self.W = nn.Parameter(torch.randn([args.hidden_size, args.output_size]).type(dtype))
        self.b = nn.Parameter(torch.randn([args.output_size]).type(dtype))
        self.final = nn.Linear(self.hidden_size, args.output_size, bias= True)

        self.activateion = nn.ReLU()    

    def forward(self,X, hidden, cell):
        
        # Add batch dimension to input
        X = X.unsqueeze(1)
        # X = X.unsqueeze(0)
        outputs, (hidden, cell) = self.lstm(X, (hidden, cell))
        outputs = outputs[-1]  # 최종 예측 Hidden Layer

        model = self.final(outputs)

        return model, hidden, cell

    def init_hidden_cell(self, batch_size):
        h = torch.zeros(1, batch_size, self.hidden_size)        
        c = torch.zeros(1, batch_size, self.hidden_size)        
        return h, c

print('hello')
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', action='store', type=int, default=64, help='hidden_size', required=False)
    parser.add_argument('--input_size', action='store', type=int, default=7, help='input_size', required=False) #joint1~6, t
    parser.add_argument('--output_size', action='store', type=int, default=7, help='output_size', required=False) #joint1~6, dt
    parser.add_argument('--batch_size', action='store', type=int, default=1, help='output_size', required=False) #chunk?
    parser.add_argument('--epochs', action='store', type=int, default=10, help='output_size', required=False)
    parser.add_argument('--file_name', action='store', type=str, default='test', help="model's path will be generated in this file", required=False) 
    args = parser.parse_args()
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")

    #train data
    actions_dataset = XArmDataset(path='data', sequence_length=5)
    

    #train
    model = SimpleLearner(args).to(device)
    criterion = nn.MSELoss()#MSEloss, nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-1) #torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = args.epochs

    train_loader = DataLoader(actions_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    for epoch in range(num_epochs):
        for coordinates, next_coordinates in train_loader:
            hidden, cell = model.init_hidden_cell(args.batch_size)
            hidden = hidden.to(device)
            cell = cell.to(device)
            coordinates, next_coordinates = coordinates.to(device), next_coordinates.to(device)
            
            optimizer.zero_grad()
            output, hidden, cell = model(coordinates, hidden, cell)
            loss = criterion(output, next_coordinates)
            loss.backward()
            optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    torch.autograd.set_detect_anomaly(True)
    for epoch in tqdm(range(num_epochs)):
        for actions, pos in actions_dataset:
            hidden = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True).to(device)
            cell = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True).to(device)
            for act, p in zip(actions, pos):
                p = p.to(device)
                act = act.to(device)
                optimizer.zero_grad()

                # Copy hidden and cell states to avoid in-place operations
                hidden = hidden.detach()
                cell = cell.detach()
                output, hidden, cell = model(p, hidden, cell)
                loss = criterion(output, act)
                # retain_graph = True if act is not actions[-1] else False
                loss.backward(retain_graph=True)
                optimizer.step()


        if (epoch + 1) %(num_epochs/10) == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            
            


    # inference

    for kk in range(1):
        inference_data=[]
        poss = init_positions = torch.from_numpy(np.array([[99.466791,28.341529,23.782218,-2.526858,-96.803282,-11.766376, 0.0]], dtype=np.float32)).to(device)
        states = []
        result = []
        h = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True).to(device)
        c = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True).to(device)
        j1, j2, j3, j4, j5, j6, t = [], [], [], [], [], [], []
        at = 0
        for i in range(60):
            predict, h, c = model(poss, h, c)
            poss = predict
            print(poss, predict)
            dt = predict[0, -1]
            predict = predict[0, 0:-1]
            act_dict = {}
            state_list = predict.tolist()
            inference_data.append(state_list[:6])
            j1.append(state_list[0])
            j2.append(state_list[1])
            j3.append(state_list[2])
            j4.append(state_list[3])
            j5.append(state_list[4])
            j6.append(state_list[5])
            dt = dt.cpu()
            dt = dt.detach().numpy()
            t.append(dt)
            at += dt
            result.append(predict)

        # 3D 플롯 설정 및 그리기
        visualize=False
        if visualize:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(j4, j5, j6, label='3D trajectory')
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            ax.legend()
            plt.savefig(f'result_3d_{kk}.png')
            plt.show()

            # t에 따른 x, y, z 값도 2D 플롯으로 각각 저장
            plt.figure()
            plt.plot(t, j4, label='x')
            plt.plot(t, j5, label='y')
            plt.plot(t, j6, label='z')
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.legend()
            plt.savefig(f'result_2d_{kk}.png')
            plt.show()

        # print(inference_data)
        testfile = os.path.join("scenarios", args.file_name+'.csv')
        with open(testfile, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for point in inference_data:
                data = {k:0. for k in fieldnames}
                data['joint1'] = point[0]
                data['joint2'] = point[1]
                data['joint3'] = point[2]
                data['joint4'] = point[3]
                data['joint5'] = point[4]
                data['joint6'] = point[5]
                writer.writerow(data)

    #eval
    #test data
    # position, 
    # test_dataset = [] 
    # for i in range(3):
    #     test_dataset.append(generate_test_scenario(True))
    
    # #train
    # criterion = nn.MSELoss()#MSEloss
    # torch.autograd.set_detect_anomaly(True)
    # losses= []
    # for epoch in range(1):
    #     for actions, pos in test_dataset:
    #         hidden = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True).to(device)
    #         cell = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True).to(device)
    #         for act, p in zip(actions, pos):
    #             p = p.to(device)
    #             act = act.to(device)

    #             # Copy hidden and cell states to avoid in-place operations
    #             hidden = hidden.detach()
    #             cell = cell.detach()
    #             output, hidden, cell = model(p, hidden, cell)
    #             loss = criterion(output, act)
    #             # retain_graph = True if act is not actions[-1] else False
    #             losses+=loss.item()


    #     if (epoch + 1) %(num_epochs/10) == 0:
    #         print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))    
    # print(losses)