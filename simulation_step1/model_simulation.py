import torch
import argparse
import datetime
from random_d import *
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F
dtype = torch.float
#lstm
class SimpleLearner(nn.Module): 
    def __init__(self, args):
        super(SimpleLearner, self).__init__()
        self.state = State(**{"x":0, "y":0, "z": 0, "t": 0})
        self.hidden_size = args.hidden_size

        self.lstm = nn.LSTM(input_size=args.input_size, hidden_size=args.hidden_size, dropout=0.2)
        self.W = nn.Parameter(torch.randn([args.hidden_size, args.output_size]).type(dtype))
        self.b = nn.Parameter(torch.randn([args.output_size]).type(dtype))
        self.final = nn.Linear(self.hidden_size, args.output_size, bias= True)

        self.activateion = nn.ReLU()    

    def forward(self,X, hidden, cell):
        
        # Add batch dimension to input
        X = X.unsqueeze(0)
        outputs, (hidden, cell) = self.lstm(X, (hidden, cell))
        outputs = outputs[-1]  # 최종 예측 Hidden Layer
        # print("Shape of outputs:", outputs.shape)
        # print("Shape of self.W:", self.W.shape)

        # model = torch.mm(outputs, self.W) + self.b  # 최종 예측 최종 출력 층
        model = self.final(outputs)
        # model = self.activateion(model)
        return model, hidden, cell

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)        


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', action='store', type=int, default=64, help='hidden_size', required=False)
    parser.add_argument('--input_size', action='store', type=int, default=4, help='input_size', required=False) #x, y, z, t
    parser.add_argument('--output_size', action='store', type=int, default=4, help='output_size', required=False) #dx, dy, dz, dt
    parser.add_argument('--batch_size', action='store', type=int, default=1, help='output_size', required=False) #chunk?
    parser.add_argument('--epochs', action='store', type=int, default=10, help='output_size', required=False) 
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
    # position, 
    actions_dataset = [] 
    for i in range(10):
        actions_dataset.append(generate_random_scenario(True))
    
    # print('actions',actions_dataset[0][0])
    # print('pos',actions_dataset[0][1])

    #train
    model = SimpleLearner(args).to(device)
    # criterion = nn.L1Loss()#MSEloss
    criterion = nn.MSELoss()#MSEloss
    # optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = args.epochs
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

    for kk in range(0):
        poss = init_positions = torch.from_numpy(np.array([[0., 0., 0., 0.]], dtype=np.float32)).to(device)
        states = []
        for pos in poss:
            state_dict = {}
            for loc, val in zip(['x', 'y', 'z', 't'], pos):
                state_dict[loc] = val
            states.append(state_dict)
        for state_dict in states:
            Testbed = State(**state_dict)
        result = []
        h = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True).to(device)
        c = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True).to(device)
        x = []
        y = []
        z = []
        t = []
        at = 0
        for i in range(50):
            predict, h, c = model(poss, h, c)
            poss += predict
            print(poss, predict)
            dt = predict[0, -1]
            predict = predict[0, 0:-1]
            predict = predict.tolist()
            act_dict = {}
            for loc, val in zip(['dx', 'dy', 'dz'], predict):
                act_dict[loc] = float(val)
            action = Action(**act_dict)
            Testbed.act(action, dt)
            state_list = Testbed.to_list()
            x.append(state_list[0])
            y.append(state_list[1])
            z.append(state_list[2])
            dt = dt.cpu()
            dt = dt.detach().numpy()
            t.append(at + dt)
            at += dt
            result.append(predict)

        # 3D 플롯 설정 및 그리기
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, label='3D trajectory')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.legend()
        plt.savefig(f'result_3d_{kk}.png')
        plt.show()

        # t에 따른 x, y, z 값도 2D 플롯으로 각각 저장
        plt.figure()
        plt.plot(t, x, label='x')
        plt.plot(t, y, label='y')
        plt.plot(t, z, label='z')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()
        plt.savefig(f'result_2d_{kk}.png')
        plt.show()

    #eval
    #test data
    # position, 
    test_dataset = [] 
    for i in range(3):
        test_dataset.append(generate_test_scenario(True))
    
    #train
    criterion = nn.MSELoss()#MSEloss
    torch.autograd.set_detect_anomaly(True)
    losses= []
    for epoch in range(1):
        for actions, pos in test_dataset:
            hidden = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True).to(device)
            cell = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True).to(device)
            for act, p in zip(actions, pos):
                p = p.to(device)
                act = act.to(device)

                # Copy hidden and cell states to avoid in-place operations
                hidden = hidden.detach()
                cell = cell.detach()
                output, hidden, cell = model(p, hidden, cell)
                loss = criterion(output, act)
                # retain_graph = True if act is not actions[-1] else False
                losses+=loss.item()


        if (epoch + 1) %(num_epochs/10) == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))    
    print(losses)