import torch
import argparse
import datetime
import data_utils
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


        #     a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
        #     total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        #     loss_dict = dict()
        #     all_l1 = F.l1_loss(actions, a_hat, reduction='none') 
        # #MSELoss(size_average=None, reduce=None, reduction='mean')
        # #         loss = nn.MSELoss()
        # # input = torch.randn(3, 5, requires_grad=True)
        # # target = torch.randn(3, 5)
        # # output = loss(input, target)
        # # output.backward()
        #     l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        #     loss_dict['l1'] = l1
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', action='store', type=int, default=64, help='hidden_size', required=False)
    parser.add_argument('--input_size', action='store', type=int, default=7, help='input_size', required=False) #joint1~6, t
    parser.add_argument('--output_size', action='store', type=int, default=7, help='output_size', required=False) #joint1~6, dt
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
    actions_dataset = data_utils.load_dataset_list()
    
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

    for kk in range(1):
        poss = init_positions = torch.from_numpy(np.array([[112.963614, 3.94682, 15.685751, -8.833004, -75.773611, 3.087154, 0.]], dtype=np.float32)).to(device)
        states = []
        result = []
        h = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True).to(device)
        c = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True).to(device)
        j1, j2, j3, j4, j5, j6, t = [], [], [], [], [], [], []
        at = 0
        for i in range(50):
            predict, h, c = model(poss, h, c)
            poss = predict
            print(poss, predict)
            dt = predict[0, -1]
            predict = predict[0, 0:-1]
            predict = predict.tolist()
            act_dict = {}
            state_list = predict.to_list()
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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(j1, j2, j3, label='3D trajectory')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.legend()
        plt.savefig(f'result_3d_{kk}.png')
        plt.show()

        # t에 따른 x, y, z 값도 2D 플롯으로 각각 저장
        plt.figure()
        plt.plot(t, j1, label='x')
        plt.plot(t, j2, label='y')
        plt.plot(t, j3, label='z')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()
        plt.savefig(f'result_2d_{kk}.png')
        plt.show()

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