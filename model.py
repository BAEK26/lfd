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
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,X, hidden, cell):
        
        # Add batch dimension to input
        X = X.unsqueeze(0)
        outputs, (hidden, cell) = self.lstm(X, (hidden, cell))
        outputs = outputs[-1]  # 최종 예측 Hidden Layer
        # print("Shape of outputs:", outputs.shape)
        # print("Shape of self.W:", self.W.shape)

        model = torch.mm(outputs, self.W) + self.b  # 최종 예측 최종 출력 층
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
    parser.add_argument('--hidden_size', action='store', type=int, default=32, help='hidden_size', required=False)
    parser.add_argument('--input_size', action='store', type=int, default=4, help='input_size', required=False) #x, y, z, t
    parser.add_argument('--output_size', action='store', type=int, default=4, help='output_size', required=False) #dx, dy, dz, dt
    parser.add_argument('--batch_size', action='store', type=int, default=1, help='output_size', required=False) #chunk?
    parser.add_argument('--epochs', action='store', type=int, default=100, help='output_size', required=False) 
    args = parser.parse_args()
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")

    #data
    # position, 
    actions_dataset = [] 
    for i in range(5):
        actions_dataset.append(generate_random_scenario(True))
    
    # print('actions',actions_dataset[0][0])
    # print('pos',actions_dataset[0][1])

    #train
    model = SimpleLearner(args).to(device)
    criterion = nn.L1Loss()#MSEloss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = args.epochs
    for epoch in tqdm(range(num_epochs)):
        for actions, pos in actions_dataset:
            optimizer.zero_grad()
            hidden = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True)
            cell = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True)
            # data_x = 0
            # model_x = 0
            # scenario_len = len(actions[0])
            for act, p in zip(actions, pos):
                # data_x += act[0][0]
                p = p.to(device)
                act = act.to(device)
                output, hidden, cell = model(p, hidden, cell)
                # model_x += output[0][0]
                if epoch == 50 and act is pos[-1]:
                    print('p',p)
                    print('output',output)
                    print('act',act)
                    print()
                loss = criterion(output, act)
                retain_graph = True if act is not actions[-1] else False
                loss.backward(retain_graph=retain_graph)
            # if epoch == 50:
            #     exit()
            # print('data_average: ', data_x/scenario_len, 'model_average: ', model_x/scenario_len)
            optimizer.step()


        if (epoch + 1) %(num_epochs/10) == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            
            
    #inference
    for kk in range(2):
        poss = init_positions = torch.from_numpy(np.array([[0.,0.,0., 0.]], dtype=np.float32))
        states = []
        for pos in poss:
            state_dict = {}
            for loc, val in zip(['x', 'y', 'z', 't'], pos):
                state_dict[loc] = val
            states.append(state_dict)
        for state_dict in states:
            Testbed = State(**state_dict)
        result =[]
        h = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True)
        c = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True)
        x = []
        t = []
        at = 0
        for i in range(50):
            predict, h, c = model(poss, h, c)
            dt = predict[0,-1]
            predict = predict[0, 0:-1]
            predict = predict.tolist()
            print(predict)
            act_dict={}
            for loc, val in zip(['dx','dy','dz'], predict):
                act_dict[loc]=float(val)
            action = Action(**act_dict)
            Testbed.act(action, dt)
            x.append(Testbed.to_list()[0])
            # t.append(Testbed.to_list()[-1])
            dt = dt.detach().numpy()
            t.append(at + dt)
            at += dt
            result.append(predict)
        # print(result)
        plt.plot(t,x)
        plt.savefig('result.png')
