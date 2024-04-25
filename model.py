import torch
import argparse
import datetime
from random_d import *
import torch.nn as nn
from torch.nn import functional as F
dtype = torch.float
#lstm
class SimpleLearner(nn.Module): 
    def __init__(self, args):
        super(SimpleLearner, self).__init__()
        self.state = State(**{"x":0, "y":0, "z": 0, "t": datetime.datetime.now()})
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
    parser.add_argument('--input_size', action='store', type=int, default=3, help='input_size', required=False) #x, y, z
    parser.add_argument('--output_size', action='store', type=int, default=4, help='output_size', required=False) #dt, dx, dy, dz
    parser.add_argument('--batch_size', action='store', type=int, default=1, help='output_size', required=False) #chunk?
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
    actions_dataset = [] 

    #train
    model = SimpleLearner(args).to(device)
    criterion = nn.L1Loss()#MSEloss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        for actions, pos in actions_dataset:
            hidden = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True)
            cell = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True)
            for act, p in zip(actions, pos):
                output, hidden, cell = model(p, (hidden, cell))
                loss = criterion(output, act)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            
            
    #inference
    poss = init_positions = torch.from_numpy(np.array([[0.,0.,0.]], dtype=np.float32))
    states = []
    for pos in poss:
        state_dict = {}
        for loc, val in zip(['x', 'y', 'z'], pos):
            state_dict[loc] = val
        state_dict['t'] = datetime.datetime.now()
        states.append(state_dict)
    for state_dict in states:
        Testbed = State(**state_dict)
    result =[]
    h = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True)
    c = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True)
    x = []
    t = []
    for i in range(15):
        predict, h, c = model(poss, h, c)
        predict = predict[0, 1:]

        predict = predict.tolist()
        act_dict={}
        for loc, val in zip(['dx','dy','dz'], predict):
            act_dict[loc]=float(val)
        action = Action(**act_dict)
        Testbed.act(action)
        x.append(Testbed.to_list()[0])
        # t.append(Testbed.to_list()[-1])
        t.append(i)
        result.append(predict)
    print(result)
    plt.plot(t,x)
    plt.show()
