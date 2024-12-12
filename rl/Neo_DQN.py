import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

class ReplayBuffer(): #경험 재플레이 버퍼를 구현함
    def __init__(self): # 여기서 포인트는 버퍼를 최대 크기로 설정해서 오래된 경험을 자동제거
        self.buffer = collections.deque(maxlen=buffer_limit) #경험 저장하는 큐 초기화
    
    def put(self, transition): # 상태, 행동, 보상, 다음 상태 완료 여부를 하나의 튜플(transition)로 받아 버퍼에 저장
        self.buffer.append(transition) # 현재 버퍼에 저장된 경험 개수 반환
    
    def sample(self, n): # 버퍼에서 무작위로 n개의 샘플을 추출
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch: # 
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
                #데이터타입 텐서로 바꾸기
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

# Qnet은 DQN에서 쓰이는 신경망을 정의한 클래스. 
# 상태정보를 입력으로 받아 각 행동의 Q값을 출력함.
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128) # 입력크기 4가 카트폴 환경 "상태" 차원
        self.fc2 = nn.Linear(128, 128) 
        self.fc3 = nn.Linear(128, 2) # 출력 크기 2는 카트폴 환경 "행동" 차원

    def forward(self, x): # 그냥 레이어 가지고 순전파함.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon): # 상태 obs랑 탐험 - 활용 균형을 위한 e-greedy 로 행동선택 
        out = self.forward(obs)
        coin = random.random() # e 확률로 랜덤 행동., 1-e 확률로 가장 높은 Q-값을 가진 행동 선택 (이게바로 그리디)
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()

     #train 하면서 뭘 배우냐면 Q-함수의 업데이트를 담당하게 됨.
     # input : # 현재 Q-네트워크 q와 타겟 Q-네트워크 q_target, 경험 재플레이 버퍼 memory, optimizer       
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size) # 메모리에서 배치 크기만큼 샘플 가져오기

        q_out = q(s) 
        q_a = q_out.gather(1,a) #Q(s,a) 계산하기
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask #타겟 Q-값 계산하기 
        loss = F.smooth_l1_loss(q_a, target) #스무딩하고, 네트워크 업데이트
        
        optimizer.zero_grad()
        loss.backward() # 그래디언트 역전파 
        optimizer.step()

def main():
    env = gym.make('CartPole-v1') # 환경 만들고
    q = Qnet() # Q 네트워크랑 타겟 Q-네트워크 초기화하기
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer() # 경험 재플레이 버퍼 생성하기 (반복해서 학습해야하니께)

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000): # 에프소드 반복하기. 
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s, _ = env.reset() # 항상 초기화 코드는 넣기! 
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask)) 
            # 선택된 행동에서 (s,a,r,s',done) 데이터 생성하고 버퍼에 저장
            s = s_prime

            score += r
            if done:
                break # 게임 종료될때까지 행동 선택하고 환경 진행하도록
            
        if memory.size()>2000: # 단, 일정 조건을 만족해야지 (2000번 초과) train 함수 호출해서 네트워크 학습
            train(q, q_target, memory, optimizer) # 왜냐고? 초기에는 탐험을 많이해야 로컬 최적화 막아서. 학습 진행될수록 활용 많이 하도록

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()



"""주요 흐름"""

"""
환경 초기화 및 네트워크 구성.
에피소드 단위로 상태 초기화 후 행동 선택 및 환경 진행.
경험 데이터 저장 및 일정 조건 만족 시 네트워크 학습.
타겟 네트워크를 주기적으로 업데이트.
결과를 출력하며 학습 진행 상황 확인.

ReplayBuffer: 경험 재플레이 버퍼로 데이터를 저장하고 샘플링.
Qnet: Q-함수를 근사하는 신경망.
train: Q-함수를 학습시키는 과정.
main: 전체 학습 프로세스 제어.

"""