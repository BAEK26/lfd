import torch
import datetime
import numpy as np
import random
from pydantic import BaseModel
from matplotlib import pyplot as plt
random.seed()
DEMO_SPEED = 1/2 # 1m/2s
class State(BaseModel):
    x : float
    y : float
    z : float
    t : float

    def act(self, action, deltime):
        self.x += action.dx
        # self.x += action[0]
        self.y += action.dy
        # self.y += action[1]
        self.z += action.dz
        # self.z += action[0]
        self.t += deltime
    def elapse(self, dt):
        self.t += dt
    def to_list(self):
        return [self.x, self.y, self.z, self.t]

class Action(BaseModel):
    dx : float
    dy : float
    dz : float
    def to_list(self):
        return [self.dx, self.dy, self.dz]

# class SandA(BaseModel):
#     state : State
#     actions : Action

def generate_xplus_action(state, now):
    state.t = now
    deltime = random.gauss(mu=1.0, sigma=0.5)
    deltime = abs(deltime)
    delx, dely, delz = random.gauss(mu=DEMO_SPEED*deltime, sigma=DEMO_SPEED/2*deltime), 0, 0
    if state.x > 50:
        delx = 0
    actions = Action(dx=delx, dy=dely, dz=delz)
    return actions, deltime

def generate_dir_action(state, now, dir=1):
    state.t = now
    deltime = random.gauss(mu=1.0, sigma=0.5)
    deltime = abs(deltime)
    delx, dely, delz = random.gauss(mu=DEMO_SPEED*deltime, sigma=DEMO_SPEED/2*deltime), random.gauss(mu=DEMO_SPEED*deltime, sigma=DEMO_SPEED/2*deltime), random.gauss(mu=DEMO_SPEED*deltime, sigma=DEMO_SPEED/2*deltime)
    if state.x > 50:
        delx = 0
    actions = Action(dx=delx*dir, dy=dely*dir, dz=delz*dir)
    return actions, deltime

def generate_random_scenario(save=None) -> list:
    x_dir = +1
    positions = []
    actions = []
    save_times = []
    save_xs=[]
    save_accum=0
    state = State(**{"x":0, "y":0, "z": 0, "t": 0})
    for i in range(50):
        positions.append(torch.tensor([state.to_list()]))
        action, deltime = generate_dir_action(state, save_accum, x_dir)
        # action, deltime = generate_xplus_action(state, save_accum)
        if action.dx == 0:
            continue
        actions.append(torch.tensor([action.to_list()+[deltime]]))
        state.act(action, deltime)
        save_times.append(save_accum+deltime)
        save_xs.append(state.x)
        save_accum+=deltime
        if state.x > 15:
            x_dir = -1
    
    if save:
        plt.plot(save_times,save_xs)
        # plt.savefig(f'sc-{random.randint(0,1000000)}.png')
        plt.savefig(f'sc.png')
    return [actions, positions]


if __name__ == "__main__":
    generate_random_scenario(True)
    current = State(**{"x":0, "y":0, "z": 0, "t": 0.})
    actions = []
    times = []
    xs=[]
    dts=[]
    accum_time=0
    x = 0
    for i in range(10):
        action, deltime = generate_dir_action(current, current.t, +1)
        current.act(action, deltime)
        actions.append(action.to_list())
        times.append(accum_time+deltime)
        dts.append(deltime)
        accum_time+=deltime
        xs.append(current.x)

    print(actions) #[(dx, dy, dz)]
    print(times) #[dt]
    print()
    for daction, dt in zip(actions, dts):
        print(daction[0]/dt, end=" ")
    print()
    plt.plot(times,xs)
    plt.show()
