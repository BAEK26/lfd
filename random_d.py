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
    t : datetime.datetime

    def act(self, action):
        self.x += action.dx
        # self.x += action[0]
        self.y += action.dy
        # self.y += action[1]
        self.z += action.dz
        # self.z += action[0]
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

def generate_random_action(state, now):
    state.t = now
    # deltime = np.random.normal(1, 0.5, size=(1,))
    deltime = random.gauss(mu=1.0, sigma=0.5)
    # deltime = datetime.datetime.now()-now
    # delx, dely, delz = np.random.normal(DEMO_SPEED*deltime, DEMO_SPEED/2*deltime, size=(1,)), np.random.normal(0,0,size=(1,)), np.random.normal(0,0,size=(1,))
    delx, dely, delz = random.gauss(mu=DEMO_SPEED*deltime, sigma=DEMO_SPEED/2*deltime), 0, 0
    # delx, dely, delz = DEMO_SPEED*deltime, 0, 0
    if state.x > 5:
        delx = 0
    actions = Action(dx=delx, dy=dely, dz=delz)
    # actions = np.array([delx, dely, delz])
    return actions, deltime


if __name__ == "__main__":
    current = State(**{"x":0, "y":0, "z": 0, "t": datetime.datetime.now()})
    actions = []
    times = []
    xs=[]
    dts=[]
    accum_time=0
    x = 0
    for i in range(10):
        action, deltime = generate_random_action(current, datetime.datetime.now())
        current.act(action)
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