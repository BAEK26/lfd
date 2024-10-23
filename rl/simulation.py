from model import RobotPPOModel
from env import RobotEnv

env = RobotEnv()

# 모델 불러오기
model = RobotPPOModel(env)
model.load('ppo_robot_model')

# 시뮬레이션 실행
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break
