from stable_baselines3 import PPO
from env_setup import RobotEnv
from stable_baselines3.common.envs import DummyVecEnv

# 모델 불러오기
model = PPO.load("ppo_robot_lfd")

# 환경 설정
env = DummyVecEnv([lambda: RobotEnv()])

# 시뮬레이션 실행
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
