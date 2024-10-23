from stable_baselines3 import PPO
from env_setup import RobotEnv
from stable_baselines3.common.envs import DummyVecEnv

# 환경 설정
env = DummyVecEnv([lambda: RobotEnv()])

# PPO 모델 생성
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)  # 학습 시간 조절

# 학습된 모델 저장
model.save("ppo_robot_lfd")
