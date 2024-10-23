# PPO 모델 생성 및 학습
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 학습된 모델 저장
model.save("ppo_robot_lfd")
