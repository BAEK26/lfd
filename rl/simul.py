# 모델 불러오기 및 테스트
model = PPO.load("ppo_robot_lfd")

obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
