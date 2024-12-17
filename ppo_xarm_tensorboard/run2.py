import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from xarm_env import XArmEnv
import numpy as np
from tqdm import tqdm

# (4) 여기에 이전 run.py 대비 변화를 표시:
# - EpisodeLoggerCallback 유지
# - tqdm 반복문 유지
# - demonstration 사용 가능 (env 생성 시 use_demonstrations=True)
# - run2.py로 파일명 변경

class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.current_rewards = 0
        self.current_length = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.current_rewards += reward
        self.current_length += 1
        return True

    def _on_episode_end(self):
        self.episode_rewards.append(self.current_rewards)
        self.episode_lengths.append(self.current_length)

        info = self.locals["infos"][0]
        is_success = info.get("is_success", False)
        self.episode_success.append(int(is_success))

        # 텐서보드 기록
        self.logger.record('rollout/episode_reward', self.current_rewards)
        self.logger.record('rollout/episode_length', self.current_length)
        recent_success_rate = np.mean(self.episode_success[-100:]) if len(self.episode_success) > 0 else 0.0
        self.logger.record('rollout/success_rate_100', recent_success_rate)

        # reset current stats
        self.current_rewards = 0
        self.current_length = 0

# (4) env 인스턴스 생성 시 데모 활용
env = XArmEnv(use_demonstrations=True)

vec_env = make_vec_env(lambda: env, n_envs=1)

model = PPO(
    "MultiInputPolicy",
    vec_env,
    learning_rate=0.0003,
    n_steps=512,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./ppo_xarm_tensorboard/"
)

eval_callback = EvalCallback(
    env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    render=False
)
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./logs/checkpoints/",
    name_prefix="ppo_checkpoint"
)

episode_logger = EpisodeLoggerCallback()

total_timesteps = 50000
chunk_size = 5000
current_steps = 0

print("Starting training with tqdm...")
with tqdm(total=total_timesteps, desc="Training Steps") as pbar:
    while current_steps < total_timesteps:
        steps_to_do = min(chunk_size, total_timesteps - current_steps)
        model.learn(total_timesteps=steps_to_do, 
                    callback=[eval_callback, checkpoint_callback, episode_logger],
                    reset_num_timesteps=False)
        current_steps += steps_to_do
        pbar.update(steps_to_do)

print("Training complete.")
model.save("ppo_xarm_policy")
print("Model saved as 'ppo_xarm_policy'.")

print("Loading the saved model...")
model = PPO.load("ppo_xarm_policy")

print("Starting testing...")
obs, _ = env.reset()
_states = None
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    print(f"Reward: {reward}, Info: {info}")
    if done:
        print("Episode finished. Resetting environment.")
        obs, _ = env.reset()
        _states = None

env.close()
print("Testing complete. Environment closed.")
