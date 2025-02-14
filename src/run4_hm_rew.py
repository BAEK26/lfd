import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from env7_hm_rew import XArmEnv
import numpy as np
from tqdm import tqdm
import torch
import datetime
import os
import argparse

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
        self.logger.record_mean('rollout/reward', reward)
        reward_dict = self.locals.get('infos', {})[0].get('reward_dict', None)
        
        if reward_dict is not None:
            for key, value in reward_dict.items():
                self.logger.record_mean(f"rollout/{key}", value)
                
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
        self.logger.dump(self.num_timesteps)
        # reset current stats
        self.current_rewards = 0
        self.current_length = 0

# (4) env 인스턴스 생성 시 데모 활용
parser = argparse.ArgumentParser()
parser.add_argument('--demonstrations', action='store_false', default=True, help='use demonstrations')
parser.add_argument('--checkpoint_name', type=str, default=None, help='checkpoint name')
args = parser.parse_args()

train_vec_env = make_vec_env(lambda: XArmEnv(use_demonstrations=args.demonstrations, is_eval=False), n_envs=1, monitor_dir="./logs/")
eval_vec_env = make_vec_env(lambda: XArmEnv(use_demonstrations=True, is_eval=True), n_envs=1)

# GPU 설정 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
root_logdir = "./ppo_xarm_tensorboard/"
sub_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(root_logdir, sub_dir)
checkpoint_dir = "./logs/checkpoints/"

if args.checkpoint_name:
    print(f"Loading model from checkpoint: {args.checkpoint_name}")
    checkpoint = os.path.join(checkpoint_dir, args.checkpoint_name)
    model = PPO.load(checkpoint, env=train_vec_env)
else:
    print("Training model from scratch.")
    model = PPO(
        "MultiInputPolicy",
        train_vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log=log_dir
    )

eval_callback = EvalCallback(
    eval_vec_env,
    best_model_save_path="./logs/best_model/",
    log_path=f"./logs/{sub_dir}",
    eval_freq=40960,
    deterministic=True,
    render=False
)
checkpoint_callback = CheckpointCallback(
    save_freq=10240,
    save_path=checkpoint_dir,
    name_prefix="ppo_checkpoint"
)

episode_logger = EpisodeLoggerCallback()

total_timesteps = 10000000 #500000 초기 충돌지점을 보기 위함
chunk_size = 2048
current_steps = model.num_timesteps
print('current_steps = ', current_steps)

print("Starting training with tqdm...")
with tqdm(total=total_timesteps, initial = current_steps, desc="Training Steps") as pbar:
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

test_env = XArmEnv(use_demonstrations=True, is_eval=True)
print("Starting testing...")
obs, _ = test_env.reset()
_states = None
for _ in range(4096):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = test_env.step(action)
    print(f"Reward: {reward}, Info: {info}")
    if done:
        print("Episode finished. Resetting environment.")
        obs, _ = test_env.reset()
        _states = None

train_vec_env.close()
eval_vec_env.close()
test_env.close()

print("Testing complete. Environment closed.")