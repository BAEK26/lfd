import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from env4 import XArmEnv
import numpy as np
from tqdm import tqdm

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

        # Reset stats
        self.current_rewards = 0
        self.current_length = 0


class RLTrainer:
    def __init__(self):
        print("Initializing RL Trainer...")
        self.env = XArmEnv(use_demonstrations=True)
        self.vec_env = make_vec_env(lambda: self.env, n_envs=1)
        self.total_timesteps = 50000
        self.chunk_size = 5000
        self.current_steps = 0

        self.model = PPO(
            "MultiInputPolicy",
            self.vec_env,
            learning_rate=0.0003,
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./ppo_xarm_tensorboard/"
        )

        self.eval_callback = EvalCallback(
            self.env,
            best_model_save_path="./logs/best_model/",
            log_path="./logs/",
            eval_freq=5000,
            deterministic=True,
            render=False
        )

        self.checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./logs/checkpoints/",
            name_prefix="ppo_checkpoint"
        )

        self.episode_logger = EpisodeLoggerCallback()

    def train(self):
        print("Starting training with tqdm...")
        with tqdm(total=self.total_timesteps, desc="Training Steps") as pbar:
            while self.current_steps < self.total_timesteps:
                steps_to_do = min(self.chunk_size, self.total_timesteps - self.current_steps)
                self.model.learn(
                    total_timesteps=steps_to_do,
                    callback=[self.eval_callback, self.checkpoint_callback, self.episode_logger],
                    reset_num_timesteps=False
                )
                self.current_steps += steps_to_do
                pbar.update(steps_to_do)

        print("Training complete. Model is being saved...")
        self.model.save("ppo_xarm_policy")

    def test(self):
        print("Loading the saved model...")
        model = PPO.load("ppo_xarm_policy")
        obs, _ = self.env.reset()
        _states = None
        print("Starting testing...")
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = self.env.step(action)
            print(f"Reward: {reward}, Info: {info}")
            if done:
                print("Episode finished. Resetting environment.")
                obs, _ = self.env.reset()
                _states = None

        self.env.close()
        print("Testing complete. Environment closed.")


if __name__ == "__main__":
    trainer = RLTrainer()
    trainer.train()
    trainer.test()
