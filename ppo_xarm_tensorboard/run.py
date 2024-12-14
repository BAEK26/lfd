import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from xarm_env import XArmEnv  # 사용자 정의 환경

# 사용자 정의 환경 생성
env = XArmEnv()

# Stable Baselines3에서 사용할 벡터화된 환경 생성
# 실제 환경에서는 n_envs=1로 설정
vec_env = make_vec_env(lambda: env, n_envs=1)

# PPO 모델 초기화
model = PPO(
    "MultiInputPolicy",  # 다층 퍼셉트론 정책 (와우 이거 Lstm으로도 바꿀수 있다네)
    #"MlpLstmPolicy"
    vec_env,  # 벡터화된 환경
    learning_rate=0.0003,  # 학습률
    n_steps=512,  #2048? 학습 단위 스텝 수 (실제 환경에서는 더 작게 설정)
    batch_size=64,  # 배치 크기
    gamma=0.99,  # 할인율
    verbose=1,  # 학습 출력
    tensorboard_log="./ppo_xarm_tensorboard/"  # 텐서보드 로그 디렉토리
)

# 학습 콜백 설정
# 학습 중간에 평가 및 최적 모델 저장
eval_callback = EvalCallback(
    env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    render=False
)

# 체크포인트 저장 콜백 (중간 학습 상태 저장)
checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # 매 10,000 스텝마다 저장
    save_path="./logs/checkpoints/",
    name_prefix="ppo_checkpoint"
)

# 모델 학습
print("Starting training...")
model.learn(total_timesteps=50000, callback=[eval_callback, checkpoint_callback])  # 학습 총 스텝 수 설정
print("Training complete.")

# 학습된 모델 저장
model.save("ppo_xarm_policy")
print("Model saved as 'ppo_xarm_policy'.")

# 학습된 모델 로드 및 테스트
print("Loading the saved model...")
model = PPO.load("ppo_xarm_policy")

# 테스트
print("Starting testing...")
obs = env.reset()[0]
_states = None
for _ in range(1000):#테스트 루프
    action, _states = model.predict(obs, deterministic=True)  # 결정론적 행동 예측, 상태 추적
    obs, reward, done,_, info = env.step(action) # 환경에 행동 적용 (수행한다고)
    print(f"Reward: {reward}, Info: {info}") # 보상과 추가정보 출력

    if done: # 에피소드 완료하면 재설정
        print("Episode finished. Resetting environment.")
        obs = env.reset()[0]
        _states = None # 상태 초기화

# 환경 종료
env.close()
print("Testing complete. Environment closed.")
