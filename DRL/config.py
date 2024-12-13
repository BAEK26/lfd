# config.py

# 하이퍼파라미터 설정
HYPERPARAMS = {
    # 공통 하이퍼파라미터
    "gamma": 0.99,              # 할인율
    "learning_rate": 0.0005,    # 학습률
    "episodes": 500,            # 총 에피소드 수
    "max_timesteps": 300,       # 한 에피소드의 최대 타임스텝

    # DQN 전용 하이퍼파라미터
    "epsilon_start": 1.0,       # 초기 탐험 비율
    "epsilon_decay": 0.995,     # 탐험 비율 감소율
    "epsilon_min": 0.01,        # 최소 탐험 비율
    "batch_size": 64,           # 미니배치 크기
    "buffer_limit": 10000,      # 재플레이 버퍼 최대 크기

    # A2C 전용 하이퍼파라미터
    "entropy_beta": 0.01        # 엔트로피 보너스 가중치
}
