from kinesthetic_teaching import KinestheticTeaching
from model import RobotPPOModel
from env import RobotEnv
import params

# 경로 기록
kinesthetic_teaching = KinestheticTeaching()
kinesthetic_teaching.record_trajectory()

# 학습 환경 초기화
env = RobotEnv()

# 강화 학습 모델 정의
model = RobotPPOModel(env)
model.train(timesteps=params['timesteps'])

# 학습된 모델 저장
model.save(params['model_save_path'])

# 시뮬레이션 실행
kinesthetic_teaching.replay_trajectory(params['trajectory_file'])
