from DMP8 import DMPTrainer, DMPDataLoader
from env4 import XArmEnv
from run2 import RLTrainer
import os

class Main:
    def __init__(self):
        pass

    def main(self):
        print("=== Starting DMP Training ===")
        #DMP 파일 위치
        base_dir = os.path.abspath(os.path.dirname((__file__)))
        data_file_path = os.path.join(base_dir, 'data', 'pumping_interpolated_trajectory.csv')
        
        #DMP 데이터 로드, 인풋 아웃풋
        data_loader = DMPDataLoader()
        data_loader.load_and_preprocess()
        train_input, train_output = data_loader.get_training_data() 

        #DMP 학습
        dmp_trainer = DMPTrainer()
        dmp_trainer.train(train_input, train_output)
        dmp_trainer.save_model()

        print("\n=== Running XArm Environment ===")
        #RL 엔브 먼저 정렬: 초기화
        xarm_env = XArmEnv(use_demonstrations=True)
        obs, _ = xarm_env.reset()
        print(f"Initial Observation : {obs}")

        print("\n=== Starting RL Training ===")
        rl_trainer = RLTrainer()
        rl_trainer.train()

        print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    m = Main()
    m.main()


"""
from DMP7 import train_dmp
from env4 import run_xarm_env
from run2 import train_rl

def main():
    print("=== Starting All Processes ===")
    
    # DMP 모델 학습 실행
    train_dmp()
    
    # XArm 환경 실행
    run_xarm_env()
    
    # RL 학습 실행
    train_rl()
    
    print("=== All Tasks Completed Successfully ===")

if __name__ == "__main__":
    main()

    """