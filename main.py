import subprocess
import os

class DataProcessingPipeline:
    def __init__(self, data_folder='data'):
        """
        데이터 파이프라인 클래스 초기화
        :param data_folder: 데이터를 저장할 폴더 경로
        """
        self.data_folder = data_folder
        os.makedirs(self.data_folder, exist_ok=True)
    
    #def run_show_scenario(self):
        """
        Step 1: show_scenario.py 실행
        """
        #print("\n[Step 1] Running show_scenario.py...")
        #try:
            #subprocess.run(["python", "show_scenario.py"], check=True)
            #print("show_scenario.py executed successfully.")
        #except subprocess.CalledProcessError as e:
            #print(f"Error while running show_scenario.py: {e}")
    
    def run_neokalman2(self):
        """
        Step 2: neokalman2.py 실행
        """
        print("\n[Step 2] Running neokalman2.py...")
        try:
            subprocess.run(["python", "neokalman2.py"], check=True)
            print("neokalman2.py executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error while running neokalman2.py: {e}")
    
    def run_pumping(self):
        """
        Step 3: pumping.py 실행
        """
        print("\n[Step 3] Running pumping.py...")
        try:
            subprocess.run(["python", "pumping.py"], check=True)
            print("pumping.py executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error while running pumping.py: {e}")
    
    def run_pipeline(self):
        """
        전체 데이터 파이프라인 실행 메서드
        """
        print("\n=== Starting Data Processing Pipeline ===")
        #self.run_show_scenario()
        self.run_neokalman2()
        self.run_pumping()
        print("\n=== All processes completed successfully. ===")


if __name__ == "__main__":
    # 데이터 처리 파이프라인 객체 생성 및 실행
    pipeline = DataProcessingPipeline()
    pipeline.run_pipeline()


"""
import subprocess
import os

# 데이터 폴더 생성
os.makedirs('data', exist_ok=True)

# Step 1: show_scenario.py 실행
print("Running show_scenario.py...")
subprocess.run(["python", "show_scenario.py"])

# Step 2: neokalman2.py 실행
print("Running neokalman2.py...")
subprocess.run(["python", "neokalman2.py"])

# Step 3: pumping.py 실행
print("Running pumping.py...")
subprocess.run(["python", "pumping.py"])

print("All processes completed successfully.")

"""