# lfd

## Supervised Learning
manual 모드로 xArm을 직접 시연하여 얻은 데이터를 학습합니다.

### 실행방법
**데이터셋 구축**
```shell
python show_scenario.py --file_name
```
n번간 시연과 Ctrl+c를 반복하여
n개의 시나리오 파일(data/robot_data_{i}.csv)을 만듦 (i : 0 ~ n-1)

**모델 학습**
```shell
air.py --file_name test_scenario #생성되는 테스트 시나리오 파일명
        --model_path path/to/model.pt # 모델을 불러옴. 쓰지 않으면 새로 만듦.
       --train  # 모델을 학습함.
       --visualize  # 학습, 테스트에 대한 결과를 시각화함.
       --save_model_at path/to/new_model.pt # 모델 저장 위치
```

학습 후 scenarios/test_scenario.csv가 생성됨.

**xArm Robot Arm 시연**
```shell
python run_scenarioFile.py --file_name test_scenario # 실행하고자하는 시나리오 파일명
```

> deprecated   
> ~~모델 학습~~
>```shell
>#bash train.sh
>```

### Postprocess on Scenarios
posprocess/*.py에 대한 설명.

#### Overview
모델이 만든 {시나리오.csv}를 변형하는 소스코드.

#### 실행방법
```shell
python postprocess/filter_sig.py
```
scenarios/*-sig.csv 새로운 파일이 생성됨.

## Reinforcement Learning
WIP

### env list
pip install \
 absl-py==2.1.0 \
 colorama==0.4.6 \
 contourpy==1.3.1 \
 cycler==0.12.1 \
 etils==1.11.0 \
 farama-notifications==0.0.4 \
 filelock==3.16.1 \
 fonttools==4.55.3 \
 fsspec==2024.10.0 \
 glfw==2.8.0 \
 grpcio==1.68.1 \
 gym==0.26.2 \
 gym-notices==0.0.8 \
 gymnasium==1.0.0 \
 gymnasium-robotics==1.3.1 \
 imageio==2.36.1 \
 importlib-resources==6.4.5 \
 jinja2==3.1.4 \
 joblib==1.4.2 \
 kiwisolver==1.4.7 \
 markdown==3.7 \
 markupsafe==3.0.2 \
 matplotlib==3.10.0 \
 mpmath==1.3.0 \
 mujoco==3.1.6 \
 networkx==3.4.2 \
 numpy==1.26.4 \
 packaging==24.2 \
 pandas==2.2.3 \
 pettingzoo==1.24.3 \
 pillow==11.0.0 \
 protobuf==5.29.1 \
 pykalman==0.9.7 \
 pyopengl==3.1.7 \
 pyparsing==3.2.0 \
 python-dateutil==2.9.0.post0 \
 pytz==2024.2 \
 scikit-learn==1.6.0 \
 scipy==1.14.1 \
 six==1.16.0 \
 stable-baselines3==2.4.0 \
 sympy==1.13.1 \
 tensorboard==2.18.0 \
 tensorboard-data-server==0.7.2 \
 threadpoolctl==3.5.0 \
 torch==2.5.1+cu121 \
 torchaudio==2.5.1+cu121 \
 torchvision==0.20.1+cu121 \
 tqdm==4.67.1 \
 typing-extensions==4.12.2 \
 zipp==3.21.0 \
 xarm-python-sdk==1.14.7

