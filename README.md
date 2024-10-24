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

xArm 확인
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
모델이 만든 시나리오.csv를 변형하는 소스코드.

#### 실행방법
```shell
python postprocess/filter_sig.py
```
scenarios/에 새로운 파일이 생성됨.

## Reinforcement Learning
WIP


