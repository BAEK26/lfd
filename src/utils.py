# 직접적인 쓸모는 없으나, 디버깅 과정에서 도움이 될 만한 기능을 모아두는 파일입니다.
# 앞으로 뭐가 추가될지는 미정입니다.

from src.trajectory import Trajectory
import numpy as np

def random_near_endpoints(trajectory, option='end', random_rate=0.1):
    """
    시작점(start)과 도착점(end) 사이의 거리를 100%로 했을 때,
    시작점 인근 10% 이내 또는 도착점 인근 10% 이내에 있는 임의의 점을 반환하는 함수.
    
    :param start: (x, y, z) 형태의 시작점 배열
    :param end: (x, y, z) 형태의 도착점 배열
    :param option: 'start' 또는 'end'를 선택하여 반환할 점의 위치를 결정
    :return: 선택된 범위 내의 임의의 점 (x, y, z)
    """
    start = trajectory.xyz[0]
    end = trajectory.xyz[-1]
    
    # 시작점과 도착점 사이의 거리 계산
    distance = np.linalg.norm(end - start)
    radius = random_rate * distance
    
    # 선택한 점의 중심 설정
    if option == 'start':
        center = start
    elif option == 'end':
        center = end
    else:
        raise ValueError("option must be 'start' or 'end'")
    
    while True:
        random_offset = np.random.uniform(-radius, radius, size=3)
        random_point = center + random_offset
        if np.linalg.norm(random_offset) <= radius:
            break
    
    return tuple(random_point)
