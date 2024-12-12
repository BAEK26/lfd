import pandas as pd

def sample_data(filepath, output_filepath, time_interval=0.1):
    """
    데이터를 지정된 시간 간격으로 샘플링.
    :param filepath: 원본 CSV 파일 경로
    :param output_filepath: 샘플링된 데이터를 저장할 CSV 파일 경로
    :param time_interval: 샘플링 시간 간격 (초 단위)
    """
    # 데이터 로드
    data = pd.read_csv(filepath)

    # timestamp 열 기준으로 샘플링
    data['timestamp'] = pd.to_numeric(data['timestamp'], errors='coerce')
    sampled_data = data.iloc[::int(time_interval * 100), :].reset_index(drop=True)

    # 샘플링된 데이터 저장
    sampled_data.to_csv(output_filepath, index=False)
    return sampled_data


if __name__ == "__main__":
    input_filepath = r'data\relative_neo_show_scenario.csv'
    output_filepath = r'data\test.csv'
    
    # 0.1초 간격으로 샘플링
    sampled_data = sample_data(input_filepath, output_filepath, time_interval=0.1)

    # 샘플링된 데이터 출력
    print(sampled_data.head())

    print(f"샘플링된 데이터가 {output_filepath}에 저장되었습니다.")
