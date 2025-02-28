#include "HX711.h"

// HX711 모듈의 핀 정의
const int LOADCELL_DOUT_PIN = 3; // DT 핀
const int LOADCELL_SCK_PIN = 2;  // SCK 핀

HX711 scale;
float calibration_factor = 1.0; // 초기 보정값

void setup() {
  Serial.begin(9600);
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  
  Serial.println("HX711 무게 감지 예제");
  
  // 초기 영점 설정 (로드셀 위에 아무것도 없는 상태에서)
  Serial.println("영점 설정 중...");
  scale.set_scale(); 
  scale.tare(); // 영점 조정
  Serial.println("영점 설정 완료");
  
  // 무게추를 올려달라고 요청
  Serial.println("정확한 무게의 무게추를 로드셀에 올려주세요.");
  Serial.println("올린 무게를 (g 단위로) 시리얼 모니터에 입력하고 엔터를 눌러주세요.");
}

void loop() {
  // 시리얼 입력이 있다면 보정 모드
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.length() > 0) {
      float known_weight = input.toFloat();
      
      if (known_weight > 0) {
        Serial.print("입력된 무게: ");
        Serial.print(known_weight);
        Serial.println(" g");
        
        // 보정값 계산
        calibration_factor = scale.get_units(10) / known_weight;
        scale.set_scale(calibration_factor);
        
        Serial.print("보정 완료! 새로운 보정값: ");
        Serial.println(calibration_factor, 5);
        
        Serial.println("이제부터 무게를 측정하여 출력합니다.");
      } else {
        Serial.println("올바른 무게를 입력해주세요 (숫자, g 단위).");
      }
    }
  }
  
  // 보정이 완료된 후 무게 출력
  if (calibration_factor != 1.0) {
    if (scale.is_ready()) {
      float weight = scale.get_units(5); // 평균 5회 측정
      Serial.print("측정된 무게: ");
      Serial.print(weight, 2); // 소수점 두 자리까지 표시
      Serial.println(" g");
    } else {
      Serial.println("HX711을 찾을 수 없습니다.");
    }
    delay(500);
  }
}
