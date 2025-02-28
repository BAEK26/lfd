#include "HX711.h"

// HX711 모듈의 핀 정의
const int LOADCELL_DOUT_PIN = 3; // DT 핀
const int LOADCELL_SCK_PIN = 2;  // SCK 핀

// Tare 버튼 핀
const int TARE_BUTTON_PIN = 7;

HX711 scale;

// 고정된 보정값
const float CALIBRATION_FACTOR = -719.78350;

// 버튼 상태 변수
bool tareButtonPressed = false;

void setup() {
  Serial.begin(115200);
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);

  // 고정된 보정값 설정
  scale.set_scale(CALIBRATION_FACTOR); 
  
  scale.tare(); 
  // Tare 버튼 핀 설정
  pinMode(TARE_BUTTON_PIN, INPUT_PULLUP); // 풀업 저항 사용
}

void loop() {
  // Tare 버튼을 누르면 영점 조정
  if (digitalRead(TARE_BUTTON_PIN) == LOW) {
    if (!tareButtonPressed) { // 버튼이 처음 눌린 경우에만 동작
      Serial.println("Tare 버튼 눌림 - 영점 조정 수행 중...");
      scale.tare(); 
      Serial.println("영점 조정 완료");
      tareButtonPressed = true;
    }
  } else {
    tareButtonPressed = false;
  }
  
  // 무게 출력
  if (scale.is_ready()) {
    float weight = scale.get_units(5); // 평균 5회 측정
    Serial.print(weight, 2); // 소수점 두 자리까지 표시
    Serial.println(" g");
  } 
  delay(100);
}
