# LlamaChat Android (Native GGUF + Chat UI)

오프라인 GGUF(Q4) 모델 추론을 위한 안드로이드 네이티브 앱 스켈레톤과 Jetpack Compose 채팅 UI를 제공합니다.

## 요구사항
- Android Studio 최신 버전
- Android Gradle Plugin 8.6.x, Kotlin 2.0.x
- NDK r26+ 권장, CMake 3.22+
- minSdk 24, targetSdk 35
- ABI: arm64-v8a

## 주요 기능
- 채팅 UI(메시지 리스트, 입력, 전송)
- 네이티브(JNI) 브리지 스켈레톤
  - init → completionStart(stream) → completionStop → free
  - 디폴트 샘플링 파라미터 적용
- 스텁 모드 기본 제공
  - `third_party/llama.cpp`가 없거나 네이티브 초기화 실패 시, UI를 테스트할 수 있는 토큰 스트림을 가짜로 출력

## 디렉터리 구조
- `app/src/main/cpp/` JNI 및 CMake 설정
- `third_party/llama.cpp/` 업스트림 서브모듈 위치
- `app/src/main/java/com/example/llama/*` UI, ViewModel, Repository
- `app/src/main/java/com/example/llama/nativebridge/*` JNI 래퍼/콜백 인터페이스

## llama.cpp 연동
서브모듈 추가:
```bash
git submodule add https://github.com/ggerganov/llama.cpp third_party/llama.cpp
git submodule update --init --recursive
```

## 모델 배치
- 권장 모델: `llama31-banyaa-q4_k_m.gguf`
- 배치 방법:
  - Android Studio의 Device File Explorer로 `/data/data/com.example.llama/files/models/`에 복사, 또는
  - `app/src/main/assets/`에 `llama31-banyaa-q4_k_m.gguf`를 넣으면 최초 실행 시 `filesDir/models`로 복사 시도
  - 세션(옵션): `/data/data/com.example.llama/files/session/llama-session.bin` 같은 내부경로 사용 권장

## 빌드
1. Android Studio에서 열기
2. NDK/CMake 설치 확인
3. 실행
   - `third_party/llama.cpp`가 없으면 스텁 모드로 빌드되어 UI 테스트 가능
   - 서브모듈 추가 후 재빌드하면 네이티브 연동 활성화

## 참고 옵션
- `app/src/main/cpp/CMakeLists.txt`에서 `GGML_USE_K_QUANTS=1` 정의 포함
- `app/build.gradle.kts`에 `abiFilters "arm64-v8a"`
- ProGuard: `proguard-rules.pro`에서 브리지 보존
 - Stop 시퀀스: `completionStart`의 `stopSequences`로 전달하면 JNI에서 후처리 검출

## 주의
- 기본 llama.cpp 디코딩 루프 및 스트리밍 콜백은 구현되어 있습니다. 단, KV 세션 저장/복원은 현재 기본 스텁 동작으로 제공됩니다.
- 장치 성능에 따라 `nCtx`, `nThreads`, `nBatch`를 조정하세요.


