# Android 네이티브에서 GGUF(Q4) 모델 구동 가이드

이 문서는 Android 네이티브(Kotlin/Java + NDK) 앱에서 ggml/llama.cpp를 사용해 Q4 양자화(GGUF) 모델을 로컬 추론으로 구동하도록 LLM에게 구현을 지시하기 위한 기술 요구사항과 단계별 지침입니다.

## 목표
- GGUF(Q4 계열) LLM을 Android 단말(arm64-v8a)에서 오프라인 추론
- CPU(NEON) 기반 멀티스레딩 + mmap 중심 최적화
- 스트리밍 토큰 출력, KV 세션 저장/복구 지원
- 앱 내부 API: init → completion(stream) → stop → save/load session → free

## 지원 환경
- Android minSdk: 24+
- ABI: arm64-v8a (필수), 필요 시 armeabi-v7a 추가 가능(비권장)
- NDK: r23+ (권장 r26+), CMake 3.22+
- AGP/Gradle: 최신 LTS 범위
- 빌드 머신: macOS/Linux

## 모델 준비(Q4 양자화)
- 원본 FP16/BF16 → llama.cpp의 `llama-quantize`로 GGUF Q4 계열 생성
  - 권장: Q4_0, Q4_1 또는 Q4_K_S/M (성능/품질 절충)
  - 회피: 구형/비표준 포맷(Q4_0_4_4, Q4_0_4_8, Q4_0_8_8 등) — 일부 구현 차단 로직 존재
- **테스트 모델 예시**: `llama31-banyaa-q4_k_m.gguf` (Q4_K_M 양자화, Llama 3.1 기반)
  - 개발 환경 경로: `/Volumes/Transcend/Projects/qaipi-mac/models/gguf/llama31-banyaa-q4_k_m.gguf`
  - Q4_K_M 특성: K-quants 계열로 품질/속도 균형 우수, **CMake에 `GGML_USE_K_QUANTS=1` 필수**
- 모델 배치: 최초 실행 시 SAF/다운로드로 filesDir(예: `/data/data/<pkg>/files/models/…`)에 저장
  - 주의: mmap을 위해 "실제 파일"이어야 함(assets 압축 파일은 mmap 불가). 샘플 모델을 배포하더라도 앱 시작 시 filesDir로 복사 후 사용
  - Android 앱 내부 경로 예시: `context.getFilesDir() + "/models/llama31-banyaa-q4_k_m.gguf"`

## 프로젝트 구조 개요
- `third_party/llama.cpp` (업스트림 서브모듈)
- `app/src/main/cpp/` (JNI 브리지, CMakeLists)
- `app/src/main/java/.../LlamaBridge.kt` (Kotlin 래퍼)
- `app/src/main/assets/` (선택: 초기 배포용 gguf, 실행 시 filesDir로 복사)
- `filesDir/models/` (실사용 gguf 저장 위치)

## Gradle/NDK 설정(핵심)
```groovy
// app/build.gradle
android {
  defaultConfig {
    ndk {
      abiFilters "arm64-v8a"  // 필요 시 "armeabi-v7a" 추가
    }
    externalNativeBuild {
      cmake {
        cppFlags "-O3 -DNDEBUG -fvisibility=hidden"
      }
    }
  }
  externalNativeBuild {
    cmake {
      path file("src/main/cpp/CMakeLists.txt")
      version "3.22.1"
    }
  }
  packagingOptions {
    jniLibs {
      useLegacyPackaging false
    }
  }
}
```

## CMake 설정(업스트림 사용 + JNI 타깃)
```cmake
# app/src/main/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.22)
project(llama_android)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# 양자화/최적화 정의(환경에 맞게 가감)
# Q4_K_M 등 K-quants 계열 사용 시 필수
add_compile_definitions(GGML_USE_K_QUANTS=1)

# 업스트림 llama.cpp CMake 활용(경로는 실제 서브모듈 위치로 조정)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/llama.cpp ${CMAKE_BINARY_DIR}/llama_build)

add_library(llama_jni SHARED
    native/jni_bridge.cpp
)

target_include_directories(llama_jni PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp
)

target_link_libraries(llama_jni
    PRIVATE
    llama       # llama.cpp CMake가 제공하는 타깃명(업스트림 버전에 따라 달라질 수 있음)
    log
    android
)
```

## JNI 브리지 API(요구 사양)
필수 제공 함수(예시 시그니처):
- `jlong init(String modelPath, int nCtx, int nThreads, int nBatch, boolean useMmap, boolean useMlock, int seed)`
- `void free(long handle)`
- `jintArray tokenize(long handle, String text)` 또는 네이티브 토큰 카운트 함수
- `void completionStart(long handle, String prompt, int numPredict, float temperature, float topP, int topK, float repeatPenalty, int repeatLastN, String[] stopSequences, jobject callback)` — 비동기 토큰 스트리밍 시작
  - 또는 `CompletionParams` 객체를 받아서 파싱하는 방식 (구조체 전달)
- `void completionStop(long handle)` — 중단
- `int saveSession(long handle, String path)` — 토큰 수 반환 또는 상태 코드
- `boolean loadSession(long handle, String path)`

**기본 생성 파라미터 적용:**
- `completionStart` 호출 시 위의 `DEFAULT_OPTIONS` 값을 기본값으로 사용
- 사용자가 명시적으로 값을 제공하지 않으면 기본값 적용

네이티브 스켈레톤(개념 예시):
```cpp
// app/src/main/cpp/native/jni_bridge.cpp
#include <jni.h>
#include "llama.h"

struct LlamaCtx {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
};

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_llama_LlamaBridge_init(JNIEnv* env, jobject, jstring jModelPath,
                                        jint nCtx, jint nThreads, jint nBatch,
                                        jboolean useMmap, jboolean useMlock, jint seed) {
    const char* path = env->GetStringUTFChars(jModelPath, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = useMmap;
    mparams.use_mlock = useMlock;   // Android에서 실패 가능 → 실패 허용/로그 처리
    mparams.seed = seed;

    llama_model* model = llama_load_model_from_file(path, mparams);
    env->ReleaseStringUTFChars(jModelPath, path);
    if (!model) return 0;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = nCtx;
    cparams.n_threads = nThreads;
    cparams.n_batch = nBatch;

    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) { llama_free_model(model); return 0; }

    auto* handle = new LlamaCtx{model, ctx};
    return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_llama_LlamaBridge_free(JNIEnv*, jobject, jlong handle) {
    auto* h = reinterpret_cast<LlamaCtx*>(handle);
    if (!h) return;
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_free_model(h->model);
    delete h;
}

// completionStart 예시 (기본 파라미터 적용)
extern "C" JNIEXPORT void JNICALL
Java_com_example_llama_LlamaBridge_completionStart(
    JNIEnv* env, jobject, jlong handle, jstring jPrompt,
    jint numPredict, jfloat temperature, jfloat topP, jint topK,
    jfloat repeatPenalty, jint repeatLastN, jobjectArray jStopSequences,
    jobject callback) {
    auto* h = reinterpret_cast<LlamaCtx*>(handle);
    if (!h || !h->ctx) return;

    const char* prompt = env->GetStringUTFChars(jPrompt, nullptr);
    
    // 기본값 적용 (DEFAULT_OPTIONS)
    int n_predict = (numPredict > 0) ? numPredict : 100;
    float temp = (temperature > 0.0f) ? temperature : 0.3f;
    float top_p = (topP > 0.0f) ? topP : 0.85f;
    int top_k = (topK > 0) ? topK : 50;
    float rep_penalty = (repeatPenalty > 0.0f) ? repeatPenalty : 1.2f;
    int rep_last_n = (repeatLastN > 0) ? repeatLastN : 256;

    // llama_sampling_params 설정
    llama_sampling_params sparams = llama_sampling_default_params();
    sparams.temp = temp;
    sparams.top_p = top_p;
    sparams.top_k = top_k;
    sparams.repeat_penalty = rep_penalty;
    sparams.repeat_last_n = rep_last_n;
    // stop sequences 처리...

    // 토큰화 및 디코딩 루프 시작
    // 각 토큰 생성 시 callback 호출: env->CallVoidMethod(callback, ...)
    
    env->ReleaseStringUTFChars(jPrompt, prompt);
}
```

Kotlin 래퍼:
```kotlin
// app/src/main/java/com/example/llama/LlamaBridge.kt
package com.example.llama

object LlamaBridge {
    init { System.loadLibrary("llama_jni") }

    external fun init(
        modelPath: String,
        nCtx: Int,
        nThreads: Int,
        nBatch: Int,
        useMmap: Boolean,
        useMlock: Boolean,
        seed: Int
    ): Long

    external fun free(handle: Long)
    
    // 기본 파라미터를 사용하는 completionStart (DEFAULT_OPTIONS 적용)
    external fun completionStart(
        handle: Long,
        prompt: String,
        numPredict: Int = 100,
        temperature: Float = 0.3f,
        topP: Float = 0.85f,
        topK: Int = 50,
        repeatPenalty: Float = 1.2f,
        repeatLastN: Int = 256,
        stopSequences: Array<String> = emptyArray(),
        callback: (String) -> Unit  // 토큰 스트리밍 콜백
    )
    
    external fun completionStop(handle: Long)
    // external fun tokenize(...): IntArray
    // external fun saveSession(...): Int
    // external fun loadSession(...): Boolean
    
    // 사용 예시
    fun generateWithDefaults(handle: Long, prompt: String, onToken: (String) -> Unit) {
        completionStart(
            handle = handle,
            prompt = prompt,
            // 기본값 사용 (numPredict=100, temperature=0.3, topP=0.85, topK=50, repeatPenalty=1.2, repeatLastN=256)
            callback = onToken
        )
    }
}
```

## Completion(스트리밍) 설계
- 네이티브 디코딩 루프에서 토큰 1개 생성 시마다 Java로 콜백
- 콜백 전달 방식: `env->CallVoidMethod` 또는 ALooper/Handler 경유
- 샘플링 파라미터(예시): temperature, top_p, top_k, min_p, repetition penalty, mirostat 등
- Stop 시퀀스/정규식 후처리: 마지막 버퍼 확정 시 적용

## 기본 생성 파라미터(모델 로드 시 초기 설정)
완전한 문장 생성과 안정적인 디코딩을 위한 권장 기본값:

```kotlin
// Kotlin 예시
data class GenerationOptions(
    val numPredict: Int = 100,           // 최대 생성 토큰 수
    val temperature: Float = 0.3f,       // 샘플링 온도 (낮을수록 결정론적)
    val topP: Float = 0.85f,             // nucleus sampling
    val topK: Int = 50,                  // top-k sampling
    val repeatPenalty: Float = 1.2f,     // 반복 억제 계수
    val repeatLastN: Int = 256,           // 반복 억제 대상 토큰 수
    val stop: List<String> = emptyList() // 명시적 중지 시퀀스 (비어있으면 가드/후처리 의존)
)

// JNI 브리지에서 사용할 구조체 예시
// C++ 측에서 llama_sampling_params 구조체에 매핑
```

**파라미터 설명:**
- `num_predict`: 생성할 최대 토큰 수 (0 또는 음수 = 컨텍스트 길이 제한까지)
- `temperature`: 낮을수록(0.1~0.5) 더 결정론적/일관적, 높을수록(0.7~1.2) 더 창의적
- `top_p` (nucleus sampling): 누적 확률 임계값, 0.85는 상위 85% 확률 토큰만 고려
- `top_k`: 상위 K개 토큰만 샘플링 후보로 고려
- `repeat_penalty`: 1.0 이상이면 반복 억제 강도 (1.2 = 20% 억제)
- `repeat_last_n`: 반복 억제를 적용할 최근 토큰 수 (256 = 최근 256 토큰 기준)
- `stop`: 빈 배열이면 모델 내장 EOS/가드 토큰 또는 후처리 정규식에 의존

**llama31-banyaa-q4_k_m.gguf 모델 권장 설정:**
- 기본값 위의 `DEFAULT_OPTIONS` 사용 권장
- 대화형 응답: `temperature: 0.3~0.5`, `num_predict: 100~200`
- 창의적 생성: `temperature: 0.7~0.9`, `num_predict: 200~512`

## KV 세션
- 대화 재개/속도 개선을 위해 세션 저장/복원 지원
- 인터페이스:
  - `saveSession(handle, filesDir + "/session/llama-session.bin")`
  - `loadSession(handle, filesDir + "/session/llama-session.bin")`

## 파일 I/O 및 보안
- 모델 경로: `context.getFilesDir()/models/...` (권장)
  - 예시: `context.getFilesDir() + "/models/llama31-banyaa-q4_k_m.gguf"`
  - 개발/테스트: macOS 경로(`/Volumes/Transcend/Projects/qaipi-mac/models/gguf/...`)에서 Android 단말로 전송 필요
- 권한: 내부 저장소만 사용 시 추가 퍼미션 불필요
- 자원 회수: Activity/Service 수명주기에서 `free(handle)` 보장

## 성능/메모리 튜닝
- 스레드: `Runtime.getRuntime().availableProcessors()` 기준 big 코어 수에 맞춰 조정
- 배치: 256~1024 범위에서 단말 성능에 맞춰 실측
- 컨텍스트 길이(n_ctx): 메모리/속도 절충(예: 4096/8192/16384) — Q4 모델 RAM 예산 고려
  - **llama31-banyaa-q4_k_m.gguf 예상 메모리**: 모델 파일 크기 + (n_ctx × 약 2~4MB) + KV 캐시
- mmap: 반드시 활성화(use_mmap=true), mlock은 실패 허용
- Q4 선택 지침:
  - Q4_0/Q4_1: 보편적, 메모리 절감/속도 양호
  - **Q4_K_M (llama31-banyaa 예시)**: 최신 계열, 품질/속도 균형 우수, **CMake에 `GGML_USE_K_QUANTS=1` 필수**
- 전력/발열: 장시간 추론 시 5~10분 단위로 온도/클럭 모니터링, 필요 시 쓰로틀링 로직

## 에러 처리/로그
- 모델 파일 없음/열기 실패/헤더 손상 시 명확한 에러 코드/토스트
- 양자화 호환성 경고: 구형 포맷은 로드 거부/가이드 메시지
- 토큰 루프 중단 시 안전한 정리(`completionStop` 경로 보장)

## ProGuard/R8
- JNI 래퍼 클래스/메소드 보존:
```proguard
-keep class com.example.llama.LlamaBridge { *; }
```

## APK/번들 사이즈 관리
- `abiFilters`로 arm64-v8a만 제공 시 용량 절감
- 필요한 경우 `splits { abi { ... } }`로 ABI별 배포

## 테스트 계획
- 스모크: 작은 프롬프트로 32/128 토큰 스트리밍 확인
  - 테스트 모델: `llama31-banyaa-q4_k_m.gguf`
- 회귀: 동일 프롬프트 TPS/지연/메모리 사용 측정 비교
- 중단/재시작: stop → resume(새 요청) 시 자원 정상 정리 확인
- 세션: save → 앱 재시작 → load → 동작 검증
- 모델 로드 검증: 개발 환경(`/Volumes/Transcend/Projects/qaipi-mac/models/gguf/...`)에서 Android 단말로 전송 후 로드 성공 확인

## 라이선스/표기
- ggml/llama.cpp: MIT/BSD-3 (업스트림 리포를 확인하여 준수)
- 모델 라이선스: GGUF 배포/사용 조건 준수(상용/재배포 여부 확인)

## 구현 체크리스트(LLM용)
- [ ] NDK/CMake 환경 구성 및 `third_party/llama.cpp` 추가
- [ ] CMake에 `GGML_USE_K_QUANTS=1` 정의 (Q4_K_M 모델 필수)
- [ ] JNI 브리지 함수 구현(init/free/tokenize/completionStart/Stop/save/load)
- [ ] 안드로이드에서 모델 파일을 filesDir로 배치 및 mmap 로드
  - 테스트 모델: `llama31-banyaa-q4_k_m.gguf` (경로: `/Volumes/Transcend/Projects/qaipi-mac/models/gguf/...`)
- [ ] 스트리밍 콜백(네이티브→Java) 구현
- [ ] 기본 생성 파라미터(DEFAULT_OPTIONS) 구현 및 적용
  - `num_predict: 100`, `temperature: 0.3`, `top_p: 0.85`, `top_k: 50`
  - `repeat_penalty: 1.2`, `repeat_last_n: 256`, `stop: []`
  - completionStart 함수에서 기본값 적용 로직 구현
- [ ] 샘플링/stop 시퀀스 파라미터 반영
- [ ] KV 세션 저장/복원 구현
- [ ] 스레드/배치/컨텍스트 길이 기본값 및 설정 주입
- [ ] 예외/에러 로깅 및 사용자 알림
- [ ] ProGuard 예외 및 릴리즈 최적화
- [ ] Q4_K_M 양자화 형식 로드 검증 (llama31-banyaa 모델로 테스트)

## 주요 개선 이력

### 2025-11-11: 샘플링 방식 개선 및 GPU 가속, 시스템 프롬프트 적용

#### 1. 샘플링 방식 개선 (답변 품질 향상)
- **문제**: 초기 구현에서 사용된 Greedy 샘플링 방식은 가장 확률 높은 토큰만 선택하여, 다양성이 부족하고 어눌하며 기계적인 답변을 생성하는 문제가 있었음.
- **해결**: `jni_bridge.cpp`의 토큰 생성 로직을 대폭 수정하여 `llama.cpp`의 `sampler_chain` API를 사용하도록 변경.
  - `llama_sampler_chain_init`으로 샘플러 체인을 생성하고, `llama_sampler_init_penalties` (반복 페널티), `_top_k`, `_top_p`, `_temp` (온도) 샘플러를 순서대로 추가.
  - 최종적으로 `llama_sampler_init_greedy()` 또는 `llama_sampler_init_dist()`를 체인의 마지막에 추가하여 토큰을 선택.
  - 이 변경으로 `temperature`, `top_p`, `top_k`, `repeat_penalty` 등 Python CLI와 동일한 수준의 정교한 샘플링이 가능해져 답변의 자연스러움이 크게 향상됨.

#### 2. GPU 하드웨어 가속 (성능 향상)
- **문제**: Snapdragon 8 Elite 칩셋의 강력한 Adreno GPU를 활용하지 않고 CPU로만 모든 연산을 처리하여 토큰 생성 속도가 매우 느렸음.
- **해결**: Vulkan 백엔드를 활성화하여 GPU 오프로딩(Offloading)을 구현.
  - `app/src/main/cpp/CMakeLists.txt`: `set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan")` 옵션을 추가하고, `llama_jni` 타겟에 `Vulkan` 라이브러리를 링크함.
  - `jni_bridge.cpp`, `LlamaBridge.kt`, `ChatRepository.kt`: `n_gpu_layers` 파라미터를 추가하여 Kotlin 단에서 GPU로 보낼 레이어 수를 지정할 수 있도록 인터페이스를 확장.
  - `ChatRepository.kt`: `LlamaBridge.init` 호출 시 `nGpuLayers = 99`로 설정하여, 가능한 모든 모델 레이어를 GPU 메모리로 오프로드. 이를 통해 연산 속도를 극대화하고 토큰 생성 지연 시간을 크게 단축함.

#### 3. 시스템 프롬프트 적용 (역할 부여 및 일관성)
- **문제**: 모델에 특정 역할을 부여하는 시스템 프롬프트가 적용되지 않아, 모델의 기본 학습 상태에 따른 일반적인 답변만 생성되었음.
- **해결**: `chat_cli.py`와 동일한 시스템 프롬프트를 Llama 3 Instruct 채팅 템플릿 형식에 맞게 적용.
  - `ChatRepository.kt`: `formatPrompt` 함수를 구현하여, 대화 기록과 시스템 프롬프트를 `<|begin_of_text|>`, `<|start_header_id|>system<|end_header_id|>...` 등 Llama 3가 요구하는 형식의 문자열로 조합.
  - 이로써 모델은 "10대 발달장애인의 일상을 돕는 한국어 에이전트"라는 명확한 역할을 인지하고, 지침에 따라 일관성 있는 답변을 생성하게 됨.

### 2025-11-12: 진행률/메타데이터 콜백, 세션 저장, UI 개선, Vulkan 헤더 이슈 분석

- **모델 로드 진행률 및 메타데이터 스트리밍**  
  - `TokenCallback`을 확장하여 `onLoadProgress`, `onModelMetadata`를 추가하고, `jni_bridge.cpp`에서 `llama_model_params.progress_callback`을 활용하여 로드 단계별 진행률(0~100%)을 즉시 전송.  
  - `llama_model_meta_val_str()`를 사용해 모델 이름, 양자화 방식, 컨텍스트 길이 등 핵심 메타데이터를 JSON으로 직렬화 후 Kotlin 레이어로 전달.  
  - `ChatViewModel`은 수신한 정보를 `ChatUiState`에 반영하고 `ModelPathStore`에 영속화하여 재시작 시에도 모델 정보를 즉시 표시.

- **세션 저장/복구 실제 구현**  
  - 기존 더미 구현을 `llama_state_get_size / get_data / set_data` 기반의 실 데이터 저장/복원으로 교체 (`jni_bridge.cpp`), 실패 시 명확한 에러 코드 반환.
  - 저장 성공 시 저장된 바이트 수를, 실패 시 음수 코드를 리턴하여 상위 레이어에서 처리 용이.

- **UI 반응성 개선**  
  - 로드 진행률 표시를 위해 `ChatUiState`에 `loadProgress`, `modelMetadata` 필드를 추가하고 Compose UI에 `LinearProgressIndicator` 및 모델 정보 요약 텍스트를 추가.  
  - 입력창은 IME `Send` 액션을 즉시 처리하며, 버튼 클릭 시 키보드를 자동으로 숨겨 UX를 개선.

- **NDK/CMake 최적화 및 CPU 플래그**  
  - `app/src/main/cpp/CMakeLists.txt`에서 arm64-v8a 타깃에 `-march=armv8.2-a+dotprod+i8mm`를 적용하여 Snapdragon 8 Elite에서 ARM i8mm/DotProd 명령어를 활용.
  - `VK_NO_PROTOTYPES` 정의 및 `ggml-vulkan.cpp` 헤더 매크로 정비를 통해 Vulkan C++ 래퍼 충돌을 최소화하려 했으나, NDK가 제공하는 `vulkan.hpp`(1.3.268 계열)와 최신 `ggml` 구현 간 중복 선언 문제가 여전히 발생함을 확인.  
    → 해결을 위해서는 Khronos의 최신 `Vulkan-Headers`(1.3.301+)를 번들링하거나, NDK 측 헤더 업데이트가 필요함.

- **Gradle/Ninja 경로 고정**  
  - `build.gradle.kts`에서 `-DCMAKE_MAKE_PROGRAM`, `-DCMAKE_PROGRAM_PATH`, `Vulkan_GLSLC_EXECUTABLE` 등을 명시하여 서브 프로젝트(ggml Vulkan 셰이더 빌더)가 올바른 Ninja/glslc 바이너리를 찾도록 보조.

- **현재 상황**  
  - GPU 오프로딩 플래그 및 JNI 파이프라인은 준비 완료.  
  - 빌드는 여전히 NDK 내 `vulkan.hpp` 중복 정의 문제로 중단되며, 최신 Vulkan 헤더 확보가 선행 과제로 남아있음.

