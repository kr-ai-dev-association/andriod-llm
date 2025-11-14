# Llama 3.1 구현 상세 설명

이 문서는 BanyaLLM 프로젝트에서 구현된 샘플링 방법, 토큰 생성 방법, 특수 토큰 처리 방법에 대해 자세히 설명합니다.

## 1. 샘플링 방법 (Sampling)

샘플링은 `LlamaContext.swift`의 `initialize()` 메서드에서 **체인(Chain) 방식**으로 구성됩니다.

### 샘플링 파이프라인 구성

```swift
// Sampling 초기화 (Llama 3.1 최적화)
let sparams = llama_sampler_chain_default_params()
self.sampling = llama_sampler_chain_init(sparams)

// 1. Top-K 샘플링 (0 = 비활성화, Llama 3.1 권장)
llama_sampler_chain_add(self.sampling, llama_sampler_init_top_k(0))

// 2. Top-P (Nucleus Sampling) - 0.9
llama_sampler_chain_add(self.sampling, llama_sampler_init_top_p(0.9, 1))

// 3. Min-P - 낮은 확률 토큰 배제 (Llama 3.1 핵심 설정)
llama_sampler_chain_add(self.sampling, llama_sampler_init_min_p(0.05, 1))

// 4. Temperature - 창의성 조절 (0.6 = 더 결정론적, 반복 감소)
llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.6))

// 5. Repeat Penalty - 반복 방지 강화
llama_sampler_chain_add(self.sampling, llama_sampler_init_penalties(
    64,     // last_n: 최근 64 토큰만 고려
    1.15,   // repeat_penalty: 강한 반복 패널티
    0.1,    // freq_penalty: 빈도 패널티
    0.1     // presence_penalty: 존재 패널티
))

// 6. Dist 샘플링 (최종 토큰 선택)
llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(UInt32.random(in: 0...1000)))
```

### 샘플링 파이프라인 순서

1. **Top-K 샘플링** (비활성화)
   - K=0으로 설정되어 비활성화됨
   - Llama 3.1에서는 Top-P와 Min-P 조합을 권장

2. **Top-P (Nucleus Sampling)**
   - 누적 확률 0.9까지의 토큰만 고려
   - 확률 분포의 상위 90%만 유지하여 품질 향상

3. **Min-P**
   - 최소 확률 0.05 미만의 토큰을 배제
   - Llama 3.1의 핵심 설정으로 낮은 확률 토큰 제거

4. **Temperature**
   - 값: 0.6 (낮을수록 더 결정론적)
   - 창의성과 일관성의 균형 조절
   - 반복 감소 효과

5. **Repeat Penalty**
   - `last_n=64`: 최근 64개 토큰만 고려하여 반복 감지 정확도 향상
   - `repeat_penalty=1.15`: 반복 토큰의 확률을 1.15배 감소
   - `freq_penalty=0.1`: 빈도 기반 패널티 추가
   - `presence_penalty=0.1`: 이미 나온 단어에 대한 존재 패널티

6. **Dist 샘플링**
   - 최종 토큰 선택 단계
   - 랜덤 시드로 다양성 확보

## 2. 토큰 생성 방법 (Token Generation)

토큰 생성은 `completionLoop()` 메서드에서 수행됩니다.

### 토큰 생성 프로세스

```swift
func completionLoop() throws -> String {
    // 1. 샘플링: 다음 토큰 ID 선택
    let lastTokenIdx = max(0, lastBatchSize > 0 ? lastBatchSize - 1 : max(0, n_cur - 1))
    let new_token_id = llama_sampler_sample(sampling, context, lastTokenIdx)
    
    // 2. 종료 조건 확인
    let isEOG = (new_token_id == 128001 || new_token_id == 128008 || new_token_id == 128009)
    if isEOG || n_cur == n_len {
        isDone = true
        return ""
    }
    
    // 3. 특수 토큰 처리 (ID 레벨)
    if new_token_id >= 128000 {
        // 배치에 추가하지만 출력하지 않음
        llama_batch_add(&batch, new_token_id, n_cur, [0], true)
        llama_decode(context, batch)
        return ""
    }
    
    // 4. 디토큰화: 토큰 ID → UTF-8 바이트
    let new_token_cchars = token_to_piece(token: new_token_id)
    temporary_invalid_cchars.append(contentsOf: new_token_cchars)
    
    // 5. UTF-8 문자열 변환 (부분 바이트 처리)
    var new_token_str: String
    if let string = String(validatingUTF8: temporary_invalid_cchars + [0]) {
        temporary_invalid_cchars.removeAll()
        new_token_str = string
    } else {
        // 부분 바이트 처리 로직
        new_token_str = ""
    }
    
    // 6. 특수 토큰 문자열 필터링
    // ... (아래 특수 토큰 처리 섹션 참조)
    
    // 7. 배치에 추가 및 디코딩
    llama_batch_clear(&batch)
    llama_batch_add(&batch, new_token_id, n_cur, [0], true)
    llama_decode(context, batch)
    
    return new_token_str
}
```

### 상세 단계 설명

1. **샘플링 단계**
   - `llama_sampler_sample()` 함수로 다음 토큰 ID 선택
   - 이전 디코딩의 마지막 토큰 인덱스를 사용

2. **종료 조건 확인**
   - **EOG 토큰 감지**: 128001 (`<|end_of_text|>`), 128008 (`<|eom_id|>`), 128009 (`<|eot_id|>`)
   - **최대 토큰 수 도달**: `n_len=256` (최대 생성 토큰 수)

3. **특수 토큰 처리 (ID 레벨)**
   - 토큰 ID가 128000 이상이면 특수 토큰으로 간주
   - 배치에 추가하여 디코딩은 수행하지만 출력하지 않음

4. **디토큰화 (Token to Piece)**
   - `token_to_piece()` 함수로 토큰 ID를 UTF-8 바이트 배열로 변환
   - 부분 바이트는 `temporary_invalid_cchars`에 누적

5. **UTF-8 문자열 변환**
   - 완전한 UTF-8 문자를 구성할 때까지 대기
   - `String(validatingUTF8:)`로 안전하게 변환
   - 부분 바이트가 있으면 다음 토큰과 결합하여 처리

6. **디코딩 (Forward Pass)**
   - `llama_decode()` 함수로 다음 토큰 생성을 위한 forward pass 수행
   - 배치에 새 토큰을 추가하고 KV cache 업데이트

## 3. 특수 토큰 처리 방법 (Special Token Handling)

특수 토큰 처리는 **두 단계**로 수행됩니다:

### 3.1 토큰 ID 레벨 필터링

```swift
// 특수 토큰 필터링 (Llama 3.1 특수 토큰은 출력하지 않음)
// 128000-128255: 모든 특수 토큰 범위
if new_token_id >= 128000 {
    // 특수 토큰은 배치에 추가하지만 출력하지 않음
    llama_batch_clear(&batch)
    llama_batch_add(&batch, new_token_id, n_cur, [0], true)
    
    n_decode += 1
    n_cur += 1
    
    if llama_decode(context, batch) != 0 {
        throw LlamaError.batchSizeExceeded
    }
    
    lastBatchSize = 1
    return "" // 빈 문자열 반환 (특수 토큰은 출력 안 함)
}
```

**특징:**
- 토큰 ID가 128000 이상이면 특수 토큰으로 간주
- 배치에 추가하여 디코딩은 수행하지만 출력하지 않음
- 모델의 내부 상태는 유지하되 사용자에게는 보이지 않음

### 3.2 문자열 레벨 필터링

일반 토큰으로 특수 토큰 문자열이 생성될 수 있으므로, 문자열 레벨에서도 필터링을 수행합니다.

#### 완전한 특수 토큰 패턴 제거

```swift
let specialTokenPatterns = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|python_tag|>",
    "<|finetune_right_pad_id|>"
]

for pattern in specialTokenPatterns {
    new_token_str = new_token_str.replacingOccurrences(of: pattern, with: "")
}
```

#### 정규식 패턴 제거

```swift
// reserved_special_token 패턴 제거
if let regex = try? NSRegularExpression(pattern: "<\\|reserved_special_token_\\d+\\|>", options: []) {
    let range = NSRange(new_token_str.startIndex..., in: new_token_str)
    new_token_str = regex.stringByReplacingMatches(
        in: new_token_str,
        options: [],
        range: range,
        withTemplate: ""
    )
}
```

#### 부분 특수 토큰 패턴 필터링

토큰이 분해되어 생성되는 경우를 대비한 처리:

```swift
// 단독 파이프 제거
if new_token_str == "|" {
    new_token_str = ""
}

// '<|' 또는 '|>' 포함 시 제거
if new_token_str.contains("<|") || new_token_str.contains("|>") {
    new_token_str = new_token_str.replacingOccurrences(of: "<|", with: "")
    new_token_str = new_token_str.replacingOccurrences(of: "|>", with: "")
}

// 정규식으로 부분 패턴 제거
if let regex = try? NSRegularExpression(pattern: "<\\|.*?\\|>", options: []) {
    let range = NSRange(new_token_str.startIndex..., in: new_token_str)
    new_token_str = regex.stringByReplacingMatches(
        in: new_token_str,
        options: [],
        range: range,
        withTemplate: ""
    )
}
```

### 3.3 추가 필터링 (LlamaManager)

`LlamaManager.swift`에서는 누적된 텍스트에 대해 더 강화된 필터링을 수행합니다:

```swift
func filterSpecialTokens(_ text: String) -> String {
    var cleaned = text
    
    // 1. 완전한 특수 토큰 패턴 제거 (반복적으로 제거하여 중첩 패턴도 처리)
    var previousLength = 0
    var iterations = 0
    while cleaned.count != previousLength && iterations < 10 {
        previousLength = cleaned.count
        for pattern in specialTokenPatterns {
            cleaned = cleaned.replacingOccurrences(of: pattern, with: "")
        }
        iterations += 1
    }
    
    // 2. reserved_special_token 패턴 제거
    if let regex = try? NSRegularExpression(pattern: "<\\|reserved_special_token_\\d+\\|>", options: []) {
        let range = NSRange(cleaned.startIndex..., in: cleaned)
        cleaned = regex.stringByReplacingMatches(
            in: cleaned,
            options: [],
            range: range,
            withTemplate: ""
        )
    }
    
    // 3. 부분 특수 토큰 패턴 제거 (공격적 필터링)
    var foundPattern = true
    var patternIterations = 0
    while foundPattern && patternIterations < 10 {
        patternIterations += 1
        foundPattern = false
        
        // 방법 1: "<|" + "|>" 조합 찾기
        if let startRange = cleaned.range(of: "<|", options: .backwards),
           let endRange = cleaned.range(of: "|>", range: startRange.upperBound..<cleaned.endIndex) {
            cleaned = String(cleaned[..<startRange.lowerBound]) + String(cleaned[endRange.upperBound...])
            foundPattern = true
            continue
        }
        
        // 방법 2: 단독 파이프 제거
        if cleaned.contains("|") && !cleaned.contains("<|") && !cleaned.contains("|>") {
            cleaned = cleaned.replacingOccurrences(of: "|", with: "")
            foundPattern = true
        }
        
        // 방법 3: 정규식으로 부분 패턴 제거
        if let regex = try? NSRegularExpression(pattern: "<\\|[^|]*\\|>", options: []) {
            let range = NSRange(cleaned.startIndex..., in: cleaned)
            let newCleaned = regex.stringByReplacingMatches(
                in: cleaned,
                options: [],
                range: range,
                withTemplate: ""
            )
            if newCleaned != cleaned {
                cleaned = newCleaned
                foundPattern = true
            }
        }
        
        // 방법 4: 공백 + "<|" 또는 "|>" + 공백 패턴 제거
        cleaned = cleaned.replacingOccurrences(of: " <|", with: "")
        cleaned = cleaned.replacingOccurrences(of: "<| ", with: "")
        cleaned = cleaned.replacingOccurrences(of: " |>", with: "")
        cleaned = cleaned.replacingOccurrences(of: "|> ", with: "")
    }
    
    // 4. 이상한 패턴 제거 (<kts:1> 등)
    if let regex = try? NSRegularExpression(pattern: "<[^>]*>", options: []) {
        let range = NSRange(cleaned.startIndex..., in: cleaned)
        cleaned = regex.stringByReplacingMatches(
            in: cleaned,
            options: [],
            range: range,
            withTemplate: ""
        )
    }
    
    // 5. 특수 문자 조합 제거 (^^ 등 불필요한 이모지)
    cleaned = cleaned.replacingOccurrences(of: "^^", with: "")
    cleaned = cleaned.replacingOccurrences(of: "^^^", with: "")
    
    return cleaned
}
```

**이 함수의 특징:**
- **반복 제거**: 중첩된 특수 토큰 패턴도 처리
- **다양한 방법**: 여러 방법으로 부분 패턴 탐지 및 제거
- **무한 루프 방지**: 최대 10회 반복으로 제한
- **추가 패턴 제거**: HTML 태그 형식, 특수 문자 조합 등도 제거

## 요약

### 샘플링 방법
- **체인 방식**: Top-P + Min-P + Temperature + Repeat Penalty를 순차적으로 적용
- **Llama 3.1 최적화**: Top-K 비활성화, Min-P 활성화로 품질 향상
- **반복 방지**: Repeat Penalty로 최근 64 토큰을 고려하여 반복 감소

### 토큰 생성 방법
- **순차적 생성**: 샘플링 → EOG 확인 → 특수 토큰 필터링 → 디토큰화 → 디코딩
- **UTF-8 처리**: 부분 바이트를 누적하여 완전한 문자로 변환
- **배치 처리**: 효율적인 디코딩을 위한 배치 시스템

### 특수 토큰 처리 방법
- **이중 필터링**: 토큰 ID 레벨과 문자열 레벨에서 모두 필터링
- **완전한 제거**: 완전한 특수 토큰과 부분 패턴 모두 제거
- **강화된 필터링**: LlamaManager에서 추가 필터링으로 완벽한 제거 보장

이러한 구현을 통해 Llama 3.1 모델의 특수 토큰이 사용자에게 노출되지 않도록 처리하며, 자연스러운 텍스트 생성이 가능합니다.

---

# Android 구현 상세 설명

이 문서는 Android 버전에서 적용된 최적화 및 문제 해결 과정을 상세히 기록합니다.

## 1. Vulkan 백엔드 최적화 (2025-11-15)

### 1.1 문제 상황

초기 구현에서 Q4_K_M 모델을 사용했을 때 Adreno 830 GPU에서 다음과 같은 문제가 발생했습니다:

```
I AdrenoVK-0: Failed to link shaders.
I AdrenoVK-0: Pipeline create failed
```

이는 Q4_K quantization의 복잡한 쉐이더 구조가 Adreno 830의 Vulkan 드라이버와 호환되지 않아 발생한 문제였습니다.

### 1.2 해결 방법

#### 모델 변경: Q4_K → Q4_0

Q4_K 모델을 Q4_0으로 변환하여 쉐이더 호환성 문제를 해결했습니다:

```bash
# Q4_K → F16 → Q4_0 변환
./llama-quantize llama31-banyaa-q4_k_m.gguf llama31-banyaa-f16.gguf F16
./llama-quantize --allow-requantize llama31-banyaa-f16.gguf llama31-banyaa-q4_0.gguf Q4_0
```

**변경 이유:**
- Q4_0은 더 단순한 커널 구조를 가져 쉐이더 컴파일/링크가 안정적
- Adreno 830의 Vulkan 드라이버와 호환성 향상
- 메모리 사용량은 약간 증가하지만 안정성 확보

#### Vulkan 메모리 할당 최적화

**파일:** `app/src/main/cpp/native/jni_bridge.cpp`

```cpp
llama_model_params mparams = llama_model_default_params();
// Optimized Vulkan settings for Adreno 830 stability
if (nGpuLayers == -1) {
    mparams.n_gpu_layers = 5;  // Default to 5 layers for stability
} else {
    mparams.n_gpu_layers = nGpuLayers;
}
mparams.use_mmap = useMmap;
mparams.use_mlock = useMlock;
mparams.use_extra_bufts = false;  // Q4_0 doesn't require extra buffers
// Use DEVICE_LOCAL memory (no_host=true) for better GPU performance and stability
// This ensures model weights are stored in GPU memory, reducing host-device transfers
mparams.no_host = true;  // DEVICE_LOCAL memory for GPU weights
```

**주요 설정:**
- `no_host = true`: DEVICE_LOCAL 메모리 사용으로 GPU 메모리에 가중치 직접 저장
- `use_extra_bufts = false`: Q4_0은 추가 버퍼 불필요
- `n_gpu_layers = 5`: GPU 레이어 최소화로 쉐이더 연산 감소

#### 배치 크기 최적화

**파일:** `app/src/main/cpp/native/jni_bridge.cpp`

```cpp
llama_context_params cparams = llama_context_default_params();
cparams.n_ctx = nCtx;
cparams.n_threads = nThreads;
cparams.n_threads_batch = nThreads;
cparams.n_batch = nBatch;  // 32로 감소
// Reduced micro-batch size to minimize memory pressure and Vulkan operations
// Lower n_ubatch reduces concurrent GPU operations, improving stability on Adreno 830
cparams.n_ubatch = 2;  // Reduced from 4 to 2 for stability
cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // Android에서 비활성화
```

**파일:** `app/src/main/java/com/example/llama/data/ChatRepository.kt`

```kotlin
handle = LlamaBridge.init(
    modelPath = modelPath,
    nCtx = 768,
    nThreads = 8,
    nBatch = 32,  // Reduced from 64 to 32 to reduce memory pressure
    nGpuLayers = 5,  // Reduced to 5 layers to minimize Vulkan shader operations
    useMmap = true,
    useMlock = false,
    seed = 0,
    callback = callback
)
```

**최적화 이유:**
- `n_batch = 32`: 메모리 압력 감소 (64 → 32)
- `n_ubatch = 2`: 동시 GPU 연산 최소화 (4 → 2)
- `n_gpu_layers = 5`: Vulkan 쉐이더 연산 최소화로 드라이버 크래시 방지

### 1.3 최적화 결과

최적화 후 로그에서 확인된 결과:

```
I AdrenoVK-0: Application Name    : ggml-vulkan
I AdrenoVK-0: Api Version         : 0x00403000
D BanyaChatLlama: ggml_vulkan: Found 1 Vulkan devices:
D BanyaChatLlama: ggml_vulkan: 0 = Adreno (TM) 830 ... | uma: 1 | fp16: 1
I BanyaChatLlama: llama_model_load_from_file_impl: using device Vulkan0 (Adreno (TM) 830) - 15210 MiB free
I BanyaChatLlama: load_tensors: offloading 5 repeating layers to GPU
I BanyaChatLlama: load_tensors: offloaded 5/33 layers to GPU
I BanyaChatLlama: load_tensors:      Vulkan0 model buffer size =   662.50 MiB
I BanyaChatLlama: llama_kv_cache:    Vulkan0 KV buffer size =    15.00 MiB
I BanyaChatLlama: llama_context:    Vulkan0 compute buffer size =     0.59 MiB
I BanyaChatLlama: llama_context: Vulkan_Host compute buffer size =     1.01 MiB
```

**성공 지표:**
- ✅ Vulkan 초기화 성공
- ✅ Adreno 830 GPU 정상 감지 및 사용
- ✅ 5개 레이어 GPU 오프로드 성공
- ✅ "Failed to link shaders" 에러 완전 해결
- ✅ 메모리 할당 정상 (모델 버퍼 662.50 MiB, KV 캐시 15.00 MiB)

## 2. JNI 콜백 충돌 문제 해결 (2025-11-15)

### 2.1 문제 상황

앱이 한참 동작한 후 크래시가 발생했으며, 로그에서 다음과 같은 JNI 오류가 확인되었습니다:

```
F m.example.llama: JNI DETECTED ERROR IN APPLICATION: can't call void com.example.llama.ui.ChatViewModel$2$1.onLoadProgress(int) on instance of com.example.llama.ui.ChatViewModel$generate$1$1
F libc: Fatal signal 6 (SIGABRT), code -1 (SI_QUEUE) in tid 15916 (Thread-4)
```

**원인 분석:**
- `preload()`와 `generateStream()`에서 서로 다른 콜백 객체가 전달됨
- 모델 로딩이 완료된 후에도 `progressFn`이 호출될 수 있음
- 이전 콜백 객체가 이미 GC되었거나 다른 인스턴스로 대체됨

### 2.2 해결 방법

#### 2.2.1 Kotlin 레벨 수정

**파일:** `app/src/main/java/com/example/llama/data/ChatRepository.kt`

```kotlin
suspend fun generateStream(
    messages: List<ChatMessage>,
    callback: TokenCallback
) = withContext(Dispatchers.Default) {
    val prompt = formatPrompt(messages)
    Log.d("BanyaChat", "generateStream(): called with promptLen=${prompt.length}")
    // Ensure model is initialized, but don't pass callback if already initialized
    // to avoid JNI callback conflicts between preload and generateStream
    if (handle == 0L) {
        // Only pass callback if model is not yet initialized
        ensureInit(callback)
    } else {
        // Model already initialized, just ensure it's ready
        ensureInit(object : TokenCallback {
            override fun onLoadProgress(progress: Int) {}
            override fun onModelMetadata(json: String) {}
            override fun onToken(token: String) {}
            override fun onCompleted() {}
            override fun onError(message: String) {}
        })
    }
    // ... rest of the code
}
```

**해결 방법:**
- 이미 초기화된 경우 더미 콜백 객체를 전달하여 JNI 콜백 충돌 방지
- 모델이 아직 초기화되지 않은 경우에만 실제 콜백 전달

#### 2.2.2 JNI 레벨 수정

**파일:** `app/src/main/cpp/native/jni_bridge.cpp`

```cpp
auto progressFn = [](float progress, void * user) -> bool {
    auto* ctx = static_cast<LoadProgressContext*>(user);
    if (!ctx || !ctx->callback || !g_OnLoadProgress) {
        return true;
    }
    // Attach current thread to JVM (progress callback may run on different thread)
    JNIEnv* threadEnv = attachThread();
    if (!threadEnv) {
        ALOGE("progressFn(): Failed to attach thread");
        return true;
    }
    // Check if callback object is still valid before calling
    // This prevents JNI errors when callback object has been replaced or GC'd
    if (!threadEnv->IsSameObject(ctx->callback, nullptr)) {
        jint percent = static_cast<jint>(std::lround(progress * 100.0f));
        if (percent < 0) percent = 0;
        if (percent > 100) percent = 100;
        ALOGD("progressFn(): progress=%d%%", (int)percent);
        threadEnv->CallVoidMethod(ctx->callback, g_OnLoadProgress, percent);
        if (threadEnv->ExceptionCheck()) {
            ALOGE("progressFn(): Exception in callback - clearing and ignoring");
            threadEnv->ExceptionClear();
        }
    } else {
        ALOGD("progressFn(): callback object is null, skipping");
    }
    // Note: We don't detach here as the thread may be reused
    return true;
};
```

**해결 방법:**
- `IsSameObject()`로 콜백 객체 유효성 검사 추가
- 예외 발생 시 안전하게 처리하고 계속 진행
- null 체크로 GC된 객체 접근 방지

### 2.3 해결 결과

- ✅ JNI 콜백 충돌 완전 해결
- ✅ 모델 로딩 완료 후에도 안정적 동작
- ✅ 크래시 없이 정상 동작 확인

## 3. UI 스레드 안전성 보장

### 3.1 문제 상황

초기 구현에서 JNI 콜백이 백그라운드 스레드에서 직접 UI를 업데이트하려고 시도하여 문제가 발생할 수 있었습니다.

### 3.2 해결 방법

**파일:** `app/src/main/java/com/example/llama/ui/ChatViewModel.kt`

```kotlin
class ChatViewModel(app: Application) : AndroidViewModel(app) {
    private val repo = ChatRepository(app)
    private val mainHandler = Handler(Looper.getMainLooper())  // Main thread handler
    
    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState
    
    init {
        viewModelScope.launch(Dispatchers.Default) {
            repo.preload(object : TokenCallback {
                override fun onLoadProgress(progress: Int) {
                    // Dispatch to main thread for UI updates
                    mainHandler.post {
                        _uiState.value = _uiState.value.withProgress(progress)
                    }
                }
                override fun onModelMetadata(json: String) {
                    // ... metadata parsing ...
                    mainHandler.post {
                        _uiState.value = _uiState.value.withMetadata(metadata)
                    }
                }
                override fun onToken(token: String) {
                    mainHandler.post {
                        _uiState.value = _uiState.value.appendToLastAssistant(token)
                    }
                }
                override fun onCompleted() {
                    mainHandler.post {
                        _uiState.value = _uiState.value.copy(isGenerating = false)
                    }
                }
                override fun onError(message: String) {
                    mainHandler.post {
                        _uiState.value = _uiState.value.copy(isGenerating = false)
                        _uiState.value = _uiState.value.addMessage(
                            ChatMessage(text = "에러: $message", isUser = false)
                        )
                        _uiState.value = _uiState.value.withProgress(0)
                    }
                }
            })
        }
    }
}
```

**주요 개선사항:**
- `Handler(Looper.getMainLooper())`로 메인 스레드 핸들러 생성
- 모든 UI 업데이트를 `mainHandler.post { }`로 래핑
- 모델 로딩은 `Dispatchers.Default`에서 백그라운드 스레드로 실행

## 4. 모델 로딩 프로그레스 바 구현

### 4.1 구현 내용

**파일:** `app/src/main/java/com/example/llama/ui/ChatScreen.kt`

```kotlin
// Model loading progress indicator - show prominently when loading
if (uiState.loadProgress in 0..99) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                MaterialTheme.colorScheme.surfaceVariant,
                MaterialTheme.shapes.medium
            )
            .padding(12.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "모델 로딩 중...",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = "${uiState.loadProgress}%",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.primary
            )
        }
        Spacer(modifier = Modifier.height(8.dp))
        LinearProgressIndicator(
            progress = uiState.loadProgress / 100f,
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp),
            trackColor = MaterialTheme.colorScheme.surfaceVariant,
            color = MaterialTheme.colorScheme.primary
        )
    }
    Spacer(modifier = Modifier.height(8.dp))
}
```

**파일:** `app/src/main/java/com/example/llama/data/ChatRepository.kt`

```kotlin
private fun ensureInit(callback: TokenCallback) {
    if (handle != 0L) return
    Log.d("BanyaChat", "ensureInit(): start")
    // Notify UI that model loading is starting
    callback.onLoadProgress(0)  // Explicitly trigger progress bar UI
    // ... rest of initialization ...
}
```

**주요 기능:**
- 모델 로딩 시작 시 `onLoadProgress(0)` 호출로 프로그레스 바 표시
- 0-99% 범위에서 프로그레스 바 표시
- 100% 도달 시 프로그레스 바 자동 숨김
- Material Design 3 스타일 적용

## 5. CMake 빌드 설정

### 5.1 Vulkan 백엔드 활성화

**파일:** `app/build.gradle.kts`

```kotlin
externalNativeBuild {
    cmake {
        cppFlags("-O3", "-DNDEBUG", "-fvisibility=hidden")
        arguments(
            "-DUSE_LLAMA=ON",
            "-DGGML_VULKAN=ON",  // Vulkan backend 활성화
            "-DGGML_VULKAN_USE_VOLK=ON",  // VOLK 라이브러리 사용
            "-DGGML_K_QUANTS=ON",  // K-quantization 지원
            "-DCMAKE_MAKE_PROGRAM=${android.sdkDirectory}/cmake/3.22.1/bin/ninja",
            "-DCMAKE_PROGRAM_PATH=${android.sdkDirectory}/cmake/3.22.1/bin",
            "-DVulkan_GLSLC_EXECUTABLE=${android.ndkDirectory}/shader-tools/$hostTag/glslc",
            "-DVulkan_INCLUDE_DIR=${project.projectDir}/third_party/Vulkan-Headers/include",
            "-DCMAKE_INCLUDE_PATH=${project.projectDir}/third_party/Vulkan-Headers/include"
        )
    }
}
```

**주요 설정:**
- `GGML_VULKAN=ON`: Vulkan 백엔드 활성화
- `GGML_VULKAN_USE_VOLK=ON`: VOLK 라이브러리로 Vulkan 로딩
- `GGML_K_QUANTS=ON`: K-quantization 지원 (Q4_K 등)
- Vulkan 헤더 및 쉐이더 컴파일러 경로 설정

## 6. 최종 설정 요약

### 6.1 모델 설정
- **모델 타입**: Q4_0 (Q4_K 대비 쉐이더 호환성 향상)
- **모델 경로**: `/data/user/0/com.example.llama/files/models/llama31-banyaa-q4_0.gguf`
- **대체 경로**: Download 디렉토리 (`/sdcard/Download/llama31-banyaa-q4_0.gguf`)

### 6.2 Vulkan 설정
- **GPU 레이어**: 5개 (33개 중)
- **배치 크기**: 32 (메모리 압력 감소)
- **마이크로 배치**: 2 (동시 GPU 연산 최소화)
- **메모리 타입**: DEVICE_LOCAL (`no_host = true`)
- **추가 버퍼**: 비활성화 (`use_extra_bufts = false`)

### 6.3 샘플링 설정 (iOS와 동일)
- **Temperature**: 0.6
- **Top-P**: 0.9
- **Top-K**: 0 (비활성화)
- **Repeat Penalty**: 1.15
- **Repeat Last N**: 64

### 6.4 성능 지표
- **모델 버퍼**: 662.50 MiB (GPU 메모리)
- **KV 캐시**: 15.00 MiB (GPU 메모리)
- **Compute 버퍼**: 0.59 MiB (GPU) + 1.01 MiB (Host)
- **사용 가능한 GPU 메모리**: 15210 MiB

## 7. 문제 해결 체크리스트

### 7.1 해결된 문제들
- ✅ Q4_K 모델의 "Failed to link shaders" 에러
- ✅ Vulkan 드라이버 크래시 (`vkCmdBindPipeline`)
- ✅ JNI 콜백 충돌 (SIGABRT)
- ✅ UI 스레드 안전성 문제
- ✅ 모델 로딩 프로그레스 바 미표시

### 7.2 최적화 적용 사항
- ✅ DEVICE_LOCAL 메모리 사용으로 GPU 성능 향상
- ✅ 배치 크기 감소로 메모리 압력 완화
- ✅ GPU 레이어 최소화로 드라이버 안정성 확보
- ✅ Flash Attention 비활성화로 Android 호환성 향상

## 8. 참고 자료

- **llama.cpp 공식 문서**: https://github.com/ggerganov/llama.cpp
- **Vulkan 백엔드 가이드**: https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#vulkan
- **iOS 구현 참고**: `llamaImp.md` (이 문서의 상단 섹션)

---

**작성일**: 2025-11-15  
**대상 플랫폼**: Android (Adreno 830 GPU)  
**모델**: Llama 3.1 8B Q4_0

