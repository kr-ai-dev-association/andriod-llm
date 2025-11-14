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

