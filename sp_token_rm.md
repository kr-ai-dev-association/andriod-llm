# 특수 토큰 제거 로직 상세 문서

## 개요

BanyaLLM은 Llama 3.1 모델을 사용하며, 모델이 생성하는 응답에서 특수 토큰(special tokens)을 제거하는 다층 필터링 시스템을 구현하고 있습니다. 특수 토큰은 모델의 내부 제어를 위한 메타데이터이므로 사용자에게 표시되어서는 안 됩니다.

## 필터링 계층 구조

특수 토큰 제거는 **2단계 필터링**으로 구성됩니다:

1. **토큰 레벨 필터링** (`LlamaContext.swift`) - 각 토큰이 생성될 때 즉시 필터링
2. **텍스트 레벨 필터링** (`LlamaManager.swift`) - 누적된 전체 텍스트에서 추가 필터링

---

## 1단계: 토큰 레벨 필터링

**위치**: `LlamaContext.swift` → `completionLoop()` 함수

### 목적
각 토큰이 생성되는 즉시 특수 토큰을 제거하여 초기 단계에서 불필요한 텍스트가 누적되지 않도록 합니다.

### 처리 단계

#### 1.1 완전한 특수 토큰 패턴 제거

**대상 패턴**:
```swift
let specialTokenPatterns = [
    "<|begin_of_text|>",      // 텍스트 시작 마커
    "<|end_of_text|>",        // 텍스트 종료 마커
    "<|start_header_id|>",    // 헤더 시작
    "<|end_header_id|>",      // 헤더 종료
    "<|eot_id|>",             // End of Turn ID
    "<|eom_id|>",             // End of Message ID
    "<|python_tag|>",         // Python 코드 태그
    "<|finetune_right_pad_id|>" // 파인튜닝 패딩 ID
]
```

**처리 방식**: 단순 문자열 치환
```swift
for pattern in specialTokenPatterns {
    new_token_str = new_token_str.replacingOccurrences(of: pattern, with: "")
}
```

#### 1.2 Reserved Special Token 패턴 제거

**대상 패턴**: `<|reserved_special_token_\d+|>`

**예시**:
- `<|reserved_special_token_1|>`
- `<|reserved_special_token_128|>`

**처리 방식**: 정규식 사용
```swift
let regex = NSRegularExpression(pattern: "<\\|reserved_special_token_\\d+\\|>")
```

**정규식 설명**:
- `<\\|` - 리터럴 `<|` (파이프는 이스케이프 필요)
- `reserved_special_token_` - 고정 문자열
- `\\d+` - 하나 이상의 숫자
- `\\|>` - 리터럴 `|>`

#### 1.3 부분 특수 토큰 패턴 제거

토큰이 분해되어 생성되는 경우를 처리합니다.

**대상 패턴**:
- `<|` - 특수 토큰 시작 부분
- `|>` - 특수 토큰 종료 부분
- 단독 `|` - 파이프 문자만 있는 경우

**처리 방식**:

1. **단독 파이프 제거**:
```swift
if new_token_str == "|" {
    new_token_str = ""
}
```

2. **부분 패턴 포함 시 제거**:
```swift
if new_token_str.contains("<|") || new_token_str.contains("|>") {
    new_token_str = new_token_str.replacingOccurrences(of: "<|", with: "")
    new_token_str = new_token_str.replacingOccurrences(of: "|>", with: "")
}
```

3. **정규식으로 부분 패턴 제거**:
```swift
let regex = NSRegularExpression(pattern: "<\\|.*?\\|>")
```

이 정규식은 `<|`와 `|>` 사이의 모든 문자를 포함하는 패턴을 찾아 제거합니다.

---

## 2단계: 텍스트 레벨 필터링

**위치**: `LlamaManager.swift` → `generate()` 함수 내부 → `filterSpecialTokens()` 함수

### 목적
토큰 레벨 필터링을 통과한 특수 토큰이나, 여러 토큰이 조합되어 생성된 특수 토큰 패턴을 제거합니다.

### 처리 단계

#### 2.1 완전한 특수 토큰 패턴 제거 (반복 처리)

**특징**: 중첩된 패턴도 처리하기 위해 반복적으로 제거합니다.

**처리 방식**:
```swift
var previousLength = 0
var iterations = 0
while cleaned.count != previousLength && iterations < 10 {
    previousLength = cleaned.count
    for pattern in specialTokenPatterns {
        cleaned = cleaned.replacingOccurrences(of: pattern, with: "")
    }
    iterations += 1
}
```

**예시**:
- 입력: `"안녕<|eot_id|>하세요<|eot_id|>"`
- 1회차: `"안녕하세요"`
- 2회차: 변화 없음 → 종료

**안전장치**: 최대 10회 반복으로 무한 루프 방지

#### 2.2 Reserved Special Token 패턴 제거

토큰 레벨과 동일한 정규식을 사용하지만, 누적된 텍스트 전체에 대해 적용합니다.

```swift
let regex = NSRegularExpression(pattern: "<\\|reserved_special_token_\\d+\\|>")
```

#### 2.3 부분 특수 토큰 패턴 제거 (공격적 필터링)

여러 토큰이 조합되어 특수 토큰이 생성되는 경우를 처리합니다.

**처리 방법 4가지**:

**방법 1: `<|` + `|>` 조합 찾기**
```swift
if let startRange = cleaned.range(of: "<|", options: .backwards),
   let endRange = cleaned.range(of: "|>", range: startRange.upperBound..<cleaned.endIndex) {
    cleaned = String(cleaned[..<startRange.lowerBound]) + 
             String(cleaned[endRange.upperBound...])
}
```

- `backwards` 옵션으로 뒤에서부터 검색하여 가장 최근 패턴부터 제거
- `<|`와 `|>` 사이의 모든 내용을 제거

**방법 2: 단독 파이프 제거**
```swift
if cleaned.contains("|") && !cleaned.contains("<|") && !cleaned.contains("|>") {
    cleaned = cleaned.replacingOccurrences(of: "|", with: "")
}
```

- 특수 토큰 패턴이 아닌 단독 파이프만 제거
- 조건: `<|`나 `|>`가 없는 경우에만

**방법 3: 정규식으로 부분 패턴 제거**
```swift
let regex = NSRegularExpression(pattern: "<\\|[^|]*\\|>")
```

- `<|`와 `|>` 사이에 파이프가 없는 모든 패턴 제거
- `[^|]*`는 파이프를 제외한 모든 문자 매칭

**방법 4: 공백과 결합된 패턴 제거**
```swift
cleaned = cleaned.replacingOccurrences(of: " <|", with: "")
cleaned = cleaned.replacingOccurrences(of: "<| ", with: "")
cleaned = cleaned.replacingOccurrences(of: " |>", with: "")
cleaned = cleaned.replacingOccurrences(of: "|> ", with: "")
```

- 공백과 함께 나타나는 특수 토큰 패턴 제거
- 예: `"안녕 <|eot_id|> 하세요"` → `"안녕 하세요"`

**안전장치**: 최대 10회 반복으로 무한 루프 방지

#### 2.4 기타 이상한 패턴 제거

**대상**: `<kts:1>`, `<tag>`, `<unknown>` 등 HTML/XML 스타일 태그

**처리 방식**:
```swift
let regex = NSRegularExpression(pattern: "<[^>]*>")
```

- `<`로 시작하고 `>`로 끝나는 모든 패턴 제거
- `[^>]*`는 `>`를 제외한 모든 문자 매칭

**주의사항**: 이 패턴은 실제 HTML 태그도 제거할 수 있으므로, 모델이 HTML을 생성하는 경우 주의가 필요합니다.

#### 2.5 특수 문자 조합 제거

**대상**: `^^`, `^^^` 등 불필요한 특수 문자 조합

**처리 방식**:
```swift
cleaned = cleaned.replacingOccurrences(of: "^^", with: "")
cleaned = cleaned.replacingOccurrences(of: "^^^", with: "")
```

**목적**: 모델이 생성하는 불필요한 이모지나 특수 문자 제거

---

## 필터링 흐름도

```
토큰 생성 (LlamaContext)
    ↓
[1단계: 토큰 레벨 필터링]
    ├─ 완전한 특수 토큰 패턴 제거
    ├─ Reserved Special Token 제거
    └─ 부분 특수 토큰 패턴 제거
    ↓
토큰 문자열 반환
    ↓
누적 (accumulatedRaw)
    ↓
[2단계: 텍스트 레벨 필터링]
    ├─ 완전한 특수 토큰 패턴 제거 (반복)
    ├─ Reserved Special Token 제거
    ├─ 부분 특수 토큰 패턴 제거 (4가지 방법)
    ├─ 기타 이상한 패턴 제거
    └─ 특수 문자 조합 제거
    ↓
최종 응답 (cleanedText)
```

---

## 처리 예시

### 예시 1: 완전한 특수 토큰
**입력**: `"안녕하세요<|eot_id|>반갑습니다"`
**1단계 결과**: `"안녕하세요반갑습니다"`
**2단계 결과**: `"안녕하세요반갑습니다"` (변화 없음)

### 예시 2: 분해된 특수 토큰
**입력**: `"안녕<|하세요|>반갑습니다"`
**1단계 결과**: `"안녕하세요반갑습니다"` (부분 패턴 제거)
**2단계 결과**: `"안녕하세요반갑습니다"` (변화 없음)

### 예시 3: 중첩된 패턴
**입력**: `"안녕<|eot_id|><|eom_id|>하세요"`
**1단계 결과**: `"안녕하세요"` (각 토큰에서 제거)
**2단계 결과**: `"안녕하세요"` (추가 안전장치)

### 예시 4: Reserved Token
**입력**: `"안녕<|reserved_special_token_128|>하세요"`
**1단계 결과**: `"안녕하세요"` (정규식으로 제거)
**2단계 결과**: `"안녕하세요"` (변화 없음)

### 예시 5: 공백과 결합
**입력**: `"안녕 <|eot_id|> 하세요"`
**1단계 결과**: `"안녕  하세요"` (토큰 제거, 공백 남음)
**2단계 결과**: `"안녕 하세요"` (공백 패턴 제거)

---

## 성능 고려사항

1. **반복 제한**: 각 필터링 단계에서 최대 반복 횟수를 제한하여 무한 루프 방지
2. **정규식 최적화**: 간단한 패턴은 문자열 치환 사용, 복잡한 패턴만 정규식 사용
3. **조기 종료**: 텍스트 길이가 변하지 않으면 즉시 종료

---

## 주의사항

1. **HTML/XML 태그 제거**: 2.4단계에서 `<[^>]*>` 패턴이 실제 HTML 태그도 제거할 수 있음
2. **파이프 문자**: 실제 파이프 문자(`|`)를 사용하는 경우도 제거될 수 있음
3. **성능**: 매우 긴 텍스트의 경우 반복 필터링으로 인한 성능 저하 가능

---

## 개선 제안

1. **화이트리스트 방식**: 특정 패턴만 제거하도록 화이트리스트 방식 도입
2. **컨텍스트 인식**: HTML 태그와 특수 토큰을 구분하는 로직 추가
3. **캐싱**: 동일한 패턴 반복 제거 시 캐싱 활용
4. **로깅**: 제거된 특수 토큰을 로그로 기록하여 디버깅 용이성 향상

---

## 관련 파일

- `BanyaLLM/Models/LlamaContext.swift` - 토큰 레벨 필터링
- `BanyaLLM/Models/LlamaManager.swift` - 텍스트 레벨 필터링 (라인 488-575)

---

## 버전 정보

- **작성일**: 2025-01-13
- **대상 모델**: Llama 3.1
- **Swift 버전**: 5.0+

