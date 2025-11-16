# SET_ROWS Q4_0 크래시 상세 분석

## 현재 상황 요약

### 크래시 정보
- **크래시 위치**: OpenCL 드라이버 내부 (`qCLDrvAPI_clSetKernelArg+100`)
- **크래시 타입**: `SIGSEGV` (Segmentation Fault)
- **Fault Address**: `0x48008` (매우 작은 주소, NULL 포인터 접근 가능성)
- **크래시 시점**: `enqueue_ndrange_kernel` 호출 직후

### 로그 분석 결과

#### 성공한 단계들
1. ✅ **커널 선택**: Q4_0 i64 커널이 정상적으로 선택됨
   - 커널 포인터: `0xb400006da49f1cd0` (유효한 포인터)
2. ✅ **모든 인자 설정 성공**: 19개 인자 모두 `clSetKernelArg` 성공
   - `clSetKernelArg(10)` (fastdiv_vals ne11_): 성공
   - `clSetKernelArg(11)` (fastdiv_vals ne12_): 성공
   - 모든 인자 값이 정상적으로 설정됨
3. ✅ **Workgroup 크기 계산**: 정상 완료
   - `nth0=32`, `rows_per_workgroup=2`
   - `global_work_size=[128, 2, 1]`, `local_work_size=[32, 2, 1]`
4. ✅ **`enqueue_ndrange_kernel` 호출**: 함수 호출 자체는 성공
   - 로그에 "completed successfully" 출력

#### 문제 발생 지점
- **크래시 발생**: `enqueue_ndrange_kernel` 완료 직후 약 10ms 후
- **크래시 위치**: OpenCL 드라이버 내부 (`libOpenCL_adreno.so`)
- **스택 트레이스**: `qCLDrvAPI_clSetKernelArg+100`

## 문제 분석

### 1. `clEnqueueNDRangeKernel`의 비동기 특성

```cpp
// third_party/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp:610-621
void enqueue_ndrange_kernel(...) {
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, 
                                     global_work_size, local_work_size, 
                                     0, NULL, NULL));
}
```

**중요**: `clEnqueueNDRangeKernel`은 **비동기 함수**입니다:
- 함수 호출 자체는 즉시 반환됨 (성공 코드 반환)
- 실제 커널 실행은 GPU에서 비동기적으로 수행됨
- 따라서 "completed successfully" 로그는 함수 호출 성공만 의미함

### 2. 크래시 발생 시점 분석

**타임라인**:
1. `22:17:28.339` - `enqueue_ndrange_kernel` 호출 완료 (로그 출력)
2. `22:17:28.349` - 크래시 발생 (약 10ms 후)

**가능한 시나리오**:
1. **커널 실행 중 크래시**: GPU에서 커널이 실행되는 동안 OpenCL 드라이버 내부에서 메모리 접근 오류 발생
2. **커널 실행 후 정리 과정에서 크래시**: 커널 실행 완료 후 드라이버가 리소스를 정리하는 과정에서 문제 발생
3. **다음 작업 준비 중 크래시**: 다음 OpenCL 작업을 준비하는 과정에서 이전 작업의 리소스에 접근하려다 크래시

### 3. Fault Address 분석

**Fault Address**: `0x48008`
- 매우 작은 주소 (약 294KB)
- 일반적으로 NULL 포인터나 초기화되지 않은 포인터 접근을 의미
- OpenCL 드라이버 내부의 구조체 멤버 접근 중 문제 발생 가능성

### 4. Q4_0 커널 특이점

**Q4_0 커널의 특징**:
- F32 소스 데이터를 Q4_0 형식으로 양자화
- `quantize_q4_0_f32` 함수 사용
- 다른 커널들(F32, F16)과 달리 양자화 과정 포함

**가능한 문제**:
1. **메모리 접근 범위 초과**: 양자화 과정에서 버퍼 범위를 벗어난 접근
2. **정렬 문제**: Q4_0 블록 구조체의 메모리 정렬 문제
3. **커널 파라미터 불일치**: 커널 정의와 호출 시 파라미터 불일치

## 가능한 원인들

### 원인 1: 커널 실행 중 메모리 접근 오류 (가장 유력)
- **증거**: 크래시가 `enqueue_ndrange_kernel` 직후 발생
- **가능성**: Q4_0 커널 내부에서 잘못된 메모리 접근
- **확인 방법**: 커널 코드 검토, 메모리 범위 검증

### 원인 2: OpenCL 드라이버 버그
- **증거**: 크래시가 드라이버 내부에서 발생
- **가능성**: Adreno 드라이버의 Q4_0 커널 처리 버그
- **확인 방법**: 다른 GPU에서 테스트, 드라이버 업데이트

### 원인 3: 커널 인자 검증 지연
- **증거**: `clSetKernelArg`는 성공했지만 크래시는 드라이버 내부에서 발생
- **가능성**: 드라이버가 커널 실행 시점에 인자를 검증하면서 문제 발견
- **확인 방법**: 인자 값 재검증, 다른 값으로 테스트

### 원인 4: 메모리 정렬 문제
- **증거**: `fastdiv_vals` 구조체 전달
- **가능성**: 구조체 정렬이 OpenCL 커널의 `uint4`와 불일치
- **확인 방법**: 구조체 정렬 확인, `uint4`로 직접 전달 테스트

## 다음 단계

### 1. 커널 코드 검증
- Q4_0 커널의 메모리 접근 범위 확인
- `quantize_q4_0_f32` 함수의 버퍼 범위 검증
- 인덱스 계산 로직 검토

### 2. 동기화 추가
- `clFinish` 호출로 커널 실행 완료 대기
- 크래시 발생 시점 정확히 파악

### 3. 메모리 검증
- 버퍼 크기 확인
- 오프셋 값 검증
- 메모리 범위 초과 여부 확인

### 4. 대안 시도
- F32/F16 커널과 동일한 방식으로 처리 후 CPU에서 양자화
- 다른 양자화 방식 시도
- 커널 파라미터 조정

## 현재 상태

- ✅ SET_ROWS Q4_0 커널 구현 완료
- ✅ 커널 초기화 성공
- ✅ 인자 설정 성공
- ❌ 커널 실행 중/후 크래시 발생
- 🔄 디버깅 진행 중

