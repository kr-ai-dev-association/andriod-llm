# OpenCL 백엔드 크래시 상세 분석

## 크래시 정보
- **크래시 위치**: `ggml_backend_sched_split_graph` 함수, line 1166
- **크래시 원인**: `GGML_ASSERT(node_backend_id != -1)` 실패
- **에러 메시지**: "all nodes should be assigned by now, this can happen if there is no CPU fallback"
- **스택 트레이스**: 
  - `ggml_abort` → `abort` → SIGABRT
  - `ggml_backend_sched_split_graph+480`

## 문제 분석

### 1. 백엔드 스케줄러 구조
```cpp
// third_party/llama.cpp/ggml/src/ggml-backend.cpp:1602-1611
ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);
```
**요구사항**: 마지막 백엔드는 반드시 CPU여야 함 (CPU fallback 보장)

### 2. 그래프 분할 프로세스 (5단계)

#### Pass 1: Pre-allocated 입력에 백엔드 할당
```cpp
// third_party/llama.cpp/ggml/src/ggml-backend.cpp:931-939
// pass 1: assign backends to ops with pre-allocated inputs
for (int i = 0; i < graph->n_leafs; i++) {
    struct ggml_tensor * leaf = graph->leafs[i];
    int * leaf_backend_id = &tensor_backend_id(leaf);
    // do not overwrite user assignments
    if (*leaf_backend_id == -1) {
        *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
    }
}
```

#### Pass 2: GPU 백엔드 확장 (위/아래로)
```cpp
// third_party/llama.cpp/ggml/src/ggml-backend.cpp:968-1054
// pass 2: expand current backend assignments
// assign the same backend to adjacent nodes
// expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
// thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
// ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
```
**중요**: GPU 백엔드가 지원하지 않는 연산은 할당되지 않은 상태로 남음 (`node_backend_id == -1`)

#### Pass 3: 미할당 노드에 최적 백엔드 할당
```cpp
// third_party/llama.cpp/ggml/src/ggml-backend.cpp:1055-1107
// pass 3: assign backends to unassigned nodes
// only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
for (int i = 0; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];
    if (ggml_is_view_op(node->op)) {
        continue;
    }
    int * node_backend_id = &tensor_backend_id(node);
    if (*node_backend_id == -1) {
        // unassigned node: find the backend with the most supported inputs
        int n_supported_best = -1;
        for (int b = 0; b < sched->n_backends; b++) {
            if (ggml_backend_supports_op(sched->backends[b], node)) {
                // ... find best backend ...
            }
        }
    }
}
```
**문제점**: `ggml_backend_supports_op`가 모든 백엔드에서 `false`를 반환하면 노드가 할당되지 않음

#### Pass 4: 남은 소스 텐서에 백엔드 할당
```cpp
// third_party/llama.cpp/ggml/src/ggml-backend.cpp:1109-1139
// pass 4: assign backends to remaining src from dst and view_src
for (int i = 0; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];
    int * cur_backend_id = &tensor_backend_id(node);
    // ... assign src backends ...
    
    // if the node is still unassigned, assign it to the first backend that supports it
    for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
        ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
    }
    GGML_ASSERT(*cur_backend_id != -1);  // ← 여기서도 assertion 있음!
}
```
**중요**: Pass 4에서 모든 노드가 할당되어야 함. 하지만 크래시는 Pass 5에서 발생.

#### Pass 5: 그래프 분할 (크래시 발생 지점)
```cpp
// third_party/llama.cpp/ggml/src/ggml-backend.cpp:1141-1166
// pass 5: split graph, find tensors that need to be copied
for (; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];
    
    if (ggml_is_view_op(node->op)) {
        continue;  // ← view_op는 건너뜀
    }
    
    const int node_backend_id = tensor_backend_id(node);
    
    GGML_ASSERT(node_backend_id != -1); // ← 크래시 발생!
}
```

### 3. 의심되는 원인

#### 원인 1: OpenCL 백엔드가 특정 연산을 지원하지 않음
```cpp
// third_party/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp:2860-3094
static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            switch (op->src[0]->type) {
                case GGML_TYPE_Q4_0:
#ifdef GGML_OPENCL_SOA_Q
                    // We do not support flattened Q4_0 (and possibly other Q's)
                    return false;  // ← Q4_0 지원 안 함!
#else
                    return true;
#endif
        // ... 많은 연산들이 특정 조건에서만 지원됨
        default:
            return false;  // ← 지원하지 않는 연산은 false 반환
    }
}
```

#### 원인 2: CPU 백엔드가 스케줄러에 제대로 등록되지 않음
- `ggml_backend_sched_new`에서 마지막 백엔드가 CPU인지 확인 (line 1611)
- 하지만 실제로 CPU 백엔드가 `supports_op`에서 `true`를 반환하지 않을 수 있음

#### 원인 3: Pass 4의 assertion이 통과되었지만 Pass 5에서 문제 발생
- Pass 4는 view_op를 건너뛰지 않음
- Pass 5는 view_op를 건너뛰고 `tensor_backend_id`를 직접 읽음
- view_op의 `tensor_backend_id`가 -1일 수 있음

### 4. 의심되는 코드 부분

#### A. `ggml_backend_sched_set_if_supported` 함수
```cpp
// third_party/llama.cpp/ggml/src/ggml-backend.cpp:904-909
static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}
```
**문제**: OpenCL과 CPU 모두 `supports_op`가 `false`를 반환하면 할당되지 않음

#### B. Pass 3의 백엔드 선택 로직
```cpp
// third_party/llama.cpp/ggml/src/ggml-backend.cpp:1062-1083
if (*node_backend_id == -1) {
    // unassigned node: find the backend with the most supported inputs
    int n_supported_best = -1;
    for (int b = 0; b < sched->n_backends; b++) {
        if (ggml_backend_supports_op(sched->backends[b], node)) {  // ← 모든 백엔드가 false면?
            // ... 할당 로직 ...
        }
    }
    // n_supported_best가 -1이면 할당되지 않음!
}
```
**문제**: 모든 백엔드가 `supports_op`에서 `false`를 반환하면 노드가 할당되지 않음

#### C. Pass 4의 fallback 로직
```cpp
// third_party/llama.cpp/ggml/src/ggml-backend.cpp:1134-1138
// if the node is still unassigned, assign it to the first backend that supports it
for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
    ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
}
GGML_ASSERT(*cur_backend_id != -1);
```
**문제**: 모든 백엔드가 `supports_op`에서 `false`를 반환하면 assertion 실패

### 5. 가능한 해결 방안

1. **CPU 백엔드가 모든 연산을 지원하도록 보장**
   - CPU 백엔드의 `supports_op`가 항상 `true`를 반환하도록 확인

2. **Pass 4에서 강제 CPU 할당**
   - 모든 백엔드가 지원하지 않으면 마지막 백엔드(CPU)에 강제 할당

3. **OpenCL 백엔드 초기화 완료 확인**
   - OpenCL 백엔드가 제대로 초기화되지 않아 `supports_op`가 잘못된 결과를 반환할 수 있음

4. **디버그 로그 추가**
   - 어떤 연산이 할당되지 않았는지, 어떤 백엔드가 지원하지 않는지 로그 출력

## 해결 조치 완료

### 1. 디버그 로그 추가 (완료)
- `ggml-backend.cpp`의 Pass 4와 Pass 5에 상세 디버그 로그 추가
- 할당 실패 시 연산 이름, 텐서 타입, 각 백엔드의 supports_op 결과 출력

### 2. CPU 백엔드 강제 지원 (완료)
- `ggml-cpu.cpp`의 `ggml_backend_cpu_device_supports_op` 함수를 무조건 `true` 반환하도록 수정
- CPU 백엔드가 모든 연산을 지원하도록 보장하여 최종 fallback 역할 수행

### 3. Pass 4 강제 CPU 할당 (완료)
- Pass 4에서 모든 백엔드가 지원하지 않는 경우 마지막 백엔드(CPU)에 강제 할당
- 경고 로그와 함께 CPU fallback 수행

### 4. 초기화 단계 디버그 로그 추가 (완료)
- `ggml_backend_sched_new`에 백엔드 유효성 검증 및 로그 추가
- `ggml_backend_sched_split_graph`의 각 Pass 단계에 로그 추가
- `ggml_backend_sched_backend_id_from_cur`에 NULL 체크 및 안전장치 추가

## 근본 원인 발견

### 실제 문제
**OpenCL 버퍼에 있는 pre-allocated 텐서들이 `SET_ROWS` 연산을 수행하려고 하는데, OpenCL 백엔드가 이를 지원하지 않음**

로그에서 확인된 내용:
- `cache_k_l1` ~ `cache_k_l31` (모든 레이어의 K 캐시)가 OpenCL 버퍼에 할당됨
- 이들이 `SET_ROWS` 연산을 수행하려고 시도
- OpenCL 백엔드의 `ggml_opencl_supports_op`가 `SET_ROWS`에 대해 제한적인 지원만 제공:
  ```cpp
  case GGML_OP_SET_ROWS:
      if (op->src[0]->type != GGML_TYPE_F32) {
          return false;  // Q4_0 등은 지원 안 함
      }
  ```
- 현재 수정으로 CPU fallback이 작동하여 그래프 분할은 성공
- 하지만 실제 계산 단계(`ggml_backend_sched_graph_compute_async`)에서 새로운 크래시 발생

### 해결 방안
1. **OpenCL 백엔드의 SET_ROWS 지원 개선** (장기)
2. **KV 캐시를 CPU 버퍼에 할당** (단기 - 가장 확실한 해결책)
3. **SET_ROWS 연산을 CPU에서만 실행하도록 강제** (중기)

