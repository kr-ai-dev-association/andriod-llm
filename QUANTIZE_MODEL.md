# 모델 Q4_0 양자화 가이드

원본 모델: https://huggingface.co/banyaaiofficial/llama3.1_LoRA

## 1. 환경 준비

```bash
cd /Volumes/Transcend/Projects/andriod-llm/third_party/llama.cpp

# Python 의존성 설치
pip3 install -r requirements.txt
```

## 2. Hugging Face 모델 다운로드

```bash
# Hugging Face CLI 설치 (선택사항)
pip3 install huggingface_hub

# 모델 다운로드
python3 -m huggingface_hub download banyaaiofficial/llama3.1_LoRA --local-dir ./models/llama3.1_LoRA
```

또는 수동으로 다운로드:
- https://huggingface.co/banyaaiofficial/llama3.1_LoRA 에서 모델 파일들을 다운로드
- `./models/llama3.1_LoRA/` 디렉토리에 저장

## 3. Hugging Face 모델을 GGUF로 변환

```bash
# F16 형식으로 변환
python3 convert_hf_to_gguf.py ./models/llama3.1_LoRA --outdir ./models/llama3.1_LoRA_gguf --outtype f16
```

## 4. Q4_0으로 양자화

```bash
# llama-quantize 빌드 (필요한 경우)
cmake -B build
cmake --build build --config Release -t llama-quantize

# Q4_0으로 양자화
./build/bin/llama-quantize \
    ./models/llama3.1_LoRA_gguf/ggml-model-f16.gguf \
    ./models/llama3.1_LoRA_gguf/ggml-model-Q4_0.gguf \
    Q4_0
```

또는 간단하게:
```bash
./build/bin/llama-quantize \
    ./models/llama3.1_LoRA_gguf/ggml-model-f16.gguf \
    Q4_0
```

## 5. Android 앱에 모델 복사

```bash
# 양자화된 모델을 Android 앱의 assets 또는 다운로드 디렉토리로 복사
adb push ./models/llama3.1_LoRA_gguf/ggml-model-Q4_0.gguf /sdcard/Download/
```

또는 Android Studio에서:
- `app/src/main/assets/` 디렉토리에 `ggml-model-Q4_0.gguf` 파일 복사

## 참고사항

- Q4_0은 Q4_K보다 약간 낮은 정확도를 가지지만, AdrenoVK 드라이버와 호환성이 좋습니다
- 모델 크기: Q4_0은 약 4.34GB (Llama-3-8B 기준)
- 양자화 시간: 모델 크기에 따라 수분~수십분 소요될 수 있습니다

