#include <jni.h>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <random>
#include <android/log.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <nlohmann/json.hpp>

#ifndef LLAMA_STUB_MODE
#define LLAMA_STUB_MODE 0
#endif

#if LLAMA_STUB_MODE
// Stub mode: no llama.cpp includes. Provide minimal behavior.
#else
#include "llama.h"
#endif

static JavaVM* g_JavaVM = nullptr;
static jclass g_CallbackClass = nullptr;
static jmethodID g_OnToken = nullptr;
static jmethodID g_OnCompleted = nullptr;
static jmethodID g_OnError = nullptr;
static jmethodID g_OnLoadProgress = nullptr;
static jmethodID g_OnModelMetadata = nullptr;

static void ensureCallbackRefs(JNIEnv* env, jobject callback);

#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "BanyaChatJNI", __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, "BanyaChatJNI", __VA_ARGS__)

static void llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    const char* tag = "BanyaChatLlama";
    int priority = ANDROID_LOG_DEFAULT;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR:
            priority = ANDROID_LOG_ERROR;
            break;
        case GGML_LOG_LEVEL_WARN:
            priority = ANDROID_LOG_WARN;
            break;
        case GGML_LOG_LEVEL_INFO:
            priority = ANDROID_LOG_INFO;
            break;
        case GGML_LOG_LEVEL_DEBUG:
        default:
            priority = ANDROID_LOG_DEBUG;
            break;
    }
    // llama.cpp log is often multi-line, but android log truncates after newline.
    // So, we print line by line.
    const char *start = text;
    const char *end = start;
    while (*end) {
        while (*end && *end != '\n') {
            end++;
        }
        std::string line(start, end - start);
        __android_log_print(priority, tag, "%s", line.c_str());
        if (*end == '\n') {
            end++;
        }
        start = end;
    }
}


struct LlamaCtx {
#if LLAMA_STUB_MODE
    int dummy;
#else
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
#endif
    std::atomic<bool> stopRequested = false;
};

struct LoadProgressContext {
    jobject callback = nullptr;
    std::atomic<bool> completed = false;  // Track if loading is completed
};

// Forward declarations
static JNIEnv* attachThread();
static void detachThread();

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
	g_JavaVM = vm;
	llama_log_set(llama_log_callback, nullptr); // Set log callback
	llama_backend_init(); // false = no NUMA
	ALOGD("JNI_OnLoad: LLAMA_STUB_MODE=%d", LLAMA_STUB_MODE);
	return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_init(
        JNIEnv* env, jobject /*thiz*/,
        jstring jModelPath,
        jint nCtx, jint nThreads, jint nBatch, jint nGpuLayers,
        jboolean useMmap, jboolean useMlock, jint seed,
        jobject callback) {
    auto* handle = new LlamaCtx();
#if LLAMA_STUB_MODE
    (void) jModelPath;
    (void) nCtx; (void) nThreads; (void) nBatch; (void) nGpuLayers;
    (void) useMmap; (void) useMlock; (void) seed; (void) callback;
    ALOGD("init(): STUB build active. Returning dummy handle.");
    return reinterpret_cast<jlong>(handle);
#else
    if (callback) {
        ensureCallbackRefs(env, callback);
    }

    jobject callbackGlobal = callback ? env->NewGlobalRef(callback) : nullptr;
    // Allocate progress context on heap so it persists during async model loading
    auto* progressCtx = new LoadProgressContext{callbackGlobal};

    auto progressFn = [](float progress, void * user) -> bool {
        auto* ctx = static_cast<LoadProgressContext*>(user);
        if (!ctx) {
            return true;
        }
        // If loading is already completed, don't do anything
        if (ctx->completed.load()) {
            return true;
        }
        // Just track progress, don't call callback from background thread
        // This prevents JNI callback conflicts when callback object is replaced
        jint percent = static_cast<jint>(std::lround(progress * 100.0f));
        if (percent < 0) percent = 0;
        if (percent > 100) percent = 100;
        
        // Mark as completed when reaching 100%
        if (percent >= 100) {
            ctx->completed.store(true);
        }
        
        // Log progress but don't call callback from background thread
        // Callback will be called from main thread after model loading completes
        ALOGD("progressFn(): progress=%d%% (callback disabled to prevent JNI conflicts)", (int)percent);
        return true;
    };

    const char* path = env->GetStringUTFChars(jModelPath, nullptr);
    ALOGD("init(): modelPath=%s nCtx=%d nThreads=%d nBatch=%d nGpuLayers=%d useMmap=%d useMlock=%d seed=%d",
          path ? path : "(null)", (int)nCtx, (int)nThreads, (int)nBatch, (int)nGpuLayers, (int)useMmap, (int)useMlock, (int)seed);

    llama_model_params mparams = llama_model_default_params();
    // Optimized Vulkan settings for Adreno 830
    // Reduced GPU layers to 29 to test if 30th layer causes crashes
    if (nGpuLayers == -1) {
        // Default to 29 layers - testing to find maximum stable layers
        mparams.n_gpu_layers = 29;
    } else {
        mparams.n_gpu_layers = nGpuLayers;
    }
    mparams.use_mmap = useMmap;
    mparams.use_mlock = useMlock;
    // Q4_0 doesn't require extra buffers
    mparams.use_extra_bufts = false;
    // Use DEVICE_LOCAL memory (no_host=true) for better GPU performance and stability
    // This ensures model weights are stored in GPU memory, reducing host-device transfers
    mparams.no_host = true;  // DEVICE_LOCAL memory for GPU weights
    mparams.progress_callback = progressFn;
    mparams.progress_callback_user_data = progressCtx;

    ALOGD("init(): Calling llama_model_load_from_file with n_gpu_layers=%d, use_extra_bufts=%d, no_host=%d...",
          (int)mparams.n_gpu_layers, (int)mparams.use_extra_bufts, (int)mparams.no_host);
    llama_model* model = llama_model_load_from_file(path, mparams);
    ALOGD("init(): llama_model_load_from_file returned. model is %s", model ? "valid" : "null");

    env->ReleaseStringUTFChars(jModelPath, path);

    if (!model) {
        ALOGE("init(): llama_load_model_from_file failed");
        if (callbackGlobal && g_OnError) {
            jstring err = env->NewStringUTF("모델을 로드할 수 없습니다.");
            // Safely call error callback with type validation
            if (g_CallbackClass && !env->IsInstanceOf(callbackGlobal, g_CallbackClass)) {
                ALOGE("init(): Error callback object is not an instance of TokenCallback - skipping");
            } else {
                env->CallVoidMethod(callbackGlobal, g_OnError, err);
                if (env->ExceptionCheck()) {
                    ALOGE("init(): Exception in error callback - clearing");
                    env->ExceptionClear();
                }
            }
            env->DeleteLocalRef(err);
        }
        if (callbackGlobal) env->DeleteGlobalRef(callbackGlobal);
        delete progressCtx;  // Clean up progress context on error
        delete handle;
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = nCtx;
    cparams.n_threads = nThreads;
    // Use the same number of threads for batched prompt processing to speed up prefill
    cparams.n_threads_batch = nThreads;
    cparams.n_batch = nBatch;
    // Reduced micro-batch size to minimize memory pressure and Vulkan operations
    // Lower n_ubatch reduces concurrent GPU operations, improving stability on Adreno 830
    cparams.n_ubatch = 2;  // Restored to stable value (was 1, but may cause issues)
    // TEST: V-Cache Q4_0 양자화가 무작위 토큰 생성의 원인인지 확인하기 위한 테스트
    // V-Cache만 F16으로 되돌려서 문제를 분리합니다
    // 만약 이 변경으로 정상적인 텍스트가 생성된다면, V-Cache Q4_0 처리 과정(패딩 또는 디퀀타이즈 커널)에 문제가 있음이 확정됩니다
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    cparams.type_k = GGML_TYPE_Q4_0;  // K-Cache는 Q4_0 유지
    cparams.type_v = GGML_TYPE_F16;   // V-Cache를 F16으로 복원 (테스트용)

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        ALOGE("init(): llama_new_context_with_model failed - possible VRAM shortage for KV Cache");
        if (callbackGlobal && g_OnError) {
            jstring err = env->NewStringUTF("컨텍스트 초기화에 실패했습니다. VRAM 부족일 수 있습니다.");
            // Safely call error callback with type validation
            if (g_CallbackClass && !env->IsInstanceOf(callbackGlobal, g_CallbackClass)) {
                ALOGE("init(): Error callback object is not an instance of TokenCallback - skipping");
            } else {
                env->CallVoidMethod(callbackGlobal, g_OnError, err);
                if (env->ExceptionCheck()) {
                    ALOGE("init(): Exception in error callback - clearing");
                    env->ExceptionClear();
                }
            }
            env->DeleteLocalRef(err);
        }
        llama_model_free(model);
        if (callbackGlobal) env->DeleteGlobalRef(callbackGlobal);
        delete progressCtx;  // Clean up progress context on error
        delete handle;
        return 0;
    }

    // Mark progress as completed before calling final callback
    if (progressCtx) {
        progressCtx->completed.store(true);
    }
    // Safely call final progress callback with type validation
    if (callbackGlobal && g_OnLoadProgress) {
        // Verify callback type before calling to prevent JNI errors
        if (g_CallbackClass && !env->IsInstanceOf(callbackGlobal, g_CallbackClass)) {
            ALOGE("init(): Final callback object is not an instance of TokenCallback - skipping");
        } else {
            env->CallVoidMethod(callbackGlobal, g_OnLoadProgress, 100);
            if (env->ExceptionCheck()) {
                ALOGE("init(): Exception in final progress callback - clearing");
                env->ExceptionClear();
            }
        }
    }

    if (callbackGlobal && g_OnModelMetadata) {
        auto metaValue = [&](const char * key) -> std::string {
            char buf[512];
            int32_t len = llama_model_meta_val_str(model, key, buf, sizeof(buf));
            if (len >= 0) {
                return std::string(buf, len);
            }
            return "";
        };

        nlohmann::json meta;
        std::string name = metaValue("general.name");
        std::string quant = metaValue("general.file_type");
        std::string sizeLabel = metaValue("general.size_label");
        std::string contextStr = metaValue("general.context_length");

        meta["name"] = name.empty() ? "(unknown)" : name;
        meta["quantization"] = quant.empty() ? "unknown" : quant;
        meta["size_label"] = sizeLabel.empty() ? "unknown" : sizeLabel;
        meta["context_length"] = contextStr.empty() ? static_cast<int>(nCtx) : std::atoi(contextStr.c_str());

        std::string metaDump = meta.dump();
        jstring metaJson = env->NewStringUTF(metaDump.c_str());
        // Safely call metadata callback with type validation
        if (g_CallbackClass && !env->IsInstanceOf(callbackGlobal, g_CallbackClass)) {
            ALOGE("init(): Metadata callback object is not an instance of TokenCallback - skipping");
        } else {
            env->CallVoidMethod(callbackGlobal, g_OnModelMetadata, metaJson);
            if (env->ExceptionCheck()) {
                ALOGE("init(): Exception in metadata callback - clearing");
                env->ExceptionClear();
            }
        }
        env->DeleteLocalRef(metaJson);
    }

    handle->model = model;
    handle->ctx = ctx;
    ALOGD("init(): success, handle=%p", (void*)handle);

    // Don't call progress callback from JNI to avoid callback conflicts
    // Kotlin layer will detect model loading completion by checking if handle != 0
    // and update progress bar accordingly

    // Clean up progress context (callback is stored in handle if needed later)
    // Note: progressCtx->callback is the same as callbackGlobal, so we'll delete it below
    delete progressCtx;
    
    if (callbackGlobal) {
        env->DeleteGlobalRef(callbackGlobal);
    }
    return reinterpret_cast<jlong>(handle);
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_free(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong h) {
    auto* handle = reinterpret_cast<LlamaCtx*>(h);
    if (!handle) return;
#if LLAMA_STUB_MODE
#else
    if (handle->ctx) llama_free(handle->ctx);
    if (handle->model) llama_model_free(handle->model);
#endif
    delete handle;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_completionStop(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong h) {
    auto* handle = reinterpret_cast<LlamaCtx*>(h);
    if (!handle) return;
    handle->stopRequested = true;
}

static void ensureCallbackRefs(JNIEnv* env, jobject callback) {
    if (!g_CallbackClass) {
        jclass local = env->GetObjectClass(callback);
        g_CallbackClass = reinterpret_cast<jclass>(env->NewGlobalRef(local));
        env->DeleteLocalRef(local);
        g_OnToken = env->GetMethodID(g_CallbackClass, "onToken", "(Ljava/lang/String;)V");
        g_OnCompleted = env->GetMethodID(g_CallbackClass, "onCompleted", "()V");
        g_OnError = env->GetMethodID(g_CallbackClass, "onError", "(Ljava/lang/String;)V");
        g_OnLoadProgress = env->GetMethodID(g_CallbackClass, "onLoadProgress", "(I)V");
        g_OnModelMetadata = env->GetMethodID(g_CallbackClass, "onModelMetadata", "(Ljava/lang/String;)V");
        ALOGD("ensureCallbackRefs(): methods cached");
    }
}

static JNIEnv* attachThread() {
	if (!g_JavaVM) return nullptr;
	JNIEnv* env = nullptr;
	jint res = g_JavaVM->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
	if (res == JNI_OK) return env;
	#if defined(__ANDROID__)
	if (g_JavaVM->AttachCurrentThread(&env, nullptr) != 0) return nullptr;
	#else
	if (g_JavaVM->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr) != 0) return nullptr;
	#endif
	return env;
}

static void detachThread() {
	if (!g_JavaVM) return;
	g_JavaVM->DetachCurrentThread();
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_completionStart(
        JNIEnv* env, jobject /*thiz*/, jlong h,
        jstring jPrompt, jint numPredict, jfloat temperature, jfloat topP, jint topK,
        jfloat repeatPenalty, jint repeatLastN,
        jobjectArray jStopSequences, jobject callback) {
    auto* handle = reinterpret_cast<LlamaCtx*>(h);
    if (!handle) return;
    ensureCallbackRefs(env, callback);

    jobject gCallback = env->NewGlobalRef(callback);
    const char* prompt = env->GetStringUTFChars(jPrompt, nullptr);
    std::string promptStr(prompt ? prompt : "");
    env->ReleaseStringUTFChars(jPrompt, prompt);

    // Defaults if invalid values are passed
    int n_predict = (numPredict > 0) ? numPredict : 100;
    float temp = (temperature > 0.0f) ? temperature : 0.7f;  // Llama 3.1 기본값에 가까운 값
    float top_p = (topP > 0.0f) ? topP : 0.9f;  // Llama 3.1 권장값
    int top_k = (topK > 0) ? topK : 40;  // Llama 3.1 기본값
    float rep_penalty = (repeatPenalty > 0.0f) ? repeatPenalty : 1.1f;  // 한국어 튜닝 모델에 적합한 값
    int rep_last_n = (repeatLastN > 0) ? repeatLastN : 256;
    std::vector<std::string> stops;
    if (jStopSequences) {
        jsize len = env->GetArrayLength(jStopSequences);
        for (jsize i = 0; i < len; ++i) {
            jstring s = (jstring) env->GetObjectArrayElement(jStopSequences, i);
            const char* cs = env->GetStringUTFChars(s, nullptr);
            stops.emplace_back(cs ? cs : "");
            env->ReleaseStringUTFChars(s, cs);
            env->DeleteLocalRef(s);
        }
    }

    handle->stopRequested = false;

    std::thread worker([gCallback, handle, promptStr, n_predict, temp, top_p, top_k, rep_penalty, rep_last_n, stops]() {
        ALOGD("completionStart(): worker thread started");
		JNIEnv* threadEnv = attachThread();
		if (!threadEnv) {
            ALOGE("completionStart(): could not attach thread to JVM");
			return;
		}
        ALOGD("completionStart(): worker thread attached to JVM");

        struct CallbackGuard {
            JNIEnv* env;
            jobject ref;
            bool shouldDelete;
            CallbackGuard() : env(nullptr), ref(nullptr), shouldDelete(true) {}
            ~CallbackGuard() {
                if (env && ref && shouldDelete) {
                    // Only delete if thread is still attached (env is valid)
                    // Note: We can't check if thread is attached safely, so we rely on shouldDelete flag
                    // which should be set to false before detachThread() is called
                    env->DeleteGlobalRef(ref);
                }
            }
        } guard;
        guard.env = threadEnv;
        guard.ref = gCallback;

        llama_context* ctx = handle->ctx;
        if (!ctx) {
            ALOGE("completionStart(): ctx is null");
            jstring err = threadEnv->NewStringUTF("Context is null");
            if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
                threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                if (threadEnv->ExceptionCheck()) {
                    ALOGE("completionStart(): Exception in error callback - clearing");
                    threadEnv->ExceptionClear();
                }
            } else {
                ALOGE("completionStart(): Callback validation failed - skipping error callback");
            }
            threadEnv->DeleteLocalRef(err);
            // Prevent CallbackGuard from trying to delete gCallback after detachThread()
            if (guard.env && guard.ref) {
                guard.env->DeleteGlobalRef(guard.ref);
                guard.ref = nullptr;
            }
            guard.shouldDelete = false;
            detachThread();
            return;
        }

        llama_model* model = handle->model;
        if (!model) {
            ALOGE("completionStart(): model is null");
            jstring err = threadEnv->NewStringUTF("Model is null");
            if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
                threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                if (threadEnv->ExceptionCheck()) {
                    ALOGE("completionStart(): Exception in error callback - clearing");
                    threadEnv->ExceptionClear();
                }
            } else {
                ALOGE("completionStart(): Callback validation failed - skipping error callback");
            }
            threadEnv->DeleteLocalRef(err);
            // Prevent CallbackGuard from trying to delete gCallback after detachThread()
            if (guard.env && guard.ref) {
                guard.env->DeleteGlobalRef(guard.ref);
                guard.ref = nullptr;
            }
            guard.shouldDelete = false;
            detachThread();
            return;
        }
        
        const struct llama_vocab * vocab = llama_model_get_vocab(model);

        // tokenize prompt
        std::vector<llama_token> prompt_tokens;
        prompt_tokens.resize(promptStr.size() + 16);
        ALOGD("completionStart(): tokenizing prompt...");
        ALOGD("completionStart(): prompt length=%zu, first 200 chars: %.200s", promptStr.length(), promptStr.c_str());
        // Check if prompt already starts with BOS token
        bool has_bos = (promptStr.length() > 0 && promptStr.find("<|begin_of_text|>") == 0);
        ALOGD("completionStart(): has_bos=%d, add_bos=%d", has_bos ? 1 : 0, !has_bos ? 1 : 0);
        int n_tokens = llama_tokenize(
            vocab,
            promptStr.c_str(),
            (int32_t)promptStr.size(),
            prompt_tokens.data(),
            (int32_t)prompt_tokens.size(),
            !has_bos, // add_bos only if not already present
            false   // special
        );

        if (n_tokens < 0) {
            ALOGE("completionStart(): llama_tokenize failed");
            jstring err = threadEnv->NewStringUTF("Tokenization failed");
            if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
                threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                if (threadEnv->ExceptionCheck()) {
                    ALOGE("completionStart(): Exception in error callback - clearing");
                    threadEnv->ExceptionClear();
                }
            } else {
                ALOGE("completionStart(): Callback validation failed - skipping error callback");
            }
            threadEnv->DeleteLocalRef(err);
            detachThread();
            return;
        }
        prompt_tokens.resize(n_tokens);
        ALOGD("completionStart(): tokenized prompt into %d tokens", n_tokens);
        // Log first and last few tokens for debugging
        if (n_tokens > 0) {
            ALOGD("completionStart(): First 5 tokens: %d %d %d %d %d", 
                  (int)prompt_tokens[0], 
                  n_tokens > 1 ? (int)prompt_tokens[1] : -1,
                  n_tokens > 2 ? (int)prompt_tokens[2] : -1,
                  n_tokens > 3 ? (int)prompt_tokens[3] : -1,
                  n_tokens > 4 ? (int)prompt_tokens[4] : -1);
            if (n_tokens >= 5) {
                ALOGD("completionStart(): Last 5 tokens: %d %d %d %d %d", 
                      (int)prompt_tokens[n_tokens-5], 
                      (int)prompt_tokens[n_tokens-4],
                      (int)prompt_tokens[n_tokens-3],
                      (int)prompt_tokens[n_tokens-2],
                      (int)prompt_tokens[n_tokens-1]);
            }
            // Log actual token text for first and last tokens to verify prompt
            char first_token_text[256];
            int first_len = llama_token_to_piece(vocab, prompt_tokens[0], first_token_text, sizeof(first_token_text), false, false);
            if (first_len > 0 && first_len < 256) {
                first_token_text[first_len] = '\0';
                ALOGD("completionStart(): First token text: '%s' (id=%d)", first_token_text, (int)prompt_tokens[0]);
            }
            char last_token_text[256];
            int last_len = llama_token_to_piece(vocab, prompt_tokens[n_tokens-1], last_token_text, sizeof(last_token_text), false, false);
            if (last_len > 0 && last_len < 256) {
                last_token_text[last_len] = '\0';
                ALOGD("completionStart(): Last token text: '%s' (id=%d)", last_token_text, (int)prompt_tokens[n_tokens-1]);
            }
        }

        // Create sampler chain matching iOS implementation (Llama 3.1 optimized)
        auto sparams = llama_sampler_chain_default_params();
        llama_sampler * smpl = llama_sampler_chain_init(sparams);
        
        // Llama 3.1 optimized sampling chain (matching iOS implementation)
        // Order is critical: Top-P -> Min-P -> Temperature -> Repeat Penalty -> Dist
        
        // 1. Top-K (0 = disabled, Llama 3.1 recommendation: use Top-P + Min-P instead)
        // Top-K is disabled for Llama 3.1 as it works better with Top-P + Min-P combination
        if (top_k > 0) {
            ALOGD("completionStart(): WARNING: Top-K is enabled (%d) but Llama 3.1 recommends Top-K=0 (use Top-P + Min-P)", top_k);
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        }
        
        // 2. Top-P (Nucleus Sampling) - Llama 3.1 recommended: 0.9
        // Keeps tokens with cumulative probability up to top_p
        if (top_p > 0.0f && top_p < 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        } else {
            ALOGD("completionStart(): WARNING: Top-P is disabled or invalid (%.2f), using default 0.9", top_p);
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
        }
        
        // 3. Min-P (Llama 3.1 key setting - exclude low probability tokens)
        // Removes tokens with probability less than min_p * max_probability
        // This is critical for Llama 3.1 quality
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
        
        // 4. Temperature - Llama 3.1 recommended: 0.6
        // Lower temperature = more deterministic, less repetition
        if (temp > 0.0f && temp != 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        } else {
            ALOGD("completionStart(): WARNING: Temperature is disabled or invalid (%.2f), using default 0.6", temp);
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.6f));
        }
        
        // 5. Repeat Penalty (with freq_penalty and presence_penalty)
        // Llama 3.1 recommended: repeat_penalty=1.15, last_n=64, freq_penalty=0.1, presence_penalty=0.1
        if (rep_last_n != 0 && rep_penalty > 0.0f && rep_penalty != 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(rep_last_n, rep_penalty, 0.1f, 0.1f));
        } else {
            ALOGD("completionStart(): WARNING: Repeat penalty is disabled or invalid, using defaults");
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.15f, 0.1f, 0.1f));
        }
        
        // 6. Dist sampling (final token selection)
        // Random seed for diversity
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(static_cast<uint32_t>(std::random_device{}())));
        
        ALOGD("completionStart(): Sampling chain configured: Top-K=%d, Top-P=%.2f, Min-P=0.05, Temp=%.2f, Repeat=%d/%.2f",
              top_k, top_p, temp, rep_last_n, rep_penalty);
        ALOGD("completionStart(): Sampling parameters: top_k=%d, top_p=%.3f, temp=%.3f, rep_penalty=%.3f, rep_last_n=%d",
              top_k, top_p, temp, rep_penalty, rep_last_n);

        // Main generation loop - evaluate prompt and generate tokens in streaming mode
        int n_past = 0;
        int n_gen = 0;
        std::string generated;

        // Decode prompt in chunks - conservative chunk size for stability
        ALOGD("completionStart(): evaluating prompt...");
        // Use conservative chunk size (64) to avoid assertion failures
        // Large chunks (128+) can cause assertion failures in llama_decode
        const int chunk = 64; // Conservative size for stability
        uint32_t context_size = llama_n_ctx(ctx);
        
        // Evaluate prompt tokens and generate tokens in streaming mode
        // Optimize chunking for speed while maintaining stability
        for (int cur = 0; cur < n_tokens; ) {
            int remaining = n_tokens - cur;
            int n_cur = std::min(chunk, remaining);
            
            // For the last chunk, if it's small (< 32), merge with previous chunk for efficiency
            bool is_last_chunk = (cur + n_cur == n_tokens);
            if (is_last_chunk && n_cur < 32 && cur > 0) {
                // Merge small last chunk with previous chunk for better efficiency
                ALOGD("completionStart(): Last chunk size %d is small, will merge with previous", n_cur);
                // Process remaining tokens in current iteration
                n_cur = remaining;
            }
            
            // Limit maximum chunk size to 64 for stability
            // Large chunks can cause assertion failures in llama_decode
            if (n_cur > 64) {
                ALOGD("completionStart(): Chunk size %d exceeds maximum 64, limiting to 64", n_cur);
                n_cur = 64;
            }
            
            ALOGD("completionStart(): llama_decode chunk start cur=%d n_cur=%d (remaining after=%d)", cur, n_cur, n_tokens - cur - n_cur);
            
            // Initialize batch with proper sequence ID support
            llama_batch batch = llama_batch_init(n_cur, 0, 1);
            if (!batch.token || !batch.seq_id || !batch.n_seq_id || !batch.logits) {
                ALOGE("completionStart(): llama_batch_init() returned invalid batch");
                jstring err = threadEnv->NewStringUTF("Failed to initialize batch");
                if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
                    threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                    if (threadEnv->ExceptionCheck()) {
                        ALOGE("completionStart(): Exception in error callback - clearing");
                        threadEnv->ExceptionClear();
                    }
                } else {
                    ALOGE("completionStart(): Callback validation failed - skipping error callback");
                }
                threadEnv->DeleteLocalRef(err);
                llama_batch_free(batch);
                llama_sampler_free(smpl);
                // Prevent CallbackGuard from trying to delete gCallback after detachThread()
                if (guard.env && guard.ref) {
                    guard.env->DeleteGlobalRef(guard.ref);
                    guard.ref = nullptr;
                }
                guard.shouldDelete = false;
                detachThread();
                return;
            }
            
            batch.n_tokens = n_cur;
            // Set pos to nullptr to let llama_batch_allocr::init() calculate positions from memory
            // This ensures positions are consistent with the KV cache state
            free(batch.pos);
            batch.pos = nullptr;
            
            ALOGD("completionStart(): Batch initialized, filling tokens (n_past=%d)", n_past);
            for (int j = 0; j < n_cur; ++j) {
                batch.token   [j] = prompt_tokens[cur + j];
                if (batch.seq_id[j]) {
                    batch.seq_id  [j][0] = 0;
                } else {
                    ALOGE("completionStart(): batch.seq_id[%d] is null!", j);
                    llama_batch_free(batch);
                    llama_sampler_free(smpl);
                    detachThread();
                    return;
                }
                batch.n_seq_id[j] = 1;
                batch.logits  [j] = false;
            }
            // NOTE: Do NOT enable logits in the last chunk during prompt evaluation
            // This causes decode to hang on Vulkan backend. Instead, we'll decode the last token
            // separately after prompt evaluation is complete to get logits.
            // is_last_chunk is already defined above
            // Keep all logits disabled during prompt evaluation to avoid Vulkan backend issues
            // We'll handle logits separately after prompt evaluation
            
            ALOGD("completionStart(): Batch filled, calling llama_decode() for chunk cur=%d n_cur=%d (total tokens=%d, is_last_chunk=%d)", 
                  cur, n_cur, n_tokens, is_last_chunk ? 1 : 0);
            ALOGD("completionStart(): About to call llama_decode() for prompt evaluation chunk cur=%d n_cur=%d", cur, n_cur);
            
            int decode_result = llama_decode(ctx, batch);
            ALOGD("completionStart(): llama_decode() returned %d for chunk cur=%d n_cur=%d (is_last_chunk=%d)", 
                  decode_result, cur, n_cur, is_last_chunk ? 1 : 0);
            if (decode_result != 0) {
                ALOGE("completionStart(): llama_decode() failed at chunk cur=%d n_cur=%d", cur, n_cur);
                jstring err = threadEnv->NewStringUTF("Failed to decode prompt");
                if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
                    threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                    if (threadEnv->ExceptionCheck()) {
                        ALOGE("completionStart(): Exception in error callback - clearing");
                        threadEnv->ExceptionClear();
                    }
                } else {
                    ALOGE("completionStart(): Callback validation failed - skipping error callback");
                }
                threadEnv->DeleteLocalRef(err);
                llama_batch_free(batch);
                llama_sampler_free(smpl);
                // Prevent CallbackGuard from trying to delete gCallback after detachThread()
                // We need to delete it manually before detaching
                if (guard.env && guard.ref) {
                    guard.env->DeleteGlobalRef(guard.ref);
                    guard.ref = nullptr;
                }
                guard.shouldDelete = false;  // Prevent double deletion
                detachThread();
                return;
            }
            llama_batch_free(batch);
            n_past += n_cur;
            ALOGD("completionStart(): llama_decode chunk ok cur=%d n_cur=%d, n_past=%d (total tokens=%d, remaining=%d)", 
                  cur, n_cur, n_past, n_tokens, n_tokens - (cur + n_cur));
            
            // Move to next chunk
            cur += n_cur;
        }
        
        ALOGD("completionStart(): Exited prompt evaluation loop. n_past=%d, n_tokens=%d", n_past, n_tokens);
        
        // After prompt evaluation, decode the last token separately to get logits
        // This avoids the Vulkan backend hang when enabling logits in the last chunk
        if (n_past == n_tokens && n_tokens > 0) {
            ALOGD("completionStart(): Decoding last prompt token separately to get logits...");
            llama_batch last_token_batch = llama_batch_init(1, 0, 1);
            if (last_token_batch.token && last_token_batch.seq_id) {
                last_token_batch.n_tokens = 1;
                last_token_batch.token[0] = prompt_tokens[n_tokens - 1];
                last_token_batch.logits[0] = true;  // Enable logits for the last token
                free(last_token_batch.pos);
                last_token_batch.pos = nullptr;  // Let llama_batch_allocr calculate position
                if (last_token_batch.seq_id[0]) {
                    last_token_batch.seq_id[0][0] = 0;
                    last_token_batch.n_seq_id[0] = 1;
                }
                
                ALOGD("completionStart(): Calling llama_decode() for last token to get logits");
                int last_decode_result = llama_decode(ctx, last_token_batch);
                ALOGD("completionStart(): llama_decode() returned %d for last token", last_decode_result);
                if (last_decode_result != 0) {
                    ALOGE("completionStart(): Failed to decode last token for logits, result=%d", last_decode_result);
                } else {
                    // Verify logits are available after decoding last token
                    const float* last_logits = llama_get_logits_ith(ctx, 0);
                    if (last_logits) {
                        ALOGD("completionStart(): Logits available after last token decode (idx=0)");
                        // Log top 3 logits for verification
                        const llama_model* model = llama_get_model(ctx);
                        const llama_vocab* vocab = llama_model_get_vocab(model);
                        const int n_vocab = llama_vocab_n_tokens(vocab);
                        std::vector<std::pair<float, llama_token>> candidates;
                        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                            candidates.push_back({last_logits[token_id], token_id});
                        }
                        std::sort(candidates.begin(), candidates.end(), 
                            [](const std::pair<float, llama_token>& a, const std::pair<float, llama_token>& b) {
                                return a.first > b.first;
                            });
                        ALOGD("completionStart(): Top 3 logits after last prompt token decode:");
                        for (int i = 0; i < 3 && i < (int)candidates.size(); i++) {
                            char token_text[256];
                            int n_len = llama_token_to_piece(vocab, candidates[i].second, token_text, sizeof(token_text), false, false);
                            token_text[n_len] = '\0';
                            ALOGD("completionStart():   [%d] id=%d, logit=%.3f, text='%s'", 
                                  i, (int)candidates[i].second, candidates[i].first, token_text);
                        }
                    } else {
                        ALOGE("completionStart(): Logits NOT available after last token decode (idx=0)");
                    }
                }
                llama_batch_free(last_token_batch);
            } else {
                ALOGE("completionStart(): Failed to initialize batch for last token");
            }
        }
        
        ALOGD("completionStart(): Prompt evaluation complete. n_past=%d, starting token generation...", n_past);
        
        // Continue generating tokens until limit is reached
        // Note: n_gen starts at 0, so we generate tokens 0..(n_predict-1) = n_predict tokens total
        // But we check n_gen < n_predict at the start of loop, so after generating n_predict tokens, n_gen will be n_predict and loop will exit
        while (n_past < context_size && n_gen < n_predict) {
            // Check limit before generating token to ensure we don't exceed n_predict
            if (n_gen >= n_predict) {
                ALOGD("completionStart(): Reached token limit (n_gen=%d >= n_predict=%d), breaking", n_gen, n_predict);
                break;
            }
            ALOGD("completionStart(): Loop iteration n_gen=%d, n_past=%d", n_gen, n_past);
            if (handle->stopRequested) {
                ALOGD("completionStart(): Stop requested, breaking");
                break;
            }

            // Sample from logits
            // For the first token generation, use idx=0 to get logits from the last decode (last prompt token)
            // For subsequent tokens, use idx=0 to get logits from the most recent decode
            int32_t logits_idx = 0;  // Changed from -1 to 0 since we decoded the last token separately
            ALOGD("completionStart(): Calling llama_sampler_sample() with idx=%d (n_gen=%d, n_past=%d)", logits_idx, n_gen, n_past);
            
            // Check if logits are available before sampling
            const float* logits_check = llama_get_logits_ith(ctx, logits_idx);
            if (!logits_check) {
                ALOGE("completionStart(): logits are null for idx=%d, cannot sample token", logits_idx);
                jstring err = threadEnv->NewStringUTF("Logits not available");
                if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
                    threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                    if (threadEnv->ExceptionCheck()) {
                        ALOGE("completionStart(): Exception in error callback - clearing");
                        threadEnv->ExceptionClear();
                    }
                }
                threadEnv->DeleteLocalRef(err);
                break;
            }
            
            // REMOVED: Korean token boosting - this was interfering with context understanding
            // The model should generate tokens based on context, not forced language preference
            // Let the system prompt and model's natural understanding guide token selection
            
            llama_token id = llama_sampler_sample(smpl, ctx, logits_idx);
            ALOGD("completionStart(): llama_sampler_sample() returned id=%d (n_gen=%d)", (int)id, n_gen);
            
            // Log top 5 candidate tokens for analysis (for debugging context understanding)
            if (logits_check) {
                const llama_model* model = llama_get_model(ctx);
                const llama_vocab* vocab = llama_model_get_vocab(model);
                const int n_vocab = llama_vocab_n_tokens(vocab);
                std::vector<std::pair<float, llama_token>> candidates;
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.push_back({logits_check[token_id], token_id});
                }
                std::sort(candidates.begin(), candidates.end(), 
                    [](const std::pair<float, llama_token>& a, const std::pair<float, llama_token>& b) {
                        return a.first > b.first;
                    });
                ALOGD("completionStart(): Top 5 candidate tokens (context-based):");
                for (int i = 0; i < 5 && i < (int)candidates.size(); i++) {
                    char token_text[256];
                    int n_len = llama_token_to_piece(vocab, candidates[i].second, token_text, sizeof(token_text), false, false);
                    token_text[n_len] = '\0';
                    ALOGD("completionStart():   [%d] id=%d, logit=%.3f, text='%s'", 
                          i, (int)candidates[i].second, candidates[i].first, token_text);
                }
            }
            llama_sampler_accept(smpl, id);
            ALOGD("completionStart(): llama_sampler_accept() completed");

            // Check for EOG tokens (as per iOS: 128001, 128008, 128009)
            if (id == llama_vocab_eos(vocab) || id == 128001 || id == 128008 || id == 128009) {
                ALOGD("completionStart(): EOG token detected, breaking");
                break;
            }
            
            // Special token handling (ID >= 128000): skip output but still process
            // Process special tokens through normal flow but skip text output
            bool isSpecialToken = (id >= 128000);
            ALOGD("completionStart(): Token id=%d, isSpecialToken=%d", (int)id, isSpecialToken ? 1 : 0);
            
            // Convert token to piece
            ALOGD("completionStart(): Calling llama_token_to_piece()");
            std::vector<char> piece(16, 0);
            int n_len = llama_token_to_piece(vocab, id, piece.data(), piece.size(), false, false);
            ALOGD("completionStart(): llama_token_to_piece() returned n_len=%d", n_len);
            if (n_len < 0) {
                ALOGE("completionStart(): llama_token_to_piece() failed");
                break;
            }
            if (static_cast<size_t>(n_len) >= piece.size()) {
                ALOGD("completionStart(): Resizing piece buffer from %zu to %d", piece.size(), n_len + 1);
                piece.resize(n_len + 1);
                n_len = llama_token_to_piece(vocab, id, piece.data(), piece.size(), false, false);
                if (n_len < 0) {
                    ALOGE("completionStart(): llama_token_to_piece() retry failed");
                    break;
                }
            }
            piece.resize(n_len);
            std::string tokenText(piece.begin(), piece.end());
            ALOGD("completionStart(): Token text length=%zu", tokenText.length());
            
            // Only send non-special tokens to callback
            if (!isSpecialToken && !tokenText.empty()) {
                ALOGD("completionStart(): Token text='%s' (length=%zu), creating JNI string and calling callback", 
                      tokenText.c_str(), tokenText.length());
                
                // Convert UTF-8 to JNI string safely
                // NewStringUTF requires Modified UTF-8 which can fail for invalid UTF-8 bytes
                // Use byte array method which handles any UTF-8 bytes safely
                jstring tk = nullptr;
                
                // Create byte array from UTF-8 bytes and convert to String via Java
                jbyteArray byteArray = threadEnv->NewByteArray(tokenText.length());
                if (byteArray) {
                    threadEnv->SetByteArrayRegion(byteArray, 0, tokenText.length(), 
                                                  reinterpret_cast<const jbyte*>(tokenText.data()));
                    // Call Java method to convert byte array to String using UTF-8 charset
                    jclass stringClass = threadEnv->FindClass("java/lang/String");
                    if (stringClass) {
                        jmethodID stringCtor = threadEnv->GetMethodID(stringClass, "<init>", "([BLjava/lang/String;)V");
                        if (stringCtor) {
                            jstring charsetName = threadEnv->NewStringUTF("UTF-8");
                            if (charsetName) {
                                tk = (jstring)threadEnv->NewObject(stringClass, stringCtor, byteArray, charsetName);
                                if (threadEnv->ExceptionCheck()) {
                                    ALOGE("completionStart(): Exception creating String from byte array");
                                    threadEnv->ExceptionClear();
                                    tk = nullptr;
                                }
                                threadEnv->DeleteLocalRef(charsetName);
                            }
                        }
                        threadEnv->DeleteLocalRef(stringClass);
                    }
                    threadEnv->DeleteLocalRef(byteArray);
                }
                
                if (tk) {
                    // For Kotlin interfaces, each implementation is a different anonymous class.
                    // We need to get the method ID from the actual callback object's class.
                    if (gCallback && threadEnv) {
                        // Get method ID from the actual callback object's class
                        jclass callbackClass = nullptr;
                        jmethodID onTokenMethod = nullptr;
                        bool success = false;
                        
                        callbackClass = threadEnv->GetObjectClass(gCallback);
                        if (callbackClass) {
                            onTokenMethod = threadEnv->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");
                            if (onTokenMethod) {
                                ALOGD("completionStart(): Calling onToken callback with token='%s'", tokenText.c_str());
                                threadEnv->CallVoidMethod(gCallback, onTokenMethod, tk);
                                if (threadEnv->ExceptionCheck()) {
                                    ALOGE("completionStart(): Exception in token callback - clearing");
                                    threadEnv->ExceptionClear();
                                } else {
                                    ALOGD("completionStart(): Token callback completed successfully");
                                    success = true;
                                }
                            } else {
                                ALOGE("completionStart(): Failed to get onToken method ID from callback class");
                            }
                        } else {
                            ALOGE("completionStart(): Failed to get callback object class");
                        }
                        
                        // Clean up local references safely
                        if (callbackClass && threadEnv) {
                            threadEnv->DeleteLocalRef(callbackClass);
                        }
                        
                        if (!success) {
                            ALOGE("completionStart(): Token callback failed");
                        }
                    } else {
                        if (!gCallback) {
                            ALOGE("completionStart(): gCallback is null - skipping token callback");
                        }
                        if (!threadEnv) {
                            ALOGE("completionStart(): threadEnv is null - skipping token callback");
                        }
                    }
                    if (threadEnv && tk) {
                        threadEnv->DeleteLocalRef(tk);
                    }
                } else {
                    ALOGE("completionStart(): Failed to create JNI string");
                }
                generated.append(tokenText);
            } else {
                ALOGD("completionStart(): Skipping token (isSpecialToken=%d, empty=%d)", 
                      isSpecialToken ? 1 : 0, tokenText.empty() ? 1 : 0);
            }
            bool hitStop = false;
            for (const auto& stop : stops) {
                if (!stop.empty() && generated.size() >= stop.size()) {
                    if (generated.compare(generated.size() - stop.size(), stop.size(), stop) == 0) {
                        hitStop = true;
                        break;
                    }
                }
            }

            n_gen++;
            ALOGD("completionStart(): Incremented n_gen to %d", n_gen);
            
            // Check if we've reached the token limit after incrementing
            if (n_gen >= n_predict) {
                ALOGD("completionStart(): Reached token limit (n_gen=%d >= n_predict=%d), breaking before decode", n_gen, n_predict);
                break;
            }

            // Prepare and run next decode with a single token
            ALOGD("completionStart(): Initializing llama_batch");
            llama_batch gen_batch = llama_batch_init(1, 0, 1);
            if (!gen_batch.token || !gen_batch.pos || !gen_batch.seq_id || !gen_batch.n_seq_id || !gen_batch.logits) {
                ALOGE("completionStart(): llama_batch_init() returned invalid gen_batch");
                jstring err = threadEnv->NewStringUTF("Failed to initialize generation batch");
                if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
                    threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                    if (threadEnv->ExceptionCheck()) {
                        ALOGE("completionStart(): Exception in error callback - clearing");
                        threadEnv->ExceptionClear();
                    }
                } else {
                    ALOGE("completionStart(): Callback validation failed - skipping error callback");
                }
                threadEnv->DeleteLocalRef(err);
                llama_batch_free(gen_batch);
                break;
            }
            gen_batch.n_tokens = 1;
            gen_batch.token   [0] = id;
            // For token generation, set pos to nullptr to let llama_batch_allocr::init() calculate position from memory
            // This ensures positions are consistent with the KV cache state, especially after separate last token decode
            free(gen_batch.pos);
            gen_batch.pos = nullptr;  // Let llama_batch_allocr calculate position from memory
            if (gen_batch.seq_id[0]) {
                gen_batch.seq_id  [0][0] = 0;
            } else {
                ALOGE("completionStart(): gen_batch.seq_id[0] is null!");
                llama_batch_free(gen_batch);
                break;
            }
            gen_batch.n_seq_id[0] = 1;
            gen_batch.logits  [0] = true;
            ALOGD("completionStart(): Batch initialized, calling llama_decode() with token=%d, pos=auto (n_past=%d)", (int)id, n_past);
            ALOGD("completionStart(): About to call llama_decode() for token generation, n_past=%d, n_gen=%d", n_past, n_gen);

            int decode_result = llama_decode(ctx, gen_batch);
            ALOGD("completionStart(): llama_decode() returned %d for token generation", decode_result);
            if (decode_result != 0) {
                ALOGE("completionStart(): llama_decode() failed on token, result=%d", decode_result);
                jstring err = threadEnv->NewStringUTF("Failed to decode token");
                if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
                    threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                    if (threadEnv->ExceptionCheck()) {
                        ALOGE("completionStart(): Exception in error callback - clearing");
                        threadEnv->ExceptionClear();
                    }
                } else {
                    ALOGE("completionStart(): Callback validation failed - skipping error callback");
                }
                threadEnv->DeleteLocalRef(err);
                llama_batch_free(gen_batch);
                break;
            }
            ALOGD("completionStart(): Freeing batch");
            llama_batch_free(gen_batch);
            n_past++;
            ALOGD("completionStart(): Incremented n_past to %d", n_past);

            if (hitStop) {
                break;
            }
        }

        llama_sampler_free(smpl);

        // Call completed callback - dynamically get method ID from actual callback object's class
        // This is necessary because Kotlin interfaces are implemented as anonymous classes
        if (gCallback && threadEnv) {
            jclass callbackClass = nullptr;
            jmethodID onCompletedMethod = nullptr;
            bool success = false;
            
            callbackClass = threadEnv->GetObjectClass(gCallback);
            if (callbackClass) {
                onCompletedMethod = threadEnv->GetMethodID(callbackClass, "onCompleted", "()V");
                if (onCompletedMethod) {
                    ALOGD("completionStart(): Calling onCompleted callback");
                    threadEnv->CallVoidMethod(gCallback, onCompletedMethod);
                    if (threadEnv->ExceptionCheck()) {
                        ALOGE("completionStart(): Exception in completed callback - clearing");
                        threadEnv->ExceptionClear();
                    } else {
                        ALOGD("completionStart(): onCompleted callback completed successfully");
                        success = true;
                    }
                } else {
                    ALOGE("completionStart(): Failed to get onCompleted method ID from callback class");
                }
            } else {
                ALOGE("completionStart(): Failed to get callback object class for onCompleted");
            }
            
            // Clean up local references safely
            if (callbackClass && threadEnv) {
                threadEnv->DeleteLocalRef(callbackClass);
            }
            
            if (!success) {
                ALOGE("completionStart(): onCompleted callback failed");
            }
        } else {
            if (!gCallback) {
                ALOGE("completionStart(): gCallback is null - skipping onCompleted callback");
            }
            if (!threadEnv) {
                ALOGE("completionStart(): threadEnv is null - skipping onCompleted callback");
            }
        }
		detachThread();
    });
    worker.detach();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_saveSession(
        JNIEnv* env, jobject /*thiz*/, jlong h, jstring jPath) {
    auto* handle = reinterpret_cast<LlamaCtx*>(h);
    if (!handle) return -1;
    const char* path = env->GetStringUTFChars(jPath, nullptr);
    int result = -3;
#if LLAMA_STUB_MODE
    result = 0;
#else
    if (handle->ctx && path) {
        const size_t stateSize = llama_state_get_size(handle->ctx);
        if (stateSize == 0) {
            result = -4;
        } else {
            std::vector<uint8_t> buffer(stateSize);
            const size_t written = llama_state_get_data(handle->ctx, buffer.data(), buffer.size());
            if (written != buffer.size()) {
                result = -5;
            } else {
                FILE* fp = fopen(path, "wb");
                if (!fp) {
                    result = -6;
                } else {
                    const size_t out = fwrite(buffer.data(), 1, buffer.size(), fp);
                    fclose(fp);
                    result = (out == buffer.size()) ? static_cast<int>(buffer.size()) : -7;
                }
            }
        }
    } else {
        result = -2;
    }
#endif
    env->ReleaseStringUTFChars(jPath, path);
    return result;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_loadSession(
        JNIEnv* env, jobject /*thiz*/, jlong h, jstring jPath) {
    auto* handle = reinterpret_cast<LlamaCtx*>(h);
    if (!handle) return JNI_FALSE;
    const char* path = env->GetStringUTFChars(jPath, nullptr);
    bool ok = false;
#if LLAMA_STUB_MODE
    ok = true;
#else
    if (handle->ctx && path) {
        FILE* fp = fopen(path, "rb");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            const long len = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            if (len > 0) {
                std::vector<uint8_t> buffer(static_cast<size_t>(len));
                const size_t read = fread(buffer.data(), 1, buffer.size(), fp);
                if (read == buffer.size()) {
                    const size_t applied = llama_state_set_data(handle->ctx, buffer.data(), buffer.size());
                    ok = (applied == buffer.size());
                }
            }
            fclose(fp);
        }
    }
#endif
    env->ReleaseStringUTFChars(jPath, path);
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_tokenize(
        JNIEnv* env, jobject /*thiz*/, jlong h, jstring jText) {
    auto* handle = reinterpret_cast<LlamaCtx*>(h);
    if (!handle) return env->NewIntArray(0);
    const char* text = env->GetStringUTFChars(jText, nullptr);
    std::vector<int> out;
#if LLAMA_STUB_MODE
    // Return codepoint count as fake token ids (not meaningful)
    if (text) {
        for (const char* p = text; *p; ++p) {
            out.push_back(static_cast<unsigned char>(*p));
        }
    }
#else
    if (handle->model && text) {
        const llama_vocab* vocab = llama_model_get_vocab(handle->model);
        std::vector<llama_token> toks;
        toks.resize(strlen(text) + 16);
        int n = llama_tokenize(llama_model_get_vocab(handle->model), text, (int)strlen(text), toks.data(), (int)toks.size(), true, false);
        if (n > 0) {
            toks.resize(n);
            out.reserve(n);
            for (int i = 0; i < n; ++i) out.push_back((int)toks[i]);
        }
    }
#endif
    env->ReleaseStringUTFChars(jText, text);
    jintArray arr = env->NewIntArray((jsize)out.size());
    if (!out.empty()) {
        env->SetIntArrayRegion(arr, 0, (jsize)out.size(), reinterpret_cast<const jint*>(out.data()));
    }
    return arr;
}


