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
        if (!ctx || !g_OnLoadProgress) {
            return true;
        }
        // If loading is already completed, don't call callback anymore
        if (ctx->completed.load()) {
            return true;
        }
        // If callback is null, skip
        if (!ctx->callback) {
            return true;
        }
        // Attach current thread to JVM (progress callback may run on different thread)
        JNIEnv* threadEnv = attachThread();
        if (!threadEnv) {
            ALOGE("progressFn(): Failed to attach thread");
            return true;
        }
        jint percent = static_cast<jint>(std::lround(progress * 100.0f));
        if (percent < 0) percent = 0;
        if (percent > 100) percent = 100;
        
        // Mark as completed when reaching 100%
        if (percent >= 100) {
            ctx->completed.store(true);
        }
        
        // Check if callback object is still valid and of correct type before calling
        // This prevents JNI errors when callback object has been replaced or GC'd
        if (threadEnv->IsSameObject(ctx->callback, nullptr)) {
            ALOGD("progressFn(): callback object is null, skipping");
            ctx->completed.store(true);
            return true;
        }
        
        // Verify that the callback object is an instance of the correct class
        // This prevents calling methods on wrong object types (e.g., ChatViewModel$generate$1$1 vs ChatViewModel$2$1)
        if (g_CallbackClass && !threadEnv->IsInstanceOf(ctx->callback, g_CallbackClass)) {
            ALOGE("progressFn(): callback object is not an instance of TokenCallback - disabling");
            ctx->callback = nullptr;
            ctx->completed.store(true);
            return true;
        }
        
        ALOGD("progressFn(): progress=%d%%", (int)percent);
        threadEnv->CallVoidMethod(ctx->callback, g_OnLoadProgress, percent);
        if (threadEnv->ExceptionCheck()) {
            ALOGE("progressFn(): Exception in callback - disabling callback and clearing exception");
            threadEnv->ExceptionClear();
            // Disable callback to prevent future calls
            ctx->callback = nullptr;
            ctx->completed.store(true);
        }
        // Note: We don't detach here as the thread may be reused
        return true;
    };

    const char* path = env->GetStringUTFChars(jModelPath, nullptr);
    ALOGD("init(): modelPath=%s nCtx=%d nThreads=%d nBatch=%d nGpuLayers=%d useMmap=%d useMlock=%d seed=%d",
          path ? path : "(null)", (int)nCtx, (int)nThreads, (int)nBatch, (int)nGpuLayers, (int)useMmap, (int)useMlock, (int)seed);

    llama_model_params mparams = llama_model_default_params();
    // Optimized Vulkan settings for Adreno 830 stability
    // Reduced GPU layers to minimize shader operations and avoid "Failed to link shaders" errors
    if (nGpuLayers == -1) {
        // Default to 5 layers for stability
        mparams.n_gpu_layers = 5;
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
            env->CallVoidMethod(callbackGlobal, g_OnError, err);
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
    cparams.n_ubatch = 2;  // Reduced from 4 to 2 for stability
    // Disable Flash Attention on Android as it may cause hangs
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        ALOGE("init(): llama_new_context_with_model failed");
        if (callbackGlobal && g_OnError) {
            jstring err = env->NewStringUTF("컨텍스트 초기화에 실패했습니다.");
            env->CallVoidMethod(callbackGlobal, g_OnError, err);
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
    if (callbackGlobal && g_OnLoadProgress) {
        env->CallVoidMethod(callbackGlobal, g_OnLoadProgress, 100);
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
        env->CallVoidMethod(callbackGlobal, g_OnModelMetadata, metaJson);
        env->DeleteLocalRef(metaJson);
    }

    handle->model = model;
    handle->ctx = ctx;
    ALOGD("init(): success, handle=%p", (void*)handle);

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
    float temp = (temperature > 0.0f) ? temperature : 0.3f;
    float top_p = (topP > 0.0f) ? topP : 0.85f;
    int top_k = (topK > 0) ? topK : 50;
    float rep_penalty = (repeatPenalty > 0.0f) ? repeatPenalty : 1.2f;
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
            ~CallbackGuard() {
                if (env && ref) {
                    env->DeleteGlobalRef(ref);
                }
            }
        } guard{threadEnv, gCallback};

        llama_context* ctx = handle->ctx;
        if (!ctx) {
            ALOGE("completionStart(): ctx is null");
            jstring err = threadEnv->NewStringUTF("Context is null");
            threadEnv->CallVoidMethod(gCallback, g_OnError, err);
            threadEnv->DeleteLocalRef(err);
            detachThread();
            return;
        }

        llama_model* model = handle->model;
        if (!model) {
            ALOGE("completionStart(): model is null");
            jstring err = threadEnv->NewStringUTF("Model is null");
            threadEnv->CallVoidMethod(gCallback, g_OnError, err);
            threadEnv->DeleteLocalRef(err);
            detachThread();
            return;
        }
        
        const struct llama_vocab * vocab = llama_model_get_vocab(model);

        // tokenize prompt
        std::vector<llama_token> prompt_tokens;
        prompt_tokens.resize(promptStr.size() + 16);
        ALOGD("completionStart(): tokenizing prompt...");
        int n_tokens = llama_tokenize(
            vocab,
            promptStr.c_str(),
            (int32_t)promptStr.size(),
            prompt_tokens.data(),
            (int32_t)prompt_tokens.size(),
            true, // add_bos
            false   // special
        );

        if (n_tokens < 0) {
            ALOGE("completionStart(): llama_tokenize failed");
            jstring err = threadEnv->NewStringUTF("Tokenization failed");
            threadEnv->CallVoidMethod(gCallback, g_OnError, err);
            threadEnv->DeleteLocalRef(err);
            detachThread();
            return;
        }
        prompt_tokens.resize(n_tokens);
        ALOGD("completionStart(): tokenized prompt into %d tokens", n_tokens);

        // Decode prompt in small chunks to reduce peak memory usage
        // Reduced chunk size to avoid Vulkan driver crashes
        ALOGD("completionStart(): evaluating prompt...");
        const int chunk = 32; // Reduced from 64 to 32 to reduce memory pressure
        for (int cur = 0; cur < n_tokens; cur += chunk) {
            const int n_cur = std::min(chunk, n_tokens - cur);
            ALOGD("completionStart(): llama_decode chunk start cur=%d n_cur=%d", cur, n_cur);
            
            // Initialize batch with proper sequence ID support
            llama_batch batch = llama_batch_init(n_cur, 0, 1);
            if (!batch.token || !batch.pos || !batch.seq_id || !batch.n_seq_id || !batch.logits) {
                ALOGE("completionStart(): llama_batch_init() returned invalid batch");
                jstring err = threadEnv->NewStringUTF("Failed to initialize batch");
                threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                threadEnv->DeleteLocalRef(err);
                llama_batch_free(batch);
                detachThread();
                return;
            }
            
            batch.n_tokens = n_cur;
            ALOGD("completionStart(): Batch initialized, filling tokens");
            for (int j = 0; j < n_cur; ++j) {
                batch.token   [j] = prompt_tokens[cur + j];
                batch.pos     [j] = cur + j;
                if (batch.seq_id[j]) {
                    batch.seq_id  [j][0] = 0;
                } else {
                    ALOGE("completionStart(): batch.seq_id[%d] is null!", j);
                    llama_batch_free(batch);
                    detachThread();
                    return;
                }
                batch.n_seq_id[j] = 1;
                batch.logits  [j] = false;
            }
            if (cur + n_cur == n_tokens) {
                batch.logits[n_cur - 1] = true;
            }
            ALOGD("completionStart(): Batch filled, calling llama_decode() for chunk cur=%d n_cur=%d", cur, n_cur);
            int decode_result = llama_decode(ctx, batch);
            ALOGD("completionStart(): llama_decode() returned %d for chunk cur=%d n_cur=%d", decode_result, cur, n_cur);
            if (decode_result != 0) {
                ALOGE("completionStart(): llama_decode() failed at chunk cur=%d n_cur=%d", cur, n_cur);
                jstring err = threadEnv->NewStringUTF("Failed to decode prompt");
                threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                threadEnv->DeleteLocalRef(err);
                llama_batch_free(batch);
                detachThread();
                return;
            }
            llama_batch_free(batch);
            ALOGD("completionStart(): llama_decode chunk ok cur=%d n_cur=%d", cur, n_cur);
            // Report decode progress (reuse onLoadProgress as a generic progress channel)
            if (g_OnLoadProgress) {
                const int done = std::min(cur + n_cur, n_tokens);
                const int percent = (n_tokens > 0) ? (done * 100) / n_tokens : 100;
                threadEnv->CallVoidMethod(gCallback, g_OnLoadProgress, (jint) percent);
                if (threadEnv->ExceptionCheck()) {
                    threadEnv->ExceptionClear();
                }
            }
        }
        ALOGD("completionStart(): evaluate prompt ok. starting sampling...");
        // Ensure UI progress reaches 100 before sampling
        if (g_OnLoadProgress) {
            threadEnv->CallVoidMethod(gCallback, g_OnLoadProgress, (jint) 100);
            if (threadEnv->ExceptionCheck()) {
                threadEnv->ExceptionClear();
            }
        }

        // Main generation loop
        int n_past = n_tokens;
        int n_gen = 0;
        std::string generated;

        // Create sampler chain matching iOS implementation (Llama 3.1 optimized)
        auto sparams = llama_sampler_chain_default_params();
        llama_sampler * smpl = llama_sampler_chain_init(sparams);
        
        // 1. Top-K (0 = disabled, as per iOS)
        if (top_k > 0) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        }
        
        // 2. Top-P (Nucleus Sampling)
        if (top_p > 0.0f && top_p < 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        }
        
        // 3. Min-P (Llama 3.1 key setting - exclude low probability tokens)
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
        
        // 4. Temperature
        if (temp > 0.0f && temp != 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        }
        
        // 5. Repeat Penalty (with freq_penalty and presence_penalty as per iOS)
        if (rep_last_n != 0 && rep_penalty > 0.0f && rep_penalty != 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(rep_last_n, rep_penalty, 0.1f, 0.1f));
        }
        
        // 6. Dist sampling (final token selection)
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(static_cast<uint32_t>(std::random_device{}())));

        uint32_t context_size = llama_n_ctx(ctx);
        ALOGD("completionStart(): Starting generation loop, context_size=%u, n_predict=%d, n_past=%d", context_size, n_predict, n_past);
        while (n_past < context_size && n_gen < n_predict) {
            ALOGD("completionStart(): Loop iteration n_gen=%d, n_past=%d", n_gen, n_past);
            if (handle->stopRequested) {
                ALOGD("completionStart(): Stop requested, breaking");
                break;
            }

            // Sample from logits
            ALOGD("completionStart(): Calling llama_sampler_sample()");
            llama_token id = llama_sampler_sample(smpl, ctx, 0);
            ALOGD("completionStart(): llama_sampler_sample() returned id=%d", (int)id);
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
                ALOGD("completionStart(): Creating JNI string and calling callback");
                jstring tk = threadEnv->NewStringUTF(tokenText.c_str());
                if (tk) {
                    threadEnv->CallVoidMethod(gCallback, g_OnToken, tk);
                    threadEnv->DeleteLocalRef(tk);
                    ALOGD("completionStart(): Callback completed");
                } else {
                    ALOGE("completionStart(): Failed to create JNI string");
                }
                generated.append(tokenText);
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

            // Prepare and run next decode with a single token
            ALOGD("completionStart(): Initializing llama_batch");
            llama_batch gen_batch = llama_batch_init(1, 0, 1);
            if (!gen_batch.token || !gen_batch.pos || !gen_batch.seq_id || !gen_batch.n_seq_id || !gen_batch.logits) {
                ALOGE("completionStart(): llama_batch_init() returned invalid gen_batch");
                jstring err = threadEnv->NewStringUTF("Failed to initialize generation batch");
                threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                threadEnv->DeleteLocalRef(err);
                llama_batch_free(gen_batch);
                break;
            }
            gen_batch.n_tokens = 1;
            gen_batch.token   [0] = id;
            gen_batch.pos     [0] = n_past;
            if (gen_batch.seq_id[0]) {
                gen_batch.seq_id  [0][0] = 0;
            } else {
                ALOGE("completionStart(): gen_batch.seq_id[0] is null!");
                llama_batch_free(gen_batch);
                break;
            }
            gen_batch.n_seq_id[0] = 1;
            gen_batch.logits  [0] = true;
            ALOGD("completionStart(): Batch initialized, calling llama_decode() with token=%d, pos=%d", (int)id, n_past);

            int decode_result = llama_decode(ctx, gen_batch);
            ALOGD("completionStart(): llama_decode() returned %d", decode_result);
            if (decode_result != 0) {
                ALOGE("completionStart(): llama_decode() failed on token, result=%d", decode_result);
                jstring err = threadEnv->NewStringUTF("Failed to decode token");
                threadEnv->CallVoidMethod(gCallback, g_OnError, err);
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

        threadEnv->CallVoidMethod(gCallback, g_OnCompleted);
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


