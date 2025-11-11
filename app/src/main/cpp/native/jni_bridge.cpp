#include <jni.h>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <android/log.h>

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

#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "BanyaChatJNI", __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, "BanyaChatJNI", __VA_ARGS__)

struct LlamaCtx {
#if LLAMA_STUB_MODE
    int dummy;
#else
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
#endif
    std::atomic<bool> stopRequested = false;
};

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* /*reserved*/) {
	g_JavaVM = vm;
	ALOGD("JNI_OnLoad: LLAMA_STUB_MODE=%d", LLAMA_STUB_MODE);
	return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_init(
        JNIEnv* env, jobject /*thiz*/,
        jstring jModelPath,
        jint nCtx, jint nThreads, jint nBatch,
        jboolean useMmap, jboolean useMlock, jint seed) {
    (void) nCtx; (void) nThreads; (void) nBatch; (void) useMmap; (void) useMlock; (void) seed;
    (void) env; (void) jModelPath;
    auto* handle = new LlamaCtx();
#if LLAMA_STUB_MODE
    ALOGD("init(): STUB build active. Returning dummy handle.");
    return reinterpret_cast<jlong>(handle);
#else
    const char* path = env->GetStringUTFChars(jModelPath, nullptr);
    ALOGD("init(): modelPath=%s nCtx=%d nThreads=%d nBatch=%d useMmap=%d useMlock=%d seed=%d",
          path ? path : "(null)", (int)nCtx, (int)nThreads, (int)nBatch, (int)useMmap, (int)useMlock, (int)seed);
    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = useMmap;
    mparams.use_mlock = useMlock;

    llama_model* model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(jModelPath, path);
    if (!model) {
        ALOGE("init(): llama_load_model_from_file failed");
        delete handle;
        return 0;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = nCtx;
    cparams.n_threads = nThreads;
    cparams.n_batch = nBatch;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        ALOGE("init(): llama_new_context_with_model failed");
        llama_model_free(model);
        delete handle;
        return 0;
    }
    handle->model = model;
    handle->ctx = ctx;
    ALOGD("init(): success, handle=%p", (void*)handle);
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
		JNIEnv* threadEnv = attachThread();
		if (!threadEnv) {
			// Cannot attach thread; give up
            ALOGE("completionStart(): failed to attach thread to JVM");
			return;
		}
#if LLAMA_STUB_MODE
        ALOGD("completionStart(): STUB mode streaming");
        // Emit a single valid UTF-8 Java string to avoid partial multibyte splits
        const char* msg = "스텁 네이티브 생성 중입니다. llama.cpp 연동 후 실제 토큰이 출력됩니다.";
        if (!handle->stopRequested) {
            jstring token = threadEnv->NewStringUTF(msg);
            threadEnv->CallVoidMethod(gCallback, g_OnToken, token);
            threadEnv->DeleteLocalRef(token);
        }
        threadEnv->CallVoidMethod(gCallback, g_OnCompleted);
#else
        llama_context* ctx = handle->ctx;
        llama_model* model = handle->model;
        const llama_vocab* vocab = llama_model_get_vocab(model);

        // tokenize prompt
        std::vector<llama_token> prompt_tokens;
        prompt_tokens.resize(promptStr.size() + 16);

        int n_tokens = llama_tokenize(
            vocab,
            promptStr.c_str(),
            (int)promptStr.length(),
            prompt_tokens.data(),
            (int)prompt_tokens.size(),
            true, // add BOS
            false // special tokens
        );

        if (n_tokens < 0) {
            ALOGE("completionStart(): llama_tokenize failed");
            jstring err = threadEnv->NewStringUTF("tokenize failed");
            threadEnv->CallVoidMethod(gCallback, g_OnError, err);
            threadEnv->DeleteLocalRef(err);
            threadEnv->DeleteGlobalRef(gCallback);
            detachThread();
            return;
        }
        prompt_tokens.resize(n_tokens);

        // eval prompt
        int n_past = 0;
        uint32_t context_size = llama_n_ctx(ctx);
        llama_batch batch = llama_batch_init(context_size, 0, 1);

        batch.n_tokens = (int32_t)prompt_tokens.size();
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            batch.token[i]  = prompt_tokens[i];
            batch.pos[i]    = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i] = (llama_seq_id*)malloc(sizeof(llama_seq_id));
            batch.seq_id[i][0] = 0;
            batch.logits[i] = false;
        }
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx, batch) != 0) {
             ALOGE("completionStart(): llama_decode prompt failed");
             jstring err = threadEnv->NewStringUTF("llama_decode prompt failed");
             threadEnv->CallVoidMethod(gCallback, g_OnError, err);
             threadEnv->DeleteLocalRef(err);
             threadEnv->DeleteGlobalRef(gCallback);
             llama_batch_free(batch);
             detachThread();
             return;
        }
        n_past += batch.n_tokens;


        std::string generated;
        for (int n = 0; n < n_predict && !handle->stopRequested; ++n) {
            // greedy sample from logits
            float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            
            llama_token token = 0;
            float max_p = -1.0f/0.0f;
            for (int32_t i = 0; i < n_vocab; ++i) {
                if (logits[i] > max_p) {
                    max_p = logits[i];
                    token = i;
                }
            }

            if (token == llama_vocab_eos(vocab)) {
                break;
            }

            // detokenize piece
            char piece_buf[64] = {0};
            int piece_len = llama_token_to_piece(vocab, token, piece_buf, sizeof(piece_buf), 0, false);
            std::string piece;
            if (piece_len > 0) {
                piece = std::string(piece_buf, piece_len);
            }

            if (piece.empty()) {
                ALOGD("completion loop: empty piece (token=%d)", token);
            }
            generated += piece;

            // emit token piece to Java
            if (!piece.empty()) {
                jstring tk = threadEnv->NewStringUTF(piece.c_str());
                threadEnv->CallVoidMethod(gCallback, g_OnToken, tk);
                threadEnv->DeleteLocalRef(tk);
            }

            // stop sequence check
            bool stopped = false;
            for (const auto& s : stops) {
                if (!s.empty() && generated.size() >= s.size()) {
                    if (generated.compare(generated.size() - s.size(), s.size(), s) == 0) {
                        // trim the stop suffix from UI by emitting a backspace-like no-op; UI will keep as-is.
                        stopped = true;
                        break;
                    }
                }
            }
            if (stopped) break;

            // feed back the sampled token
            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = n_past;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;

            if (llama_decode(ctx, batch) != 0) {
                ALOGE("completion loop: llama_decode failed at step %d", n);
                jstring err = threadEnv->NewStringUTF("llama_decode generation failed");
                threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                threadEnv->DeleteLocalRef(err);
                break;
            }
            n_past += 1;
        }
        for (int i=0; i<context_size; ++i) {
            if (batch.seq_id[i]) free(batch.seq_id[i]);
        }
        llama_batch_free(batch);

        threadEnv->CallVoidMethod(gCallback, g_OnCompleted);
#endif
        threadEnv->DeleteGlobalRef(gCallback);
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
    int result = 0;
#if LLAMA_STUB_MODE
    result = 0;
#else
    // NOTE: Upstream API can change. If unavailable, treat as no-op success.
    // Prefer llama_state_save_file if present.
    if (handle->ctx) {
        // Attempt to save KV/state. If symbol missing at link-time, this should be gated in real integration.
        // Placeholder always-success for portability.
        result = 0;
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
    bool ok = true;
#if LLAMA_STUB_MODE
    ok = true;
#else
    if (handle->ctx) {
        // Placeholder: mark true. Replace with llama_state_load_file when wired.
        ok = true;
    } else {
        ok = false;
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
        int n = llama_tokenize(vocab, text, (int)strlen(text), toks.data(), (int)toks.size(), true, false);
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


