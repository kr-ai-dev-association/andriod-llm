package com.example.llama.data

import android.app.Application
import android.util.Log
import com.example.llama.nativebridge.LlamaBridge
import com.example.llama.nativebridge.TokenCallback
import com.example.llama.ui.model.ChatMessage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException

class ChatRepository(private val app: Application) {

	@Volatile
	private var handle: Long = 0L

	suspend fun preload(callback: TokenCallback) = withContext(Dispatchers.Default) {
		// Explicit init without sending any prompt, so UI can wait until load completes
		// Run on background thread to avoid blocking UI
		ensureInit(callback)
	}

	fun reset() {
		Log.d("BanyaChat", "ChatRepository.reset(): resetting handle from $handle to 0")
		handle = 0L
	}

	fun isInitialized(): Boolean {
		return handle != 0L
	}

	fun getHandle(): Long {
		return handle
	}

	fun stop() {
		if (handle != 0L) {
			try {
				LlamaBridge.completionStop(handle)
			} catch (_: Throwable) {
			}
		}
	}

	private fun ensureInit(callback: TokenCallback) {
		if (handle != 0L) return
		Log.d("BanyaChat", "ensureInit(): start")
		// Notify UI that model loading is starting
		callback.onLoadProgress(0)
		// Resolve model path: prefer internal, else external app-specific storage
		// Updated to use Q4_0 model instead of Q4_K_M (Q4_0 has better Vulkan compatibility on Adreno 830)
		val internal = File(app.filesDir, "models/llama31-banyaa-q4_0.gguf")
		val externalBase = app.getExternalFilesDir(null)
		val external = if (externalBase != null) File(externalBase, "models/llama31-banyaa-q4_0.gguf") else null

		// User override path (absolute). Must be a real file path the app can read.
		val overridePath = ModelPathStore.getOverridePath(app).takeIf { !it.isNullOrBlank() }
		val overrideFile = overridePath?.let { File(it) }
		
		// Note: Download directory access is restricted on Android 10+ (API 29+) due to Scoped Storage
		// Users should use the file picker to select model files, or place them in app-specific directories
		
		Log.d(
			"BanyaChat",
			"ensureInit(): override=$overridePath exists=${overrideFile?.exists()} size=${overrideFile?.length()} internalExists=${internal.exists()} internalSize=${internal.length()} external=${external?.absolutePath} externalExists=${external?.exists()} externalSize=${external?.length()}"
		)

		// Ensure internal dir and attempt asset copy if present (may fail due to space)
		if (!internal.exists()) {
			ModelFiles.ensureModelDir(app)
			try {
				ModelFiles.copyAssetIfPresent(
					context = app,
					assetName = "llama31-banyaa-q4_0.gguf",
					outName = "llama31-banyaa-q4_0.gguf"
				)
			} catch (_: IOException) {
			}
		}

		val modelFile = when {
			overrideFile != null && overrideFile.exists() -> overrideFile
			internal.exists() -> internal
			external != null && external.exists() -> external
			else -> internal // fall back; init will fail and stub will be used
		}
		val modelPath = modelFile.absolutePath
		Log.d("BanyaChat", "ensureInit(): chosen modelPath=$modelPath size=${modelFile.length()}")
		// Using Vulkan backend with Q4_0 model
		// Optimized settings to increase GPU offloading:
		// - Increased GPU layers to 20 to offload more layers to GPU
		// - Reduced batch size to 16 to minimize memory pressure
		// - Increased context size to 512 to accommodate longer prompts (289 tokens)
		handle = LlamaBridge.init(
			modelPath = modelPath,
			nCtx = 512, // Increased from 256 to accommodate longer prompts
			nThreads = 8,
			nBatch = 64, // Optimized: 64 provides best performance (128 was slower)
			nGpuLayers = 31, // Re-enabled for detailed crash analysis
			useMmap = false, // Disable mmap to avoid conflicts with Vulkan GPU offloading
			useMlock = false,
			seed = 0,
			callback = callback
		)
		Log.d("BanyaChat", "ensureInit(): LlamaBridge.init handle=$handle")
		// Don't call callback from here to avoid JNI callback conflicts
		// ChatViewModel will detect model loading completion by checking handle != 0
	}

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
		if (handle == 0L) {
			Log.w("BanyaChat", "generateStream(): handle=0 -> using STUB fallback")
			// Stub fallback for first run without native init success
			runStubStream(prompt, callback)
			return@withContext
		}
		try {
		LlamaBridge.completionStart(
			handle = handle,
			prompt = prompt,
			numPredict = 100,  // 최대 생성 토큰 수를 100으로 설정
			temperature = 0.7f,  // 반복 감소를 위해 다양성 증가 (0.6 → 0.7)
			topP = 0.9f,         // Llama 3.1 권장값: 0.9 (Top-P + Min-P 조합)
			topK = 0,            // Llama 3.1 권장: Top-K 비활성화 (Top-P + Min-P 사용)
			repeatPenalty = 1.2f,  // 반복 방지 (1.2로 설정하여 반복 감소와 대화 품질의 균형 유지)
			repeatLastN = 128,   // 더 긴 범위에서 반복 체크 (64 → 128)
			stopSequences = arrayOf(
				// 신뢰도 높은 종료 패턴만 유지 (문장 중간에 나올 수 있는 "요.", "죠." 등 제거)
				// 레벨 1: 가장 안전하고 필수적인 종료 패턴
				"습니다.", "니다.",
				// 질문형 패턴 (문장 끝에만 나타남)
				"까요?", "가요?", "나요?",
				// 감탄형 패턴 (문장 끝에만 나타남)
				"네요!", "군요!",
				// 레벨 4: 공격적이지만 목록을 보호하는 패턴
				".\n\n",
				// 기타 정지 시퀀스
				"사용자:", "질문:", "<|eot_id|>", "eotend_header", "<eotend_header>"
			),
			callback = callback
		)
			Log.d("BanyaChat", "generateStream(): completionStart dispatched")
		} catch (t: Throwable) {
			Log.e("BanyaChat", "generateStream(): completionStart error -> stub fallback: ${t.message}", t)
			runStubStream(prompt, callback)
		}
	}

	private fun formatPrompt(messages: List<ChatMessage>): String {
			// Simplified system prompt for better context understanding
		// Too long and complex prompts can confuse the model and lead to random token generation
		val systemPrompt = """너는 한국어로 대화하는 도움이 되는 AI 어시스턴트입니다. 사용자의 질문에 자연스럽고 의미 있는 한국어로 답변하세요. 절대 열거형(리스트, 번호 목록, 항목 나열 등)으로 대답하지 말고, 항상 자연스러운 문장으로 설명하세요."""
		val sb = StringBuilder()
		sb.append("<|begin_of_text|>")
		sb.append("<|start_header_id|>system<|end_header_id|>\n\n")
		sb.append(systemPrompt)
		sb.append("<|eot_id|>")

		messages.forEach {
			val role = if (it.isUser) "user" else "assistant"
			sb.append("<|start_header_id|>")
			sb.append(role)
			sb.append("<|end_header_id|>\n\n")
			sb.append(it.text)
			sb.append("<|eot_id|>")
		}

		sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
		return sb.toString()
	}

	fun saveSession(path: File): Int {
		return try {
			if (handle == 0L) -1 else LlamaBridge.saveSession(handle, path.absolutePath)
		} catch (_: Throwable) {
			-2
		}
	}

	fun loadSession(path: File): Boolean {
		return try {
			handle != 0L && LlamaBridge.loadSession(handle, path.absolutePath)
		} catch (_: Throwable) {
			false
		}
	}

	private suspend fun runStubStream(prompt: String, callback: TokenCallback) {
		callback.onLoadProgress(100)
		callback.onModelMetadata("""{"name":"stub","quantization":"stub","context_length":0,"size_label":"stub"}""")
		val fake = "이 모드는 스텁입니다. llama.cpp 연동 설정 후 실제 토큰이 스트리밍됩니다."
		for (ch in fake) {
			callback.onToken(ch.toString())
			delay(12)
		}
		callback.onCompleted()
	}
}


