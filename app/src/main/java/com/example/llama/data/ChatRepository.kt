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
		
		// Also check Download directory as fallback
		val downloadDir = android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOWNLOADS)
		val downloadFile = File(downloadDir, "llama31-banyaa-q4_0.gguf")
		
		Log.d(
			"BanyaChat",
			"ensureInit(): override=$overridePath exists=${overrideFile?.exists()} size=${overrideFile?.length()} internalExists=${internal.exists()} internalSize=${internal.length()} external=${external?.absolutePath} externalExists=${external?.exists()} externalSize=${external?.length()} downloadExists=${downloadFile.exists()} downloadSize=${downloadFile.length()}"
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
			downloadFile.exists() -> downloadFile  // Check Download directory as fallback
			else -> internal // fall back; init will fail and stub will be used
		}
		val modelPath = modelFile.absolutePath
		Log.d("BanyaChat", "ensureInit(): chosen modelPath=$modelPath size=${modelFile.length()}")
		// Using Vulkan backend with Q4_0 model
		// Optimized Vulkan settings for Adreno 830 stability:
		// - Reduced GPU layers to 5 to minimize shader operations
		// - Reduced batch size to 32 to reduce memory pressure
		// - no_host=true to use DEVICE_LOCAL memory for better GPU performance
		handle = LlamaBridge.init(
			modelPath = modelPath,
			nCtx = 768,
			nThreads = 8,
			nBatch = 32, // Reduced from 64 to 32 to reduce memory pressure and avoid crashes
			nGpuLayers = 5, // Reduced to 5 layers to minimize Vulkan shader operations
			useMmap = true,
			useMlock = false,
			seed = 0,
			callback = callback
		)
		Log.d("BanyaChat", "ensureInit(): LlamaBridge.init handle=$handle")
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
				numPredict = 100,
				temperature = 0.6f,  // iOS와 동일: 0.6
				topP = 0.9f,        // iOS와 동일: 0.9
				topK = 0,           // iOS와 동일: 0 (비활성화)
				repeatPenalty = 1.15f,  // iOS와 동일: 1.15
				repeatLastN = 64,   // iOS와 동일: 64
				stopSequences = emptyArray(),
				callback = callback
			)
			Log.d("BanyaChat", "generateStream(): completionStart dispatched")
		} catch (t: Throwable) {
			Log.e("BanyaChat", "generateStream(): completionStart error -> stub fallback: ${t.message}", t)
			runStubStream(prompt, callback)
		}
	}

	private fun formatPrompt(messages: List<ChatMessage>): String {
		val systemPrompt = """너는 10대 발달장애인의 일상을 돕는 한국어 에이전트다. 말은 간단하고 짧게 한다. 한 번에 한 단계씩 안내한다. 위급한 상황이라고 판단될 경우 즉시 보호자나 119에 연락하도록 안내한다. 복잡한 요청은 다시 확인하고 필요한 정보를 먼저 묻는다. 일정 관리, 준비물 체크, 이동 안내, 감정 조절 도움, 사회적 상황 대처 연습을 친절하게 돕는다. 물결표와 이모티콘, 과도한 문장부호(!!!, .. 등)는 사용하지 않는다. 문장부호는 최대 1개만 사용한다."""
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


