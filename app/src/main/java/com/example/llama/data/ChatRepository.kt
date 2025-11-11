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
		// Resolve model path: prefer internal, else external app-specific storage
		val internal = File(app.filesDir, "models/llama31-banyaa-q4_k_m.gguf")
		val externalBase = app.getExternalFilesDir(null)
		val external = if (externalBase != null) File(externalBase, "models/llama31-banyaa-q4_k_m.gguf") else null

		// User override path (absolute). Must be a real file path the app can read.
		val overridePath = ModelPathStore.getOverridePath(app).takeIf { !it.isNullOrBlank() }
		val overrideFile = overridePath?.let { File(it) }
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
					assetName = "llama31-banyaa-q4_k_m.gguf",
					outName = "llama31-banyaa-q4_k_m.gguf"
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
		// Defaults aligned with android.md
		handle = LlamaBridge.init(
			modelPath = modelPath,
			nCtx = 2048,
			nThreads = 6,
			nBatch = 128,
			nGpuLayers = 99, // Offload all possible layers to GPU
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
		ensureInit(callback)
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
				temperature = 0.3f,
				topP = 0.85f,
				topK = 50,
				repeatPenalty = 1.2f,
				repeatLastN = 256,
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


