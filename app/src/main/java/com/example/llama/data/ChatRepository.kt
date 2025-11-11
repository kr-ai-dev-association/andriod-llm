package com.example.llama.data

import android.app.Application
import android.util.Log
import com.example.llama.nativebridge.LlamaBridge
import com.example.llama.nativebridge.TokenCallback
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

	private fun ensureInit() {
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
			nCtx = 4096,
			nThreads = Runtime.getRuntime().availableProcessors().coerceAtLeast(2),
			nBatch = 512,
			useMmap = true,
			useMlock = false,
			seed = 0
		)
		Log.d("BanyaChat", "ensureInit(): LlamaBridge.init handle=$handle")
	}

	suspend fun generateStream(
		prompt: String,
		callback: TokenCallback
	) = withContext(Dispatchers.Default) {
		Log.d("BanyaChat", "generateStream(): called with promptLen=${prompt.length}")
		ensureInit()
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
		val fake = "이 모드는 스텁입니다. llama.cpp 연동 설정 후 실제 토큰이 스트리밍됩니다."
		for (ch in fake) {
			callback.onToken(ch.toString())
			delay(12)
		}
		callback.onCompleted()
	}
}


