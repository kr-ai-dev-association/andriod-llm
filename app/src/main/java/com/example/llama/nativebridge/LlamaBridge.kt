package com.example.llama.nativebridge

object LlamaBridge {
	init {
		try {
			System.loadLibrary("llama_jni")
		} catch (t: Throwable) {
			// In stub mode, native lib might still load but do nothing
		}
	}

	external fun init(
		modelPath: String,
		nCtx: Int,
		nThreads: Int,
		nBatch: Int,
		nGpuLayers: Int,
		useMmap: Boolean,
		useMlock: Boolean,
		seed: Int,
		callback: TokenCallback
	): Long

	external fun free(handle: Long)

	external fun completionStart(
		handle: Long,
		prompt: String,
		numPredict: Int,
		temperature: Float,
		topP: Float,
		topK: Int,
		repeatPenalty: Float,
		repeatLastN: Int,
		stopSequences: Array<String>,
		callback: TokenCallback
	)

	external fun completionStop(handle: Long)

	external fun saveSession(handle: Long, path: String): Int
	external fun loadSession(handle: Long, path: String): Boolean
	external fun tokenize(handle: Long, text: String): IntArray
}


