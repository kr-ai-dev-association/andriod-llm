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

	/**
	 * 동기식 completion 함수 - 짧은 텍스트(JSON 등)를 생성하고 즉시 반환
	 * RAG 시스템의 1차 판단 단계에서 사용
	 */
	external fun completion(
		handle: Long,
		prompt: String,
		numPredict: Int = 256,
		temperature: Float = 0.7f,
		topP: Float = 0.9f,
		topK: Int = 40,
		repeatPenalty: Float = 1.1f,
		repeatLastN: Int = 64,
		stopSequences: Array<String> = arrayOf("\n\n", "User:", "Assistant:")
	): String

	external fun saveSession(handle: Long, path: String): Int
	external fun loadSession(handle: Long, path: String): Boolean
	external fun tokenize(handle: Long, text: String): IntArray
}


