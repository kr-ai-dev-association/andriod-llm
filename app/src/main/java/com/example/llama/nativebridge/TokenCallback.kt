package com.example.llama.nativebridge

/**
 * Unified callback interface for llama native bridge events.
 *
 * Native code reports:
 * - model load progress via [onLoadProgress] (0–100)
 * - parsed model metadata as JSON via [onModelMetadata]
 * - streaming tokens via [onToken]
 * - completion and error signals
 */
interface TokenCallback {
	fun onLoadProgress(progress: Int)
	fun onModelMetadata(json: String)
	fun onToken(token: String)
	fun onCompleted()
	fun onError(message: String)
}

