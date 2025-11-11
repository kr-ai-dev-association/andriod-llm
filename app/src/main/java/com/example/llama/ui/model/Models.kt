package com.example.llama.ui.model

data class ChatMessage(
	val text: String,
	val isUser: Boolean
)

data class ModelMetadata(
	val name: String,
	val quantization: String,
	val contextLength: Int,
	val sizeLabel: String
)

data class ChatUiState(
	val messages: List<ChatMessage> = emptyList(),
	val isGenerating: Boolean = false,
	val loadProgress: Int = 0,
	val modelMetadata: ModelMetadata? = null
) {
	fun addMessage(message: ChatMessage): ChatUiState {
		return copy(messages = messages + message)
	}
	fun appendToLastAssistant(token: String): ChatUiState {
		if (messages.isEmpty()) return this
		val last = messages.last()
		return if (!last.isUser) {
			val updated = last.copy(text = last.text + token)
			copy(messages = messages.dropLast(1) + updated)
		} else {
			this
		}
	}

	fun withProgress(progress: Int): ChatUiState =
		copy(loadProgress = progress.coerceIn(0, 100))

	fun withMetadata(metadata: ModelMetadata): ChatUiState =
		copy(modelMetadata = metadata, loadProgress = 100)
}


