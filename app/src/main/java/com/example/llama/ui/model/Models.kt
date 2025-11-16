package com.example.llama.ui.model

data class ChatMessage(
	val text: String,
	val isUser: Boolean,
	val timestamp: Long = System.currentTimeMillis()
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
		if (messages.isEmpty()) {
			// If no messages, add a new assistant message with the token
			return copy(messages = messages + ChatMessage(text = token, isUser = false))
		}
		val last = messages.last()
		return if (!last.isUser) {
			// Append token to last assistant message
			val updated = last.copy(text = last.text + token)
			copy(messages = messages.dropLast(1) + updated)
		} else {
			// If last message is from user, add a new assistant message with the token
			copy(messages = messages + ChatMessage(text = token, isUser = false))
		}
	}

	fun withProgress(progress: Int): ChatUiState =
		copy(loadProgress = progress.coerceIn(0, 100))

	fun withMetadata(metadata: ModelMetadata): ChatUiState =
		copy(modelMetadata = metadata, loadProgress = 100)
}


