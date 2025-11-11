package com.example.llama.ui.model

data class ChatMessage(
	val text: String,
	val isUser: Boolean
)

data class ChatUiState(
	val messages: List<ChatMessage> = emptyList(),
	val isGenerating: Boolean = false
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
}


