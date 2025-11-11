package com.example.llama.ui

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.data.ChatRepository
import com.example.llama.data.ModelPathStore
import com.example.llama.data.ModelFiles
import com.example.llama.nativebridge.TokenCallback
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import com.example.llama.ui.model.ChatMessage
import com.example.llama.ui.model.ChatUiState
import com.example.llama.ui.model.ModelMetadata
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.IOException

class ChatViewModel(app: Application) : AndroidViewModel(app) {

	private val repo = ChatRepository(app)

	private val _uiState = MutableStateFlow(ChatUiState())
	val uiState: StateFlow<ChatUiState> = _uiState

	init {
		ModelPathStore.getModelMetadata(app)?.let { stored ->
			runCatching {
				val obj = JSONObject(stored)
				ModelMetadata(
					name = obj.optString("name", "N/A"),
					quantization = obj.optString("quantization", "N/A"),
					contextLength = obj.optInt("context_length", 0),
					sizeLabel = obj.optString("size_label", "N/A")
				)
			}.onSuccess { meta ->
				_uiState.value = _uiState.value.withMetadata(meta)
			}
		}
	}

	fun getModelPathOverride(): String {
		return ModelPathStore.getOverridePath(getApplication()) ?: ""
	}

	fun setModelPathOverride(path: String) {
		ModelPathStore.setOverridePath(getApplication(), path.ifBlank { null })
		ModelPathStore.setModelMetadata(getApplication(), null)
		repo.reset()
	}

	fun importModelFromUri(uri: Uri, onDone: (String?) -> Unit) {
		viewModelScope.launch(Dispatchers.IO) {
			val app = getApplication<Application>()
			val cr = app.contentResolver
			val name = runCatching {
				cr.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
					if (c.moveToFirst()) c.getString(0) else null
				}
			}.getOrNull() ?: "model.gguf"

			return@launch try {
				ModelFiles.ensureModelDir(app)
				val dest = File(app.filesDir, "models/$name")
				cr.openInputStream(uri).use { input ->
					dest.outputStream().use { output ->
						if (input == null) throw IOException("입력 스트림을 열 수 없습니다.")
						input.copyTo(output)
					}
				}
				ModelPathStore.setOverridePath(app, dest.absolutePath)
				repo.reset()
				withContext(Dispatchers.Main) { onDone(dest.absolutePath) }
			} catch (t: Throwable) {
				withContext(Dispatchers.Main) { onDone(null) }
			}
		}
	}

	fun send(userText: String) {
		// Add user message
		_uiState.value = _uiState.value.addMessage(ChatMessage(text = userText, isUser = true))
		// Start generation
		generate()
	}

	private fun generate() {
		if (_uiState.value.isGenerating) return
		_uiState.value = _uiState.value.copy(isGenerating = true)

		val builderIndex = _uiState.value.messages.size
		_uiState.value = _uiState.value.addMessage(ChatMessage(text = "", isUser = false))

		val messages = _uiState.value.messages

		viewModelScope.launch {
			repo.generateStream(
				messages = messages.dropLast(1), // Don't include the empty assistant message placeholder
				callback = object : TokenCallback {
					override fun onLoadProgress(progress: Int) {
						_uiState.value = _uiState.value.withProgress(progress)
					}

					override fun onModelMetadata(json: String) {
						runCatching {
							val obj = JSONObject(json)
							ModelMetadata(
								name = obj.optString("name", "N/A"),
								quantization = obj.optString("quantization", "N/A"),
								contextLength = obj.optInt("context_length", 0),
								sizeLabel = obj.optString("size_label", "N/A")
							)
						}.onSuccess { metadata ->
							ModelPathStore.setModelMetadata(getApplication(), json)
							_uiState.value = _uiState.value.withMetadata(metadata)
						}
					}

					override fun onToken(token: String) {
						_uiState.value = _uiState.value.appendToLastAssistant(token)
					}

					override fun onCompleted() {
						_uiState.value = _uiState.value.copy(isGenerating = false)
					}

					override fun onError(message: String) {
						_uiState.value = _uiState.value.copy(isGenerating = false)
						_uiState.value = _uiState.value.addMessage(
							ChatMessage(text = "에러: $message", isUser = false)
						)
						_uiState.value = _uiState.value.withProgress(0)
					}
				}
			)
		}
	}

	fun stop() {
		repo.stop()
		_uiState.value = _uiState.value.copy(isGenerating = false)
	}
}


