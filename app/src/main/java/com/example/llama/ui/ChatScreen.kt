package com.example.llama.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.TextButton
import androidx.compose.material3.Text
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.llama.ui.model.ChatMessage
import kotlinx.coroutines.launch
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts

@Composable
fun ChatScreen(vm: ChatViewModel = viewModel()) {
	val uiState by vm.uiState.collectAsState()
	val showPathDialog = remember { mutableStateOf(false) }
	val modelPathState = remember { mutableStateOf("") }
	val keyboardController = LocalSoftwareKeyboardController.current

	val filePicker = rememberLauncherForActivityResult(
		contract = ActivityResultContracts.OpenDocument()
	) { uri ->
		if (uri != null) {
			vm.importModelFromUri(uri) { saved ->
				if (saved != null) {
					modelPathState.value = saved
				}
			}
		}
	}
	Column(
		modifier = Modifier
			.fillMaxSize()
			.padding(12.dp)
	) {
		Row(
			modifier = Modifier.fillMaxWidth(),
			verticalAlignment = Alignment.CenterVertically,
			horizontalArrangement = Arrangement.SpaceBetween
		) {
			Text(
				text = "BanyaChat",
				style = MaterialTheme.typography.headlineSmall,
				modifier = Modifier
					.padding(vertical = 4.dp),
				textAlign = TextAlign.Start
			)
			TextButton(onClick = {
				// 바로 파일 선택 다이얼로그 실행
				filePicker.launch(arrayOf("*/*"))
			}) {
				Text("경로 설정")
			}
		}

		Spacer(modifier = Modifier.height(8.dp))

		// Model loading progress indicator - show prominently when loading
		if (uiState.loadProgress in 0..99) {
			Column(
				modifier = Modifier
					.fillMaxWidth()
					.background(
						MaterialTheme.colorScheme.surfaceVariant,
						MaterialTheme.shapes.medium
					)
					.padding(12.dp)
			) {
				Row(
					modifier = Modifier.fillMaxWidth(),
					horizontalArrangement = Arrangement.SpaceBetween,
					verticalAlignment = Alignment.CenterVertically
				) {
					Text(
						text = "모델 로딩 중...",
						style = MaterialTheme.typography.bodyMedium,
						color = MaterialTheme.colorScheme.onSurfaceVariant
					)
					Text(
						text = "${uiState.loadProgress}%",
						style = MaterialTheme.typography.bodyMedium,
						color = MaterialTheme.colorScheme.primary
					)
				}
				Spacer(modifier = Modifier.height(8.dp))
				LinearProgressIndicator(
					progress = uiState.loadProgress / 100f,
					modifier = Modifier
						.fillMaxWidth()
						.height(8.dp),
					trackColor = MaterialTheme.colorScheme.surfaceVariant,
					color = MaterialTheme.colorScheme.primary
				)
			}
			Spacer(modifier = Modifier.height(8.dp))
		}
		
		// Model metadata - show after loading completes
		uiState.modelMetadata?.let { meta ->
			Text(
				text = "${meta.name} · ${meta.quantization} · ctx ${meta.contextLength} · ${meta.sizeLabel}",
				style = MaterialTheme.typography.bodySmall,
				modifier = Modifier.fillMaxWidth(),
				textAlign = TextAlign.Start
			)
			Spacer(modifier = Modifier.height(4.dp))
		}

		LazyColumn(
			modifier = Modifier
				.weight(1f)
				.fillMaxWidth()
				.fillMaxHeight(),
			verticalArrangement = Arrangement.spacedBy(8.dp),
			reverseLayout = false
		) {
			items(uiState.messages) { msg ->
				ChatBubble(message = msg)
			}
		}

		Spacer(modifier = Modifier.height(8.dp))

		val text = remember { mutableStateOf("") }

		Row(
			verticalAlignment = Alignment.CenterVertically,
			modifier = Modifier.fillMaxWidth()
		) {
			OutlinedTextField(
				value = text.value,
				onValueChange = { text.value = it },
				modifier = Modifier
					.weight(1f)
					.padding(end = 8.dp),
				placeholder = { Text("메시지를 입력하세요") },
				singleLine = true,
				enabled = uiState.loadProgress == 100 && !uiState.isGenerating,
				keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
				keyboardActions = KeyboardActions(onSend = {
					if (uiState.loadProgress == 100 && !uiState.isGenerating && text.value.isNotBlank()) {
						vm.send(text.value)
						text.value = ""
						keyboardController?.hide()
					}
				})
			)
			Button(
				onClick = {
					if (uiState.isGenerating) {
						vm.stop()
					} else {
						val content = text.value.trim()
						if (uiState.loadProgress == 100 && content.isNotEmpty()) {
							vm.send(content)
							text.value = ""
							keyboardController?.hide()
						}
					}
				},
				enabled = uiState.loadProgress == 100 || uiState.isGenerating
			) {
				Text(if (uiState.isGenerating) "중지" else "전송")
			}
		}
		if (uiState.isGenerating) {
			Row(
				modifier = Modifier
					.fillMaxWidth()
					.padding(top = 8.dp),
				verticalAlignment = Alignment.CenterVertically,
				horizontalArrangement = Arrangement.Center
			) {
				CircularProgressIndicator()
			}
		}
	}

	if (showPathDialog.value) {
		ModelPathDialog(
			initial = modelPathState.value,
			onDismiss = { showPathDialog.value = false },
			onSave = { newPath ->
				vm.setModelPathOverride(newPath)
				showPathDialog.value = false
			},
			onPick = {
				// Allow any type; user selects gguf
				filePicker.launch(arrayOf("*/*"))
			}
		)
	}
}

@Composable
private fun ChatBubble(message: ChatMessage) {
	val bg = if (message.isUser) {
		MaterialTheme.colorScheme.primaryContainer
	} else {
		MaterialTheme.colorScheme.secondaryContainer
	}
	Column(
		modifier = Modifier
			.fillMaxWidth()
			.background(bg)
			.padding(12.dp)
	) {
		Text(
			text = if (message.isUser) "나" else "LLM",
			style = MaterialTheme.typography.labelMedium
		)
		Spacer(modifier = Modifier.height(4.dp))
		Text(text = message.text, style = MaterialTheme.typography.bodyMedium)
	}
}

@Composable
private fun ModelPathDialog(
	initial: String,
	onDismiss: () -> Unit,
	onSave: (String) -> Unit,
	onPick: () -> Unit
) {
	val pathState = remember { mutableStateOf(initial) }
	Column(
		modifier = Modifier
			.fillMaxWidth()
			.padding(12.dp)
	) {
		Text(text = "모델 파일 경로", style = MaterialTheme.typography.titleMedium)
		Spacer(modifier = Modifier.height(8.dp))
		OutlinedTextField(
			value = pathState.value,
			onValueChange = { pathState.value = it },
			modifier = Modifier.fillMaxWidth(),
			singleLine = true,
			placeholder = { Text("/sdcard/Android/data/com.example.llama/files/models/llama31-banyaa-q4_k_m.gguf") }
		)
		Spacer(modifier = Modifier.height(12.dp))
		Row(
			modifier = Modifier.fillMaxWidth(),
			horizontalArrangement = Arrangement.End
		) {
			TextButton(onClick = onPick) { Text("파일 선택") }
			TextButton(onClick = onDismiss) { Text("취소") }
			Spacer(modifier = Modifier.padding(horizontal = 4.dp))
			Button(onClick = { onSave(pathState.value.trim()) }) { Text("저장") }
		}
	}
}


