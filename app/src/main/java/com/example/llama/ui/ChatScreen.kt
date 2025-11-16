package com.example.llama.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.llama.ui.model.ChatMessage
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.ComponentActivity
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.setValue
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import android.util.Log

@Composable
fun ChatScreen(vm: ChatViewModel = viewModel()) {
	val uiState by vm.uiState.collectAsState()
	val showPathDialog = remember { mutableStateOf(false) }
	
	// 프로그래스바 디버깅을 위한 로그
	LaunchedEffect(uiState.loadProgress) {
		Log.d("BanyaChat", "ChatScreen: loadProgress=${uiState.loadProgress}, isGenerating=${uiState.isGenerating}")
	}
	val modelPathState = remember { mutableStateOf("") }
	val keyboardController = LocalSoftwareKeyboardController.current
	val text = remember { mutableStateOf("") }
	val context = LocalContext.current
	val activity = context as? ComponentActivity

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

	// 배경 그라데이션
	Box(
		modifier = Modifier
			.fillMaxSize()
					.background(
				Brush.verticalGradient(
					colors = listOf(
						Color(0xFFE8D5FF), // 라이트 퍼플
						Color.White
					)
				)
			)
			) {
		Column(
			modifier = Modifier.fillMaxSize()
		) {
			// 상단 헤더 (프로그레스 바 포함)
			ChatHeader(
				loadProgress = uiState.loadProgress,
				onBackClick = {
					// 이전 화면으로 이동
					activity?.finish()
				},
				onMenuClick = {
					// 아무 기능도 하지 않음
				}
			)

			// 메시지 리스트
		LazyColumn(
			modifier = Modifier
				.weight(1f)
					.fillMaxWidth(),
				contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 16.dp, vertical = 8.dp),
				verticalArrangement = Arrangement.spacedBy(12.dp)
		) {
				items(uiState.messages.size) { index ->
					val msg = uiState.messages[index]
					val isLastMessage = index == uiState.messages.size - 1
					val isThinking = isLastMessage && !msg.isUser && msg.text.isEmpty() && uiState.isGenerating
					ChatBubble(message = msg, isThinking = isThinking)
				}
			}

			// 입력 바
			ChatInputBar(
				text = text.value,
				onTextChange = { text.value = it },
			onSend = {
					if (uiState.loadProgress == 100 && !uiState.isGenerating && text.value.isNotBlank()) {
					Log.d("BanyaChat", "ChatScreen: onSend called with text='${text.value}'")
					Log.d("BanyaChat", "ChatScreen: loadProgress=${uiState.loadProgress}, isGenerating=${uiState.isGenerating}")
						vm.send(text.value)
						text.value = ""
						keyboardController?.hide()
					} else {
					Log.d("BanyaChat", "ChatScreen: onSend blocked - loadProgress=${uiState.loadProgress}, isGenerating=${uiState.isGenerating}, text.isNotBlank=${text.value.isNotBlank()}")
				}
			},
				onAttachClick = {
					filePicker.launch(arrayOf("*/*"))
				},
				enabled = uiState.loadProgress == 100 && !uiState.isGenerating,
				isGenerating = uiState.isGenerating,
				onStop = { vm.stop() }
			)
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
				filePicker.launch(arrayOf("*/*"))
			}
		)
	}
}

@Composable
private fun ChatHeader(
	loadProgress: Int,
	onBackClick: () -> Unit,
	onMenuClick: () -> Unit
) {
	// 프로그래스바 디버깅을 위한 로그
	Log.d("BanyaChat", "ChatHeader: loadProgress=$loadProgress, shouldShowProgress=${loadProgress in 0..99}")
	
	Column(
		modifier = Modifier
			.fillMaxWidth()
			.background(Color.White)
	) {
		Row(
			modifier = Modifier
				.fillMaxWidth()
				.padding(horizontal = 16.dp, vertical = 12.dp),
			verticalAlignment = Alignment.CenterVertically,
			horizontalArrangement = Arrangement.SpaceBetween
		) {
			Row(
				verticalAlignment = Alignment.CenterVertically,
				modifier = Modifier.weight(1f)
			) {
				IconButton(onClick = onBackClick) {
					Icon(
						imageVector = Icons.Default.ArrowBack,
						contentDescription = "뒤로가기",
						tint = Color(0xFF424242)
					)
				}
				
				// 프로필 이미지 (원형)
				Box(
					modifier = Modifier
						.size(40.dp)
						.clip(CircleShape)
						.background(Color(0xFFE0E0E0))
						.border(1.dp, Color(0xFFBDBDBD), CircleShape),
					contentAlignment = Alignment.Center
				) {
					Text(
						text = "B",
						style = MaterialTheme.typography.titleMedium,
						color = Color(0xFF424242),
						fontWeight = FontWeight.Bold
					)
				}
				
				Spacer(modifier = Modifier.width(12.dp))
				
				Column {
					Text(
						text = "BanyaLLM",
						style = MaterialTheme.typography.titleMedium,
						fontWeight = FontWeight.Bold,
						color = Color(0xFF212121)
					)
					if (loadProgress in 0..99) {
						// 프로그래스바만 표시, 텍스트는 제거
					} else {
						Text(
							text = "Online",
							style = MaterialTheme.typography.bodySmall,
							color = Color(0xFF66BB6A),
							fontSize = 12.sp
						)
					}
				}
			}
			
			IconButton(onClick = onMenuClick) {
				Icon(
					imageVector = Icons.Default.MoreVert,
					contentDescription = "메뉴",
					tint = Color(0xFF424242)
				)
			}
		}
		
		// 프로그레스 바 (애니메이션 포함)
		if (loadProgress < 100) {
			// 진행률이 있으면 프로그래스바 표시, 없으면 무한 로딩 인디케이터
			if (loadProgress > 0) {
				val animatedProgress by animateFloatAsState(
					targetValue = loadProgress / 100f,
					animationSpec = tween(
						durationMillis = 300,
						easing = androidx.compose.animation.core.FastOutSlowInEasing
					),
					label = "progress_animation"
				)
				
				Log.d("BanyaChat", "ChatHeader: Showing progress bar, loadProgress=$loadProgress, animatedProgress=$animatedProgress")
				
				LinearProgressIndicator(
					progress = { animatedProgress },
					modifier = Modifier
						.fillMaxWidth()
						.height(3.dp),
					color = Color(0xFF9C27B0),
					trackColor = Color(0xFFE1BEE7)
				)
			} else {
				// 진행률이 0이면 무한 로딩 인디케이터 표시
				Log.d("BanyaChat", "ChatHeader: Showing indeterminate progress bar, loadProgress=$loadProgress")
				
				LinearProgressIndicator(
					modifier = Modifier
						.fillMaxWidth()
						.height(3.dp),
					color = Color(0xFF9C27B0),
					trackColor = Color(0xFFE1BEE7)
				)
			}
		} else {
			Log.d("BanyaChat", "ChatHeader: Not showing progress bar, loadProgress=$loadProgress (completed)")
		}
	}
}

@Composable
private fun ChatBubble(message: ChatMessage, isThinking: Boolean = false) {
	// 메시지 타임스탬프를 날짜/시간 형식으로 변환
	val dateFormat = remember {
		SimpleDateFormat("yyyy년 MM월 dd일 HH시 mm분", Locale.KOREAN)
	}
	val timestamp = remember(message.timestamp) {
		dateFormat.format(Date(message.timestamp))
	}
	
	// "생각중 ..." 애니메이션
	var dotCount by remember { mutableStateOf(1) }
	
	LaunchedEffect(isThinking) {
		if (isThinking) {
			while (true) {
				dotCount = (dotCount % 3) + 1
				kotlinx.coroutines.delay(500)
			}
		}
	}
	
	Row(
		modifier = Modifier.fillMaxWidth(),
		horizontalArrangement = if (message.isUser) Arrangement.End else Arrangement.Start
	) {
		if (!message.isUser) {
			Spacer(modifier = Modifier.width(8.dp))
		}
		
		Column(
			modifier = Modifier.widthIn(max = 280.dp),
			horizontalAlignment = if (message.isUser) Alignment.End else Alignment.Start
		) {
			Box(
				modifier = Modifier
					.background(
						color = if (message.isUser) Color(0xFFE3F2FD) else Color(0xFFF5F5F5),
						shape = RoundedCornerShape(16.dp)
					)
					.padding(horizontal = 12.dp, vertical = 10.dp)
			) {
				if (isThinking && message.text.isEmpty()) {
					Text(
						text = "생각중${".".repeat(dotCount)}",
						style = MaterialTheme.typography.bodyMedium,
						color = Color(0xFF9E9E9E),
						fontSize = 14.sp
					)
				} else {
					Text(
						text = message.text,
						style = MaterialTheme.typography.bodyMedium,
						color = Color(0xFF212121),
						fontSize = 14.sp
					)
				}
			}
			// 타임스탬프 표시 (생각중 상태가 아닐 때만)
			if (!isThinking) {
				Spacer(modifier = Modifier.height(4.dp))
				Text(
					text = timestamp,
					style = MaterialTheme.typography.bodySmall,
					color = Color(0xFF9E9E9E),
					fontSize = 11.sp,
					modifier = Modifier.padding(horizontal = 4.dp)
				)
			}
		}
		
		if (message.isUser) {
			Spacer(modifier = Modifier.width(8.dp))
		}
	}
}

@Composable
private fun ChatInputBar(
	text: String,
	onTextChange: (String) -> Unit,
	onSend: () -> Unit,
	onAttachClick: () -> Unit,
	enabled: Boolean,
	isGenerating: Boolean,
	onStop: () -> Unit
) {
	Row(
		modifier = Modifier
			.fillMaxWidth()
			.background(Color.White)
			.padding(horizontal = 12.dp, vertical = 8.dp)
			.padding(bottom = 16.dp),
		verticalAlignment = Alignment.CenterVertically
	) {
		// 파일 첨부 버튼
		IconButton(
			onClick = onAttachClick,
			enabled = enabled
		) {
			Icon(
				imageVector = Icons.Default.Add,
				contentDescription = "파일 첨부",
				tint = if (enabled) Color(0xFF424242) else Color(0xFFBDBDBD)
			)
		}
		
		// 입력 필드
		TextField(
			value = text,
			onValueChange = onTextChange,
			modifier = Modifier
				.weight(1f)
				.height(56.dp),
			placeholder = { 
		Text(
					"메시지를 입력하세요", 
					fontSize = 14.sp,
					lineHeight = 20.sp
				) 
			},
			textStyle = androidx.compose.ui.text.TextStyle(
				fontSize = 14.sp,
				lineHeight = 20.sp
			),
			enabled = enabled,
			colors = TextFieldDefaults.colors(
				unfocusedContainerColor = Color(0xFFF5F5F5),
				focusedContainerColor = Color(0xFFF5F5F5),
				unfocusedIndicatorColor = Color.Transparent,
				focusedIndicatorColor = Color.Transparent,
				unfocusedPlaceholderColor = Color(0xFF9E9E9E),
				focusedPlaceholderColor = Color(0xFF9E9E9E)
			),
			shape = RoundedCornerShape(24.dp),
			singleLine = true,
			keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
			keyboardActions = KeyboardActions(onSend = {
				if (enabled && text.isNotBlank()) {
					onSend()
				}
			})
		)
		
		// 전송 버튼
		IconButton(
			onClick = {
				if (isGenerating) {
					onStop()
				} else {
					onSend()
				}
			},
			enabled = enabled || isGenerating
		) {
			Icon(
				imageVector = Icons.Default.Send,
				contentDescription = if (isGenerating) "중지" else "전송",
				tint = if (enabled || isGenerating) Color(0xFF9C27B0) else Color(0xFFBDBDBD)
			)
		}
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
			placeholder = { Text("/sdcard/Download/llama31-banyaa-q4_0.gguf") }
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


