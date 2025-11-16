package com.example.llama.ui

import android.app.Application
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.provider.OpenableColumns
import android.util.Log
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
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.TimeoutCancellationException
import org.json.JSONObject
import java.io.File
import java.io.IOException
import com.example.llama.data.TavilyApiService
import com.example.llama.data.TavilySearchRequest
import com.example.llama.nativebridge.LlamaBridge
import com.google.gson.Gson
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor

class ChatViewModel(app: Application) : AndroidViewModel(app) {

	private val repo = ChatRepository(app)
	private val mainHandler = Handler(Looper.getMainLooper())

	private val _uiState = MutableStateFlow(ChatUiState())
	val uiState: StateFlow<ChatUiState> = _uiState

	// RAG 시스템을 위한 Tavily API 설정
	// local.properties에서 읽어온 API 키 사용
	private val tavilyApiKey = com.example.llama.BuildConfig.TAVILY_API_KEY
	private val gson = Gson()
	
	// Tavily API 서비스 초기화
	private val tavilyApi: TavilyApiService by lazy {
		val loggingInterceptor = HttpLoggingInterceptor().apply {
			level = HttpLoggingInterceptor.Level.BODY
		}
		val client = OkHttpClient.Builder()
			.addInterceptor(loggingInterceptor)
			.build()
		
		Retrofit.Builder()
			.baseUrl("https://api.tavily.com/")
			.client(client)
			.addConverterFactory(GsonConverterFactory.create())
			.build()
			.create(TavilyApiService::class.java)
	}

	/**
	 * 자동 테스트 케이스 배열
	 * "우울할때 뭘 하면 좋을까?" - 미완성 종료 테스트
	 */
	private val testCases = listOf(
		"우울할때 뭘 하면 좋을까?"
	)

	/**
	 * 현재 테스트 케이스 인덱스
	 */
	private var currentTestIndex = 0

	/**
	 * 테스트 모드 활성화 여부
	 */
	private var isTestMode = false

	/**
	 * 한국어 인사말을 모아둔 Set.
	 * 빠른 조회를 위해 Set을 사용하고, 소문자로 저장하여 대소문자 구분 없이 비교합니다.
	 */
	private val koreanGreetings = setOf(
		"안녕", "안녕하세요", "안녕하세요.", "안뇽",
		"하이", "하이루", "ㅎㅇ", "ㅎ2",
		"좋은 아침", "좋은아침", "굿모닝",
		"좋은 오후", "좋은오후",
		"좋은 저녁", "좋은저녁", "굿나잇",
		"잘 지내?", "잘 지내요?", "오랜만이야", "오랜만이네요"
	)

	/**
	 * 영어 인사말을 모아둔 Set.
	 * 빠른 조회를 위해 Set을 사용하고, 소문자로 저장하여 대소문자 구분 없이 비교합니다.
	 */
	private val englishGreetings = setOf(
		"hello", "hi", "hey", "yo", "greetings",
		"sup", "what's up", "whatsup", "wassup",
		"what's good", "what's new", "what's happening",
		"how are you", "how are you?", "how are ya",
		"how's it going", "hows it going",
		"how have you been", "how've you been",
		"how do you do",
		"are you ok", "are you okay",
		"you alright?", "you alright",
		"good morning", "goodmorning", "morning",
		"good afternoon", "afternoon",
		"good evening", "evening",
		"good night", "goodnight",
		"long time no see",
		"it's been a while",
		"nice to see you", "nice to see you again",
		"good to see you", "good to see you again",
		"pleased to meet you",
		"it's a pleasure to meet you",
		"wud", "wyd", "hru"
	)

	/**
	 * 모든 인사말을 포함한 Set (기존 호환성을 위해 유지).
	 */
	private val commonGreetings = koreanGreetings + englishGreetings

	/**
	 * 한국어 인사말 답변 목록.
	 */
	private val koreanGreetingResponses = listOf(
		"안녕하세요! 무엇을 도와드릴까요?",
		"네, 안녕하세요! 어떤 질문이 있으신가요?",
		"반갑습니다! 편하게 질문해주세요.",
		"안녕하세요! 오늘은 무엇이 궁금하신가요?",
		"네, 안녕하세요! 다시 만나서 반가워요."
	)

	/**
	 * 영어 인사말 답변 목록.
	 */
	private val englishGreetingResponses = listOf(
		"Hello! How can I help you today?",
		"Hi there! What can I do for you?",
		"Greetings! Feel free to ask me anything."
	)

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

		// Preload model at startup so users don't type before load completes
		viewModelScope.launch(Dispatchers.Default) {
			var modelLoadSuccess = false
			var hasReceivedMetadata = false
			var errorMessage: String? = null
			var errorReported = false
			
			repo.preload(object : TokenCallback {
				override fun onLoadProgress(progress: Int) {
					// 모든 진행률 업데이트 (0-100%)
					mainHandler.post {
						Log.d("BanyaChat", "ChatViewModel.onLoadProgress: progress=$progress")
						_uiState.value = _uiState.value.withProgress(progress)
					}
				}
				override fun onModelMetadata(json: String) {
					// 모델 메타데이터가 수신되면 모델이 성공적으로 로드된 것
					hasReceivedMetadata = true
					errorMessage = null // 에러 메시지 초기화
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
						mainHandler.post {
							_uiState.value = _uiState.value.withMetadata(metadata)
							_uiState.value = _uiState.value.withProgress(100)
						}
					}
				}
				override fun onToken(token: String) {}
				override fun onCompleted() {}
				override fun onError(message: String) {
					// 에러 메시지를 기록하되, 즉시 표시하지 않음
					// 모델 로딩이 완료될 시간을 주고 실제 실패인지 확인
					// onError가 호출되어도 모델 로딩이 계속 진행될 수 있으므로
					// preload 완료 후에만 최종적으로 에러 메시지를 표시
					if (!hasReceivedMetadata) {
						errorMessage = message
					}
				}
			})
			
			// preload가 완료될 때까지 대기한 후 모델 로딩 상태 확인
			// 모델 메타데이터가 수신되었거나 handle이 0이 아니면 성공
			// onModelMetadata가 호출되면 자동으로 progress를 100%로 설정하므로
			// 여기서는 메타데이터를 받지 못한 경우에만 체크
			if (!hasReceivedMetadata) {
				// 모델 로딩이 완료될 시간을 줌 (최대 15초 대기)
				// 큰 모델의 경우 로딩에 시간이 걸릴 수 있음
				var waitCount = 0
				while (!hasReceivedMetadata && waitCount < 150 && !repo.isInitialized()) {
					kotlinx.coroutines.delay(100)
					waitCount++
					// 중간에 메타데이터를 받았으면 즉시 종료
					if (hasReceivedMetadata) break
				}
				modelLoadSuccess = repo.isInitialized()
				
				// 모델이 로드되었지만 메타데이터를 받지 못한 경우 progress를 100%로 설정
				if (modelLoadSuccess && !hasReceivedMetadata) {
					mainHandler.post {
						_uiState.value = _uiState.value.withProgress(100)
					}
				} else if (!modelLoadSuccess && !hasReceivedMetadata && !errorReported) {
					// 모델이 로드되지 않았고 메타데이터도 받지 못한 경우
					// 에러 메시지가 있으면 표시, 없으면 기본 메시지 표시
					errorReported = true
					mainHandler.post {
						val errorText = if (errorMessage != null) {
							"로딩 실패: $errorMessage"
						} else {
							"로딩 실패: 모델을 로드할 수 없습니다"
						}
						_uiState.value = _uiState.value.addMessage(
							ChatMessage(text = errorText, isUser = false)
						)
						_uiState.value = _uiState.value.withProgress(0)
					}
				}
			} else {
				// 메타데이터를 받았으면 이미 progress가 100%로 설정됨
				modelLoadSuccess = true
			}
			
			// Automatically send test question when model load completes successfully
			// 주석 처리: 자동 테스트 질문 비활성화
			// if (modelLoadSuccess && _uiState.value.messages.isEmpty()) {
			// 	// Only send test question if no messages exist yet (first load)
			// 	mainHandler.post {
			// 		Log.d("BanyaChat", "Model load complete, automatically sending test question")
			// 		send("내일 대한민국 서울 날씨 알려줘")
			// 	}
			// }
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
		Log.d("BanyaChat", "========================================")
		Log.d("BanyaChat", "ChatViewModel.send(): called with text='$userText'")
		Log.d("BanyaChat", "ChatViewModel.send(): text length=${userText.length}")
		// 1. 입력 정규화: 앞뒤 공백을 제거하고 모두 소문자로 변환하여 비교 준비.
		val normalizedInput = userText.trim().lowercase()
		Log.d("BanyaChat", "ChatViewModel.send(): normalizedInput='$normalizedInput'")

		// 입력이 비어있으면 아무것도 하지 않음.
		if (normalizedInput.isEmpty()) {
			return
		}

		// 2. 입력이 단순 인사말인지 확인하고 언어 판단.
		val isKoreanGreeting = koreanGreetings.any { greeting ->
			normalizedInput == greeting
		}
		val isEnglishGreeting = englishGreetings.any { greeting ->
			normalizedInput == greeting
		}
		val isGreeting = isKoreanGreeting || isEnglishGreeting

		// 3. 휴리스틱(Heuristic) 적용:
		// "안녕하세요, 오늘 날씨 어때요?"와 같은 복합적인 질문을 LLM으로 넘기기 위해,
		// 입력이 매우 짧은 경우에만 인사말로 간주합니다. (예: 단어 2개 이하)
		val wordCount = normalizedInput.split(Regex("\\s+")).size
		val isSimpleEnough = wordCount <= 2

		// 4. 조건부 처리: 입력이 '인사말'이고 '충분히 단순'할 때만 즉시 응답.
		if (isGreeting && isSimpleEnough) {
			// 사용자 메시지 추가
			_uiState.value = _uiState.value.addMessage(ChatMessage(text = userText, isUser = true))
			
			// 입력 언어에 맞는 답변만 선택.
			val randomResponse = when {
				isKoreanGreeting -> koreanGreetingResponses.random()
				isEnglishGreeting -> englishGreetingResponses.random()
				else -> koreanGreetingResponses.random() // 기본값 (발생하지 않아야 함)
			}
			
			// UI에 즉시 답변을 표시.
			_uiState.value = _uiState.value.addMessage(ChatMessage(text = randomResponse, isUser = false))
			
			// LLM을 호출하지 않고 여기서 로직 종료.
			return
		}

		// 5. 위의 조건에 해당하지 않는 모든 입력은 RAG 시스템으로 전달.
		Log.d("BanyaChat", "ChatViewModel.send(): Not a greeting, starting RAG process")
		
		// 세션 관리: 각 사용자 질의마다 새로운 세션 시작 (메모리 과부하 방지)
		// UI에는 모든 대화를 표시하지만, 프롬프트 생성 시에는 현재 질의만 사용
		// 이전 대화 기록은 UI 표시용으로만 유지하고, 실제 LLM 호출 시에는 제외
		
		// Add user message
		_uiState.value = _uiState.value.addMessage(ChatMessage(text = userText, isUser = true))
		Log.d("BanyaChat", "ChatViewModel.send(): User message added to UI state")
		Log.d("BanyaChat", "Session management: Starting new session for this query (previous context will be excluded)")
		// Start RAG process
		processUserInputWithRAG(userText)
		Log.d("BanyaChat", "ChatViewModel.send(): processUserInputWithRAG() called")
	}

	private fun generate() {
		// isGenerating 체크를 제거: RAG에서 이미 설정했을 수 있으므로
		// Block generation until model load has reached 100
		if (_uiState.value.loadProgress in 0..99) {
			_uiState.value = _uiState.value.addMessage(ChatMessage(text = "모델 로딩 중입니다. 잠시만 기다려 주세요.", isUser = false))
			return
		}
		_uiState.value = _uiState.value.copy(isGenerating = true)

		// 빈 메시지가 이미 마지막에 없으면 추가 (RAG에서 이미 추가했을 수 있음)
		val lastMessage = _uiState.value.messages.lastOrNull()
		if (lastMessage == null || lastMessage.isUser || lastMessage.text.isNotEmpty()) {
			_uiState.value = _uiState.value.addMessage(ChatMessage(text = "", isUser = false))
		}
		val builderIndex = _uiState.value.messages.size

		val messages = _uiState.value.messages
		
		// 세션 관리: 각 사용자 질의마다 새로운 세션 시작
		// 현재 질의(마지막 사용자 메시지)만 포함하고, 이전 대화 기록은 모두 제외
		val currentUserMessage = messages.lastOrNull { it.isUser }
		val sessionMessages = if (currentUserMessage != null) {
			listOf(currentUserMessage) // 현재 질의만 포함
		} else {
			emptyList()
		}
		
		Log.d("BanyaChat", "Session management: Using only current query (${sessionMessages.size} messages) instead of full history (${messages.size} messages)")

		viewModelScope.launch {
			repo.generateStream(
				messages = sessionMessages, // 현재 질의만 포함 (새 세션)
				callback = object : TokenCallback {
					override fun onLoadProgress(progress: Int) {
						// Dispatch to main thread for UI updates
						mainHandler.post {
							_uiState.value = _uiState.value.withProgress(progress)
						}
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
							mainHandler.post {
								_uiState.value = _uiState.value.withMetadata(metadata)
							}
						}
					}

					override fun onToken(token: String) {
						Log.d("BanyaChat", "ChatViewModel.onToken(): received token='$token', posting to main thread")
						mainHandler.post {
							Log.d("BanyaChat", "ChatViewModel.onToken(): updating UI with token='$token', current messages count=${_uiState.value.messages.size}")
							_uiState.value = _uiState.value.appendToLastAssistant(token)
							Log.d("BanyaChat", "ChatViewModel.onToken(): UI updated, new messages count=${_uiState.value.messages.size}, last message text='${_uiState.value.messages.lastOrNull()?.text}'")
						}
					}

					override fun onCompleted() {
						mainHandler.post {
							_uiState.value = _uiState.value.copy(isGenerating = false)
							
							// 테스트 모드에서 다음 테스트 케이스 자동 전송
							// 테스트 종료를 위해 주석처리
							/*
							if (isTestMode) {
								val lastMessage = _uiState.value.messages.lastOrNull()?.text ?: ""
								Log.d("BanyaChat", "Test case ${currentTestIndex + 1}/${testCases.size} completed. Last message: ${lastMessage.take(100)}")
								
								// 다음 테스트 케이스로 이동
								currentTestIndex++
								
								if (currentTestIndex < testCases.size) {
									// 다음 테스트 케이스 전송 (약간의 지연 후)
									mainHandler.postDelayed({
										send(testCases[currentTestIndex])
									}, 1000) // 1초 후 다음 테스트 시작
								} else {
									// 모든 테스트 완료 - 반복 시작
									Log.d("BanyaChat", "All test cases completed. Restarting test cycle...")
									currentTestIndex = 0
									mainHandler.postDelayed({
										send(testCases[currentTestIndex])
									}, 2000) // 2초 후 다시 시작
								}
							}
							*/
						}
					}

					override fun onError(message: String) {
						mainHandler.post {
							_uiState.value = _uiState.value.copy(isGenerating = false)
							_uiState.value = _uiState.value.addMessage(
								ChatMessage(text = "에러: $message", isUser = false)
							)
							_uiState.value = _uiState.value.withProgress(0)
						}
					}
				}
			)
		}
	}

	fun stop() {
		repo.stop()
		_uiState.value = _uiState.value.copy(isGenerating = false)
	}

	/**
	 * 카테고리 감지: 사용자 입력이 웹 검색이 필요한 카테고리인지 확인
	 * @return true면 웹 검색 필요, false면 LLM 자체 지식으로 답변
	 */
	private fun shouldUseWebSearch(userInput: String): Boolean {
		val normalizedInput = userInput.lowercase().trim()
		
		// 날씨 카테고리
		val weatherPattern = Regex("(날씨|기온|온도|비|눈|맑음|흐림|강수|폭염|한파|기후|예보|날씨.*알려|날씨.*어때|날씨.*어떤)")
		if (weatherPattern.containsMatchIn(normalizedInput)) {
			Log.d("BanyaChat", "Category detected: 날씨")
			return true
		}
		
		// 주식 카테고리
		val stockPattern = Regex("(주식|주가|증권|코스피|코스닥|삼성전자|애플|테슬라|비트코인|암호화폐|투자|시세|종목|증시)")
		if (stockPattern.containsMatchIn(normalizedInput)) {
			Log.d("BanyaChat", "Category detected: 주식")
			return true
		}
		
		// 대통령 카테고리
		val presidentPattern = Regex("(대통령|대통령은|대통령이|대통령의|대통령.*누구|현재.*대통령|한국.*대통령|미국.*대통령)")
		if (presidentPattern.containsMatchIn(normalizedInput)) {
			Log.d("BanyaChat", "Category detected: 대통령")
			return true
		}
		
		// 뉴스 카테고리
		val newsPattern = Regex("(뉴스|최신.*뉴스|오늘.*뉴스|뉴스.*알려|뉴스.*보여|뉴스.*있어)")
		if (newsPattern.containsMatchIn(normalizedInput)) {
			Log.d("BanyaChat", "Category detected: 뉴스")
			return true
		}
		
		// 기사 카테고리
		val articlePattern = Regex("(기사|최신.*기사|오늘.*기사|기사.*알려|기사.*보여|기사.*있어)")
		if (articlePattern.containsMatchIn(normalizedInput)) {
			Log.d("BanyaChat", "Category detected: 기사")
			return true
		}
		
		// 쇼핑 카테고리
		val shoppingPattern = Regex("(쇼핑|구매|가격|판매|할인|구매.*알려|가격.*알려|어디서.*사|어디.*팔|비교|리뷰)")
		if (shoppingPattern.containsMatchIn(normalizedInput)) {
			Log.d("BanyaChat", "Category detected: 쇼핑")
			return true
		}
		
		// 식당/맛집 카테고리
		val restaurantPattern = Regex("(식당|맛집|음식점|레스토랑|카페|식당.*추천|맛집.*추천|맛있는.*곳|먹을.*곳|식사.*곳|점심.*곳|저녁.*곳|근처.*식당|근처.*맛집|주변.*식당|주변.*맛집)")
		if (restaurantPattern.containsMatchIn(normalizedInput)) {
			Log.d("BanyaChat", "Category detected: 식당/맛집")
			return true
		}
		
		// 기타 명시적인 웹 검색 지시
		val explicitSearchPattern = Regex("(검색|찾아|알아봐|검색해|찾아줘|알아봐줘|검색.*해줘|인터넷.*검색|웹.*검색|최신.*정보|최근.*정보)")
		if (explicitSearchPattern.containsMatchIn(normalizedInput)) {
			Log.d("BanyaChat", "Category detected: 명시적인 웹 검색 지시")
			return true
		}
		
		Log.d("BanyaChat", "Category: 일반 질문 (LLM 자체 지식으로 답변)")
		return false
	}

	/**
	 * RAG 시스템: 카테고리에 해당하는 경우에만 웹 검색을 시도하고, 그 외에는 LLM 자체 지식으로 답변
	 */
	private fun processUserInputWithRAG(userInput: String) {
		if (_uiState.value.isGenerating) return
		// Block generation until model load has reached 100
		if (_uiState.value.loadProgress in 0..99) {
			_uiState.value = _uiState.value.addMessage(ChatMessage(text = "모델 로딩 중입니다. 잠시만 기다려 주세요.", isUser = false))
			return
		}

		// 카테고리 확인: 웹 검색이 필요한지 확인
		val needsWebSearch = shouldUseWebSearch(userInput)
		
		if (!needsWebSearch) {
			// 웹 검색이 필요하지 않은 경우: LLM 자체 지식으로 바로 답변
			Log.d("BanyaChat", "RAG: Category does not require web search, using LLM knowledge directly")
			generate()
			return
		}

		// 빈 응답 메시지 추가 및 isGenerating 설정 (생각중... 애니메이션 표시용)
		_uiState.value = _uiState.value.copy(isGenerating = true)
		_uiState.value = _uiState.value.addMessage(ChatMessage(text = "", isUser = false))

		viewModelScope.launch(Dispatchers.IO) {
			try {
				Log.d("BanyaChat", "RAG: Starting web search for question: $userInput")

				// --- 1단계: Tavily API 호출 (웹 검색이 필요한 카테고리일 때만) ---
				val searchRequest = TavilySearchRequest(
					api_key = tavilyApiKey,
					query = userInput, // 사용자 질문을 그대로 검색 쿼리로 사용
					max_results = 1 // 프롬프트 길이 단축을 위해 1개만 반환
				)
				
				val searchResponse = try {
					tavilyApi.search(searchRequest)
				} catch (e: Exception) {
					Log.e("BanyaChat", "RAG: Error calling Tavily API", e)
					null
				}
				
				// --- 2단계: 검색 결과 확인 및 처리 ---
				if (searchResponse != null && searchResponse.isSuccessful && searchResponse.body() != null) {
					val searchResults = searchResponse.body()!!.results
					
					if (searchResults.isNotEmpty()) {
						// 검색 결과가 있는 경우: 검색 결과를 바탕으로 답변 생성
						Log.d("BanyaChat", "RAG: Search completed, found ${searchResults.size} results")
						
						// 검색 결과를 하나의 문자열로 가공 (내용을 300자로 제한하여 프롬프트 길이 단축, 컨텍스트 크기 512 제한 고려)
						val searchContext = searchResults.joinToString("\n\n") { result ->
							val content = result.content ?: "N/A"
							val truncatedContent = if (content.length > 300) content.substring(0, 300) + "..." else content
							"제목: ${result.title ?: "N/A"}\n내용: $truncatedContent"
						}

						// 검색 결과 기반 답변 생성
						// 빈 메시지는 이미 processUserInputWithRAG() 시작 시 추가되었으므로 중복 추가하지 않음
						withContext(Dispatchers.Main) {
							_uiState.value = _uiState.value.copy(isGenerating = true)
						}

						// 웹 검색일 때만 장소/시간 context 포함 (hasSearchResults = true)
						val finalPrompt = createSynthesisPrompt(userInput, searchContext, hasSearchResults = true, includeLocationTime = true)
						
						// RAG를 사용할 때는 createSynthesisPrompt로 만든 프롬프트가 이미 완전한 프롬프트이므로
						// generateStreamWithPrompt를 직접 호출하여 중복을 방지
						// 세션 관리는 프롬프트 자체의 길이로 판단 (이전 대화 기록은 포함하지 않음)
						
						repo.generateStreamWithPrompt(
							prompt = finalPrompt,
							callback = object : TokenCallback {
								override fun onLoadProgress(progress: Int) {}
								override fun onModelMetadata(json: String) {}
								override fun onToken(token: String) {
									mainHandler.post {
										_uiState.value = _uiState.value.appendToLastAssistant(token)
									}
								}
								override fun onCompleted() {
									mainHandler.post {
										_uiState.value = _uiState.value.copy(isGenerating = false)
									}
								}
								override fun onError(message: String) {
									Log.e("BanyaChat", "RAG: Error during final answer generation: $message")
									mainHandler.post {
										_uiState.value = _uiState.value.copy(isGenerating = false)
									}
								}
							}
						)
					} else {
						// 검색 결과가 없는 경우: LLM이 직접 답변 (장소/시간 context 없이)
						Log.d("BanyaChat", "RAG: No search results found, generating answer directly")
						withContext(Dispatchers.Main) {
							generate()
						}
					}
				} else {
					// 검색 실패 또는 응답이 없는 경우: LLM이 직접 답변 (장소/시간 context 없이)
					Log.w("BanyaChat", "RAG: Search failed or no response, generating answer directly")
					withContext(Dispatchers.Main) {
						generate()
					}
				}
			} catch (e: Exception) {
				Log.e("BanyaChat", "RAG: Unexpected error in processUserInputWithRAG", e)
				withContext(Dispatchers.Main) {
					generate()
				}
			}
		}
	}

	/**
	 * 프롬프트: 검색 결과를 바탕으로 최종 답변을 생성하도록 하는 프롬프트
	 * 검색 결과가 없을 경우를 위한 가이드 포함
	 * @param includeLocationTime 웹 검색일 때만 true로 설정하여 장소/시간 정보 포함
	 */
	private fun createSynthesisPrompt(question: String, searchContext: String?, hasSearchResults: Boolean = false, includeLocationTime: Boolean = false): String {
		// Llama 3.1 표준 날짜 형식 (예: "26 Jul 2024")
		val dateFormat = java.text.SimpleDateFormat("dd MMM yyyy", java.util.Locale.ENGLISH)
		val todayDate = dateFormat.format(java.util.Date())
		
		// 웹 검색일 때만 사용할 상세 날짜/시간 및 위치 정보
		val detailedDateFormat = java.text.SimpleDateFormat("yyyy년 MM월 dd일 EEEE HH시 mm분", java.util.Locale.KOREAN)
		val currentDateTime = if (includeLocationTime) detailedDateFormat.format(java.util.Date()) else ""
		val currentLocation = if (includeLocationTime) "서울 강남구" else ""
		
		// 시스템 프롬프트 (Llama 3.1 표준 형식)
		val sb = StringBuilder()
		sb.append("<|begin_of_text|>")
		sb.append("<|start_header_id|>system<|end_header_id|>\n\n")
		
		// Llama 3.1 표준: Cutting Knowledge Date와 Today Date 포함
		sb.append("Cutting Knowledge Date: December 2023\n")
		sb.append("Today Date: $todayDate\n\n")
		
		// 시스템 프롬프트 내용
		val systemPromptContent = if (hasSearchResults && searchContext != null && searchContext.isNotEmpty()) {
			if (includeLocationTime) {
				"""너는 발달장애인의 생활에 도움을 주는 친절한 도우미입니다. 검색 결과를 바탕으로 답변하세요. 질문을 반복하지 마세요. 현재 위치와 날짜/시간 정보를 활용하여 정확하고 관련성 있는 답변을 제공하세요."""
			} else {
				"""너는 발달장애인의 생활에 도움을 주는 친절한 도우미입니다. 검색 결과를 바탕으로 답변하세요. 질문을 반복하지 마세요."""
			}
		} else {
			"너는 발달장애인의 생활에 도움을 주는 친절한 도우미입니다. 질문에 답변하세요. 질문을 반복하지 마세요."
		}
		
		sb.append(systemPromptContent)
		sb.append("<|eot_id|>")
		sb.append("<|start_header_id|>user<|end_header_id|>\n\n")
		
		// 웹 검색일 때만 사용자 질문에 현재 위치와 날짜/시간을 context로 추가
		if (includeLocationTime && currentLocation.isNotEmpty() && currentDateTime.isNotEmpty()) {
			sb.append("[현재 위치: $currentLocation, 현재 날짜/시간: $currentDateTime]\n\n")
		}
		
		if (hasSearchResults && searchContext != null && searchContext.isNotEmpty()) {
			// 프롬프트 길이 최소화: 마크다운 형식 제거, 간결한 형식
			sb.append("$searchContext\n\n$question")
		} else {
			sb.append(question)
		}
		
		sb.append("<|eot_id|>")
		sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

		return sb.toString()
	}

	/**
	 * 세션 관리: 프롬프트 길이를 추정하고 일정 임계값을 초과하면 메시지 리스트를 제한
	 * nCtx=1536이므로 안전하게 60%인 약 920 토큰을 임계값으로 설정
	 * (시스템 프롬프트가 길어졌으므로 더 보수적으로 설정)
	 */
	private fun estimatePromptTokens(messages: List<ChatMessage>): Int {
		// 간단한 추정: 한국어는 대략 1 문자 = 1 토큰, 영어는 4 문자 = 1 토큰
		// 시스템 프롬프트 오버헤드 포함 (날짜/시간, 위치 정보 추가로 약 100 토큰으로 증가)
		var totalChars = 100
		messages.forEach { message ->
			// 메시지 포맷 오버헤드 (role header 등, 약 20 토큰)
			totalChars += 20
			// 메시지 내용
			val text = message.text
			val koreanChars = text.count { it.code in 0xAC00..0xD7A3 || it.code in 0x3131..0x318E }
			val englishChars = text.length - koreanChars
			// 한국어는 1:1, 영어는 4:1 비율로 추정
			totalChars += koreanChars + (englishChars / 4)
		}
		Log.d("BanyaChat", "Session management: Estimated ${totalChars} tokens for ${messages.size} messages")
		return totalChars
	}

	/**
	 * 세션 관리: 프롬프트 길이가 임계값을 초과하면 최근 메시지만 유지
	 * 새 세션을 시작하여 메모리 사용량을 제한
	 */
	private fun limitMessagesForSession(messages: List<ChatMessage>): List<ChatMessage> {
		val maxTokens = 920 // nCtx=1536의 60% (시스템 프롬프트가 길어져서 더 보수적으로 설정)
		
		// 프롬프트 길이 추정
		val estimatedTokens = estimatePromptTokens(messages)
		
		if (estimatedTokens <= maxTokens) {
			// 임계값 이하이면 모든 메시지 유지
			return messages
		}
		
		// 임계값 초과: 새 세션 시작 (최근 4개 메시지 유지: 사용자 2개 + 어시스턴트 2개)
		val recentMessages = messages.takeLast(4)
		
		Log.w("BanyaChat", "Session management: Prompt too long ($estimatedTokens tokens > $maxTokens max), limiting to recent ${recentMessages.size} messages")
		
		return recentMessages
	}
}


