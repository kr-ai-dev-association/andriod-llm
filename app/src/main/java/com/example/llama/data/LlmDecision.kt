package com.example.llama.data

/**
 * LLM의 1차 판단 결과를 담는 데이터 클래스
 */
data class LlmDecision(
    val search_needed: Boolean,
    val search_query: String?
)

