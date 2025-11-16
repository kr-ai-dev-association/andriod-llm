package com.example.llama.data

import retrofit2.http.Body
import retrofit2.http.POST
import retrofit2.Response

/**
 * Tavily API 서비스 인터페이스
 */
interface TavilyApiService {
    @POST("search")
    suspend fun search(@Body request: TavilySearchRequest): Response<TavilySearchResponse>
}

/**
 * Tavily 검색 요청 데이터 클래스
 */
data class TavilySearchRequest(
    val api_key: String,
    val query: String,
    val max_results: Int = 3,
    val search_depth: String = "basic"
)

/**
 * Tavily 검색 응답 데이터 클래스
 */
data class TavilySearchResponse(
    val results: List<TavilySearchResult>
)

/**
 * Tavily 검색 결과 항목
 */
data class TavilySearchResult(
    val title: String?,
    val url: String?,
    val content: String?
)

