#include <jni.h>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <random>
#include <mutex>
#include <android/log.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <regex>
#include <algorithm>

#ifndef LLAMA_STUB_MODE
#define LLAMA_STUB_MODE 0
#endif

#if LLAMA_STUB_MODE
// Stub mode: no llama.cpp includes. Provide minimal behavior.
#else
#include "llama.h"
#endif

static JavaVM* g_JavaVM = nullptr;
static jclass g_CallbackClass = nullptr;
static jmethodID g_OnToken = nullptr;
static jmethodID g_OnCompleted = nullptr;
static jmethodID g_OnError = nullptr;
static jmethodID g_OnLoadProgress = nullptr;
static jmethodID g_OnModelMetadata = nullptr;

static void ensureCallbackRefs(JNIEnv* env, jobject callback);

#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "BanyaChatJNI", __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, "BanyaChatJNI", __VA_ARGS__)

static void llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    const char* tag = "BanyaChatLlama";
    int priority = ANDROID_LOG_DEFAULT;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR:
            priority = ANDROID_LOG_ERROR;
            break;
        case GGML_LOG_LEVEL_WARN:
            priority = ANDROID_LOG_WARN;
            break;
        case GGML_LOG_LEVEL_INFO:
            priority = ANDROID_LOG_INFO;
            break;
        case GGML_LOG_LEVEL_DEBUG:
        default:
            priority = ANDROID_LOG_DEBUG;
            break;
    }
    // llama.cpp log is often multi-line, but android log truncates after newline.
    // So, we print line by line.
    const char *start = text;
    const char *end = start;
    while (*end) {
        while (*end && *end != '\n') {
            end++;
        }
        std::string line(start, end - start);
        __android_log_print(priority, tag, "%s", line.c_str());
        if (*end == '\n') {
            end++;
        }
        start = end;
    }
}

// Special token filtering functions (based on Swift implementation in sp_token_rm.md)
// 1단계: 토큰 레벨 필터링 - 각 토큰이 생성될 때 즉시 필터링
static std::string filterSpecialTokensTokenLevel(const std::string& tokenText) {
    if (tokenText.empty()) {
        return tokenText;
    }
    
    std::string cleaned = tokenText;
    
    // 1.1 완전한 특수 토큰 패턴 제거
    const std::vector<std::string> specialTokenPatterns = {
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|python_tag|>",
        "<|finetune_right_pad_id|>"
    };
    
    for (const auto& pattern : specialTokenPatterns) {
        size_t pos = 0;
        while ((pos = cleaned.find(pattern, pos)) != std::string::npos) {
            cleaned.erase(pos, pattern.length());
        }
    }
    
    // 1.2 Reserved Special Token 패턴 제거 (<|reserved_special_token_\d+|>)
    try {
        std::regex reservedRegex("<\\|reserved_special_token_\\d+\\|>");
        cleaned = std::regex_replace(cleaned, reservedRegex, "");
    } catch (const std::regex_error& e) {
        ALOGE("filterSpecialTokensTokenLevel(): Regex error for reserved pattern: %s", e.what());
    }
    
    // 1.3 부분 특수 토큰 패턴 제거
    // 단독 파이프 제거
    if (cleaned == "|") {
        cleaned = "";
    }
    
    // 단독 '<' 또는 '>' 제거 (특수 토큰의 일부로 생성되는 경우)
    if (cleaned == "<" || cleaned == ">") {
        cleaned = "";
    }
    
    // 단일 문자 'e'가 생성될 때, 이전 텍스트와 합쳐져 "eot>" 패턴이 될 수 있으므로 제거
    // 하지만 여기서는 단독으로는 제거하지 않고, 텍스트 레벨 필터링에서 처리
    // 단, "e" 다음에 "ot>"가 올 가능성이 높으므로 경고만 남김
    if (cleaned == "e") {
        // 단독 "e"는 제거하지 않지만, 텍스트 레벨 필터링에서 "eot>" 패턴을 제거함
        // 여기서는 그대로 통과시킴
    }
    
    // "ot"가 생성될 때, 이전 텍스트의 마지막이 "e"인 경우 "eot" 패턴이 될 수 있으므로 제거
    if (cleaned == "ot") {
        // 단독 "ot"는 제거하지 않지만, 텍스트 레벨 필터링에서 "eot>" 패턴을 제거함
        // 여기서는 그대로 통과시킴
    }
    
    // 특수 토큰 일부 패턴 제거 (eot, eom, _id 등)
    const std::vector<std::string> partialTokenPatterns = {
        "_id",
        "eot",
        "eom",
        "begin_of_text",
        "end_of_text",
        "start_header_id",
        "end_header_id",
        "python_tag",
        "finetune_right_pad_id",
        "eotend_header",  // 로그에서 발견된 패턴
        "end_header",     // eotend_header의 일부
        "start_header",   // start_headersystemend_header의 일부
        "systemend_header"  // 복합 패턴의 일부
    };
    
    for (const auto& pattern : partialTokenPatterns) {
        // 단독으로 나타나는 경우 제거
        if (cleaned == pattern) {
            cleaned = "";
            break;
        }
        // 공백과 함께 나타나는 경우 제거
        size_t pos = 0;
        while ((pos = cleaned.find(" " + pattern, pos)) != std::string::npos) {
            cleaned.erase(pos, pattern.length() + 1);
        }
        pos = 0;
        while ((pos = cleaned.find(pattern + " ", pos)) != std::string::npos) {
            cleaned.erase(pos, pattern.length() + 1);
        }
        // 패턴으로 시작하고 다음 문자가 '>' 또는 '|'인 경우 제거 (예: "eot>", "eom>")
        if (cleaned.length() >= pattern.length() + 1 && 
            cleaned.substr(0, pattern.length()) == pattern) {
            char nextChar = cleaned[pattern.length()];
            if (nextChar == '>' || nextChar == '|' || nextChar == '_') {
                cleaned.erase(0, pattern.length() + 1);
            }
        }
        // 패턴으로 끝나고 이전 문자가 '>' 또는 '|'인 경우 제거 (예: ">eot", "|eot")
        if (cleaned.length() >= pattern.length() + 1 && 
            cleaned.substr(cleaned.length() - pattern.length()) == pattern) {
            char prevChar = cleaned[cleaned.length() - pattern.length() - 1];
            if (prevChar == '>' || prevChar == '|' || prevChar == '_') {
                cleaned.erase(cleaned.length() - pattern.length() - 1);
            }
        }
    }
    // "eot>", "eom>", "_id>" 등의 패턴 직접 제거
    if (cleaned == "eot>" || cleaned == "eom>" || cleaned == "_id>") {
        cleaned = "";
    }
    {
        size_t pos = 0;
        while ((pos = cleaned.find("eot>", pos)) != std::string::npos) {
            cleaned.erase(pos, 4);
        }
    }
    {
        size_t pos = 0;
        while ((pos = cleaned.find("eom>", pos)) != std::string::npos) {
            cleaned.erase(pos, 4);
        }
    }
    {
        size_t pos = 0;
        while ((pos = cleaned.find("_id>", pos)) != std::string::npos) {
            cleaned.erase(pos, 4);
        }
    }
    // "eotend_header>" 패턴 제거 (로그에서 발견된 패턴)
    {
        size_t pos = 0;
        while ((pos = cleaned.find("eotend_header>", pos)) != std::string::npos) {
            cleaned.erase(pos, 14);
        }
    }
    // "eotend_header" 패턴 제거 (">" 없이)
    {
        size_t pos = 0;
        while ((pos = cleaned.find("eotend_header", pos)) != std::string::npos) {
            cleaned.erase(pos, 13);
        }
    }
    
    // '<|' 또는 '|>' 포함 시 제거
    if (cleaned.find("<|") != std::string::npos || cleaned.find("|>") != std::string::npos) {
        size_t pos = 0;
        while ((pos = cleaned.find("<|", pos)) != std::string::npos) {
            cleaned.erase(pos, 2);
        }
        pos = 0;
        while ((pos = cleaned.find("|>", pos)) != std::string::npos) {
            cleaned.erase(pos, 2);
        }
    }
    
    // 공백과 함께 나타나는 '<' 또는 '>' 제거
    size_t pos = 0;
    while ((pos = cleaned.find(" <", pos)) != std::string::npos) {
        cleaned.erase(pos, 2);
    }
    pos = 0;
    while ((pos = cleaned.find("< ", pos)) != std::string::npos) {
        cleaned.erase(pos, 2);
    }
    pos = 0;
    while ((pos = cleaned.find(" >", pos)) != std::string::npos) {
        cleaned.erase(pos, 2);
    }
    pos = 0;
    while ((pos = cleaned.find("> ", pos)) != std::string::npos) {
        cleaned.erase(pos, 2);
    }
    
    // 정규식으로 부분 패턴 제거 (<|.*?|>)
    try {
        std::regex partialRegex("<\\|.*?\\|>");
        cleaned = std::regex_replace(cleaned, partialRegex, "");
    } catch (const std::regex_error& e) {
        ALOGE("filterSpecialTokensTokenLevel(): Regex error for partial pattern: %s", e.what());
    }
    
    // 프롬프트 구조 요소 제거 (header, assistant, end 등)
    const std::vector<std::string> promptStructurePatterns = {
        "_header",
        "start_header",
        "end_header",
        "assistant",
        "user",
        "system"
    };
    
    for (const auto& pattern : promptStructurePatterns) {
        size_t pos = 0;
        while ((pos = cleaned.find(pattern, pos)) != std::string::npos) {
            cleaned.erase(pos, pattern.length());
        }
    }
    
    // 변수 표현식 제거 (NAME, USER_NAME 등 대문자 변수명)
    // 대문자로만 이루어진 단어 제거 (최소 2자 이상, 최대 20자)
    try {
        std::regex varExprRegex("\\b[A-Z]{2,20}\\b");
        cleaned = std::regex_replace(cleaned, varExprRegex, "");
    } catch (const std::regex_error& e) {
        ALOGE("filterSpecialTokensTokenLevel(): Regex error for variable expression pattern: %s", e.what());
    }
    
    // 마크다운 형식 문자 제거 (단독으로 나타나는 경우)
    if (cleaned == "[" || cleaned == "]") {
        cleaned = "";
    }
    
    // 프롬프트 구조 관련 단독 문자 제거
    if (cleaned == ">" || cleaned == "<") {
        cleaned = "";
    }
    
    return cleaned;
}

// 2단계: 텍스트 레벨 필터링 - 누적된 전체 텍스트에서 추가 필터링
static std::string filterSpecialTokensTextLevel(const std::string& text) {
    if (text.empty()) {
        return text;
    }
    
    std::string cleaned = text;
    
    // 2.1 완전한 특수 토큰 패턴 제거 (반복 처리)
    const std::vector<std::string> specialTokenPatterns = {
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|python_tag|>",
        "<|finetune_right_pad_id|>"
    };
    
    size_t previousLength = 0;
    int iterations = 0;
    while (cleaned.length() != previousLength && iterations < 10) {
        previousLength = cleaned.length();
        for (const auto& pattern : specialTokenPatterns) {
            size_t pos = 0;
            while ((pos = cleaned.find(pattern, pos)) != std::string::npos) {
                cleaned.erase(pos, pattern.length());
            }
        }
        iterations++;
    }
    
    // 2.2 Reserved Special Token 패턴 제거
    try {
        std::regex reservedRegex("<\\|reserved_special_token_\\d+\\|>");
        cleaned = std::regex_replace(cleaned, reservedRegex, "");
    } catch (const std::regex_error& e) {
        ALOGE("filterSpecialTokensTextLevel(): Regex error for reserved pattern: %s", e.what());
    }
    
    // 2.3 부분 특수 토큰 패턴 제거 (공격적 필터링)
    bool foundPattern = true;
    int patternIterations = 0;
    while (foundPattern && patternIterations < 10) {
        patternIterations++;
        foundPattern = false;
        
        // 방법 1: "<|" + "|>" 조합 찾기 (뒤에서부터)
        size_t startPos = cleaned.rfind("<|");
        if (startPos != std::string::npos) {
            size_t endPos = cleaned.find("|>", startPos);
            if (endPos != std::string::npos) {
                cleaned.erase(startPos, endPos - startPos + 2);
                foundPattern = true;
                continue;
            }
        }
        
        // 방법 2: 단독 파이프 제거
        if (cleaned.find("|") != std::string::npos && 
            cleaned.find("<|") == std::string::npos && 
            cleaned.find("|>") == std::string::npos) {
            size_t pos = 0;
            while ((pos = cleaned.find("|", pos)) != std::string::npos) {
                cleaned.erase(pos, 1);
                foundPattern = true;
            }
        }
        
        // 방법 3: 정규식으로 부분 패턴 제거 (<|[^|]*|>)
        try {
            std::regex partialRegex("<\\|[^|]*\\|>");
            std::string newCleaned = std::regex_replace(cleaned, partialRegex, "");
            if (newCleaned != cleaned) {
                cleaned = newCleaned;
                foundPattern = true;
            }
        } catch (const std::regex_error& e) {
            ALOGE("filterSpecialTokensTextLevel(): Regex error for partial pattern: %s", e.what());
        }
        
        // 방법 4: 공백과 결합된 패턴 제거
        size_t pos = 0;
        while ((pos = cleaned.find(" <|", pos)) != std::string::npos) {
            cleaned.erase(pos, 3);
            foundPattern = true;
        }
        pos = 0;
        while ((pos = cleaned.find("<| ", pos)) != std::string::npos) {
            cleaned.erase(pos, 3);
            foundPattern = true;
        }
        pos = 0;
        while ((pos = cleaned.find(" |>", pos)) != std::string::npos) {
            cleaned.erase(pos, 3);
            foundPattern = true;
        }
        pos = 0;
        while ((pos = cleaned.find("|> ", pos)) != std::string::npos) {
            cleaned.erase(pos, 3);
            foundPattern = true;
        }
        
        // 방법 5: 단독 '<' 또는 '>' 제거 (특수 토큰의 일부로 생성되는 경우)
        // 줄바꿈은 제외: 줄바꿈 다음에 '<' 또는 '>'가 오는 경우는 제거하지 않음
        pos = 0;
        while ((pos = cleaned.find(" <", pos)) != std::string::npos) {
            // 다음 문자가 '|'가 아니고 공백이나 탭이면 제거 (줄바꿈은 제외)
            if (pos + 2 >= cleaned.length() || cleaned[pos + 2] == ' ' || cleaned[pos + 2] == '\t') {
                cleaned.erase(pos, 2);
                foundPattern = true;
            } else {
                pos++;
            }
        }
        pos = 0;
        while ((pos = cleaned.find("< ", pos)) != std::string::npos) {
            cleaned.erase(pos, 2);
            foundPattern = true;
        }
        pos = 0;
        while ((pos = cleaned.find(" >", pos)) != std::string::npos) {
            cleaned.erase(pos, 2);
            foundPattern = true;
        }
        pos = 0;
        while ((pos = cleaned.find("> ", pos)) != std::string::npos) {
            cleaned.erase(pos, 2);
            foundPattern = true;
        }
        
        // 방법 6: 특수 토큰 일부 패턴 제거 (eot, eom, _id 등)
        const std::vector<std::string> partialTokenPatterns = {
            "_id",
            "eot",
            "eom",
            "begin_of_text",
            "end_of_text",
            "start_header_id",
            "end_header_id",
            "python_tag",
            "finetune_right_pad_id",
            "eotend_header",  // 로그에서 발견된 패턴
            "end_header",     // eotend_header의 일부
            "start_header",   // start_headersystemend_header의 일부
            "systemend_header",  // 복합 패턴의 일부
            "_header",        // 프롬프트 구조 요소
            "assistant",      // 프롬프트 구조 요소
            "user",           // 프롬프트 구조 요소
            "system"          // 프롬프트 구조 요소
        };
        
        for (const auto& pattern : partialTokenPatterns) {
            // 공백과 함께 나타나는 경우 제거
            pos = 0;
            while ((pos = cleaned.find(" " + pattern + " ", pos)) != std::string::npos) {
                cleaned.erase(pos, pattern.length() + 2);
                foundPattern = true;
            }
            pos = 0;
            while ((pos = cleaned.find(" " + pattern, pos)) != std::string::npos) {
                // 다음 문자가 공백, 끝, 또는 특수문자면 제거 (줄바꿈은 제외)
                if (pos + pattern.length() + 1 >= cleaned.length() || 
                    cleaned[pos + pattern.length() + 1] == ' ' || 
                    cleaned[pos + pattern.length() + 1] == '\t' ||
                    cleaned[pos + pattern.length() + 1] == '>' ||
                    cleaned[pos + pattern.length() + 1] == '|') {
                    cleaned.erase(pos, pattern.length() + 1);
                    foundPattern = true;
                } else {
                    pos++;
                }
            }
        }
    }
    
    // 2.4 기타 이상한 패턴 제거 (<[^>]*>)
    try {
        std::regex htmlTagRegex("<[^>]*>");
        cleaned = std::regex_replace(cleaned, htmlTagRegex, "");
    } catch (const std::regex_error& e) {
        ALOGE("filterSpecialTokensTextLevel(): Regex error for HTML tag pattern: %s", e.what());
    }
    
    // 2.5 특수 문자 조합 제거 (^^, ^^^)
    {
        size_t pos = 0;
        while ((pos = cleaned.find("^^^", pos)) != std::string::npos) {
            cleaned.erase(pos, 3);
        }
    }
    {
        size_t pos = 0;
        while ((pos = cleaned.find("^^", pos)) != std::string::npos) {
            cleaned.erase(pos, 2);
        }
    }
    
    // 2.6 "eot>", "eom>", "_id>" 패턴 강력 제거 (텍스트 어디에 있든)
    // 여러 토큰으로 분리되어 생성된 경우를 처리
    {
        size_t pos = 0;
        while ((pos = cleaned.find("eot>", pos)) != std::string::npos) {
            cleaned.erase(pos, 4);
            // pos를 증가시키지 않아서 연속된 패턴도 제거
        }
    }
    {
        size_t pos = 0;
        while ((pos = cleaned.find("eom>", pos)) != std::string::npos) {
            cleaned.erase(pos, 4);
        }
    }
    {
        size_t pos = 0;
        while ((pos = cleaned.find("_id>", pos)) != std::string::npos) {
            cleaned.erase(pos, 4);
        }
    }
    // "eotend_header>" 패턴 강력 제거 (로그에서 발견된 패턴, 14자)
    {
        size_t pos = 0;
        while ((pos = cleaned.find("eotend_header>", pos)) != std::string::npos) {
            cleaned.erase(pos, 14);
        }
    }
    // "eotend_header" 패턴 제거 (">" 없이, 13자)
    {
        size_t pos = 0;
        while ((pos = cleaned.find("eotend_header", pos)) != std::string::npos) {
            cleaned.erase(pos, 13);
        }
    }
    // "eotend_headerstart_headersystemend_header>" 복합 패턴 제거 (로그에서 발견된 패턴)
    {
        size_t pos = 0;
        while ((pos = cleaned.find("eotend_headerstart_headersystemend_header>", pos)) != std::string::npos) {
            cleaned.erase(pos, 44);
        }
    }
    // "start_headersystemend_header>" 패턴 제거
    {
        size_t pos = 0;
        while ((pos = cleaned.find("start_headersystemend_header>", pos)) != std::string::npos) {
            cleaned.erase(pos, 31);
        }
    }
    // "systemend_header>" 패턴 제거
    {
        size_t pos = 0;
        while ((pos = cleaned.find("systemend_header>", pos)) != std::string::npos) {
            cleaned.erase(pos, 17);
        }
    }
    // "end_header>" 패턴 제거
    {
        size_t pos = 0;
        while ((pos = cleaned.find("end_header>", pos)) != std::string::npos) {
            cleaned.erase(pos, 11);
        }
    }
    // "start_header>" 패턴 제거
    {
        size_t pos = 0;
        while ((pos = cleaned.find("start_header>", pos)) != std::string::npos) {
            cleaned.erase(pos, 13);
        }
    }
    // 텍스트 끝에서 "eot", "eom" 패턴 제거 (다음 토큰에서 ">"가 올 수 있음)
    // 단, 줄바꿈(단일 또는 두 줄 바꿈) 다음에 오는 경우는 제외 (줄바꿈은 유지)
    if (cleaned.length() >= 3) {
        std::string suffix = cleaned.substr(cleaned.length() - 3);
        if (suffix == "eot" || suffix == "eom") {
            // 줄바꿈(단일 또는 두 줄 바꿈) 다음에 오는 경우는 제외
            bool shouldRemove = true;
            if (cleaned.length() > 3) {
                char prevChar = cleaned[cleaned.length() - 4];
                // 단일 줄바꿈 또는 두 줄 바꿈 다음에 오는 경우는 제외
                if (prevChar == '\n') {
                    // 두 줄 바꿈인지 확인
                    if (cleaned.length() > 4 && cleaned[cleaned.length() - 5] == '\n') {
                        shouldRemove = false; // 두 줄 바꿈 다음
                    } else {
                        shouldRemove = false; // 단일 줄바꿈 다음
                    }
                }
            }
            if (shouldRemove) {
                cleaned.erase(cleaned.length() - 3);
            }
        }
    }
    // 텍스트 끝에서 단일 문자 "e" 확인 (다음 토큰에서 "ot>"가 올 수 있음)
    // 하지만 여기서는 제거하지 않고, 다음 토큰이 추가될 때 텍스트 레벨 필터링에서 처리
    // 단, 줄바꿈(단일 또는 두 줄 바꿈) 다음에 오는 경우는 제외
    
    // 프롬프트 구조 요소 제거 (header, assistant, end 등)
    const std::vector<std::string> promptStructurePatterns = {
        "_header",
        "start_header",
        "end_header",
        "assistant",
        "user",
        "system"
    };
    
    for (const auto& pattern : promptStructurePatterns) {
        size_t pos = 0;
        while ((pos = cleaned.find(pattern, pos)) != std::string::npos) {
            cleaned.erase(pos, pattern.length());
        }
    }
    
    // 변수 표현식 제거 (NAME, USER_NAME 등 대문자 변수명)
    // 대문자로만 이루어진 단어 제거 (최소 2자 이상, 최대 20자)
    try {
        std::regex varExprRegex("\\b[A-Z]{2,20}\\b");
        cleaned = std::regex_replace(cleaned, varExprRegex, "");
    } catch (const std::regex_error& e) {
        ALOGE("filterSpecialTokensTextLevel(): Regex error for variable expression pattern: %s", e.what());
    }
    
    // 마크다운 형식 문자 제거 (단독으로 나타나는 경우)
    // "[질문]", "[답변]" 같은 패턴 제거
    try {
        std::regex markdownPatternRegex("\\[질문\\]|\\[답변\\]|\\[검색 결과\\]");
        cleaned = std::regex_replace(cleaned, markdownPatternRegex, "");
    } catch (const std::regex_error& e) {
        ALOGE("filterSpecialTokensTextLevel(): Regex error for markdown pattern: %s", e.what());
    }
    
    // 사용자 질문 패턴 제거 (예: "대한민국 대통령은 누구야?" 같은 질문이 답변에 포함되는 경우)
    // 하지만 이것은 프롬프트 구조 문제이므로, 시스템 프롬프트에서 명시적으로 지시하는 것이 더 나음
    
    return cleaned;
}

struct LlamaCtx {
#if LLAMA_STUB_MODE
    int dummy;
#else
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
#endif
    std::atomic<bool> stopRequested = false;
    std::mutex ctx_mutex;  // Mutex to protect llama_context access
};

struct LoadProgressContext {
    jobject callback = nullptr;
    std::atomic<bool> completed = false;  // Track if loading is completed
};

// Forward declarations
static JNIEnv* attachThread();
static void detachThread();

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
	g_JavaVM = vm;
	llama_log_set(llama_log_callback, nullptr); // Set log callback
	llama_backend_init(); // false = no NUMA
	ALOGD("JNI_OnLoad: LLAMA_STUB_MODE=%d", LLAMA_STUB_MODE);
	return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_init(
        JNIEnv* env, jobject /*thiz*/,
        jstring jModelPath,
        jint nCtx, jint nThreads, jint nBatch, jint nGpuLayers,
        jboolean useMmap, jboolean useMlock, jint seed,
        jobject callback) {
    auto* handle = new LlamaCtx();
#if LLAMA_STUB_MODE
    (void) jModelPath;
    (void) nCtx; (void) nThreads; (void) nBatch; (void) nGpuLayers;
    (void) useMmap; (void) useMlock; (void) seed; (void) callback;
    ALOGD("init(): STUB build active. Returning dummy handle.");
    return reinterpret_cast<jlong>(handle);
#else
    if (callback) {
        ensureCallbackRefs(env, callback);
    }

    jobject callbackGlobal = callback ? env->NewGlobalRef(callback) : nullptr;
    // Allocate progress context on heap so it persists during async model loading
    auto* progressCtx = new LoadProgressContext{callbackGlobal};

    auto progressFn = [](float progress, void * user) -> bool {
        auto* ctx = static_cast<LoadProgressContext*>(user);
        if (!ctx) {
            return true;
        }
        // If loading is already completed, don't do anything
        if (ctx->completed.load()) {
            return true;
        }
        // Just track progress, don't call callback from background thread
        // This prevents JNI callback conflicts when callback object is replaced
        jint percent = static_cast<jint>(std::lround(progress * 100.0f));
        if (percent < 0) percent = 0;
        if (percent > 100) percent = 100;
        
        // Mark as completed when reaching 100%
        if (percent >= 100) {
            ctx->completed.store(true);
        }
        
        // Log progress but don't call callback from background thread
        // Callback will be called from main thread after model loading completes
        ALOGD("progressFn(): progress=%d%% (callback disabled to prevent JNI conflicts)", (int)percent);
        return true;
    };

    const char* path = env->GetStringUTFChars(jModelPath, nullptr);
    ALOGD("init(): modelPath=%s nCtx=%d nThreads=%d nBatch=%d nGpuLayers=%d useMmap=%d useMlock=%d seed=%d",
          path ? path : "(null)", (int)nCtx, (int)nThreads, (int)nBatch, (int)nGpuLayers, (int)useMmap, (int)useMlock, (int)seed);

    llama_model_params mparams = llama_model_default_params();
    // Optimized Vulkan settings for Adreno 830
    // Reduced GPU layers to 29 to test if 30th layer causes crashes
    if (nGpuLayers == -1) {
        // Keep at 29 layers - 30+ causes Vulkan errors on this device
        mparams.n_gpu_layers = 29;
    } else {
    mparams.n_gpu_layers = nGpuLayers;
    }
    mparams.use_mmap = useMmap;
    mparams.use_mlock = useMlock;
    // Q4_0 doesn't require extra buffers
    mparams.use_extra_bufts = false;
    // Use DEVICE_LOCAL memory (no_host=true) for better GPU performance and stability
    // This ensures model weights are stored in GPU memory, reducing host-device transfers
    mparams.no_host = true;  // DEVICE_LOCAL memory for GPU weights
    mparams.progress_callback = progressFn;
    mparams.progress_callback_user_data = progressCtx;

    ALOGD("init(): Calling llama_model_load_from_file with n_gpu_layers=%d, use_extra_bufts=%d, no_host=%d...",
          (int)mparams.n_gpu_layers, (int)mparams.use_extra_bufts, (int)mparams.no_host);
    llama_model* model = llama_model_load_from_file(path, mparams);
    ALOGD("init(): llama_model_load_from_file returned. model is %s", model ? "valid" : "null");

    env->ReleaseStringUTFChars(jModelPath, path);

    if (!model) {
        ALOGE("init(): llama_load_model_from_file failed");
        if (callbackGlobal && g_OnError) {
            jstring err = env->NewStringUTF("모델을 로드할 수 없습니다.");
            // Safely call error callback with type validation
            if (g_CallbackClass && !env->IsInstanceOf(callbackGlobal, g_CallbackClass)) {
                ALOGE("init(): Error callback object is not an instance of TokenCallback - skipping");
            } else {
            env->CallVoidMethod(callbackGlobal, g_OnError, err);
                if (env->ExceptionCheck()) {
                    ALOGE("init(): Exception in error callback - clearing");
                    env->ExceptionClear();
                }
            }
            env->DeleteLocalRef(err);
        }
        if (callbackGlobal) env->DeleteGlobalRef(callbackGlobal);
        delete progressCtx;  // Clean up progress context on error
        delete handle;
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = nCtx;
    cparams.n_threads = nThreads;
    // Use the same number of threads for batched prompt processing to speed up prefill
    cparams.n_threads_batch = nThreads;
    cparams.n_batch = nBatch;
    // Optimized: n_ubatch=16 for n_batch=128 (128/16=8, no remainder)
    // KV cache is now on GPU, so larger batch size should improve performance
    cparams.n_ubatch = 16;  // Optimized for n_batch=128
    // STABLE: V-Cache를 F16으로 유지 (Q4_0 패딩 로직에 문제가 있어 Logit NaN 발생)
    // K-Cache는 Q4_0으로 유지하여 VRAM 절약
    // V-Cache F16 + K-Cache Q4_0 조합으로 안정성과 성능을 모두 확보
    // n_batch=256과 chunk 크기를 일치시켜 GPU 메모리 효율성 극대화
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    cparams.type_k = GGML_TYPE_Q4_0;  // K-Cache: Q4_0 (VRAM 절약)
    cparams.type_v = GGML_TYPE_F16;   // V-Cache: F16 (안정성 확보, Logit NaN 방지)

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        ALOGE("init(): llama_new_context_with_model failed - possible VRAM shortage for KV Cache");
        if (callbackGlobal && g_OnError) {
            jstring err = env->NewStringUTF("컨텍스트 초기화에 실패했습니다. VRAM 부족일 수 있습니다.");
            // Safely call error callback with type validation
            if (g_CallbackClass && !env->IsInstanceOf(callbackGlobal, g_CallbackClass)) {
                ALOGE("init(): Error callback object is not an instance of TokenCallback - skipping");
            } else {
            env->CallVoidMethod(callbackGlobal, g_OnError, err);
                if (env->ExceptionCheck()) {
                    ALOGE("init(): Exception in error callback - clearing");
                    env->ExceptionClear();
                }
            }
            env->DeleteLocalRef(err);
        }
        llama_model_free(model);
        if (callbackGlobal) env->DeleteGlobalRef(callbackGlobal);
        delete progressCtx;  // Clean up progress context on error
        delete handle;
        return 0;
    }

    // Mark progress as completed before calling final callback
    if (progressCtx) {
        progressCtx->completed.store(true);
    }
    // Safely call final progress callback with type validation
    if (callbackGlobal && g_OnLoadProgress) {
        // Verify callback type before calling to prevent JNI errors
        if (g_CallbackClass && !env->IsInstanceOf(callbackGlobal, g_CallbackClass)) {
            ALOGE("init(): Final callback object is not an instance of TokenCallback - skipping");
        } else {
        env->CallVoidMethod(callbackGlobal, g_OnLoadProgress, 100);
            if (env->ExceptionCheck()) {
                ALOGE("init(): Exception in final progress callback - clearing");
                env->ExceptionClear();
            }
        }
    }

    if (callbackGlobal && g_OnModelMetadata) {
        auto metaValue = [&](const char * key) -> std::string {
            char buf[512];
            int32_t len = llama_model_meta_val_str(model, key, buf, sizeof(buf));
            if (len >= 0) {
                return std::string(buf, len);
            }
            return "";
        };

        nlohmann::json meta;
        std::string name = metaValue("general.name");
        std::string quant = metaValue("general.file_type");
        std::string sizeLabel = metaValue("general.size_label");
        std::string contextStr = metaValue("general.context_length");

        meta["name"] = name.empty() ? "(unknown)" : name;
        meta["quantization"] = quant.empty() ? "unknown" : quant;
        meta["size_label"] = sizeLabel.empty() ? "unknown" : sizeLabel;
        meta["context_length"] = contextStr.empty() ? static_cast<int>(nCtx) : std::atoi(contextStr.c_str());

        std::string metaDump = meta.dump();
        jstring metaJson = env->NewStringUTF(metaDump.c_str());
        // Safely call metadata callback with type validation
        if (g_CallbackClass && !env->IsInstanceOf(callbackGlobal, g_CallbackClass)) {
            ALOGE("init(): Metadata callback object is not an instance of TokenCallback - skipping");
        } else {
        env->CallVoidMethod(callbackGlobal, g_OnModelMetadata, metaJson);
            if (env->ExceptionCheck()) {
                ALOGE("init(): Exception in metadata callback - clearing");
                env->ExceptionClear();
            }
        }
        env->DeleteLocalRef(metaJson);
    }

    handle->model = model;
    handle->ctx = ctx;
    ALOGD("init(): success, handle=%p", (void*)handle);

    // Don't call progress callback from JNI to avoid callback conflicts
    // Kotlin layer will detect model loading completion by checking if handle != 0
    // and update progress bar accordingly

    // Clean up progress context (callback is stored in handle if needed later)
    // Note: progressCtx->callback is the same as callbackGlobal, so we'll delete it below
    delete progressCtx;

    if (callbackGlobal) {
        env->DeleteGlobalRef(callbackGlobal);
    }
    return reinterpret_cast<jlong>(handle);
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_free(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong h) {
    auto* handle = reinterpret_cast<LlamaCtx*>(h);
    if (!handle) return;
#if LLAMA_STUB_MODE
#else
    if (handle->ctx) llama_free(handle->ctx);
    if (handle->model) llama_model_free(handle->model);
#endif
    delete handle;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_completionStop(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong h) {
    auto* handle = reinterpret_cast<LlamaCtx*>(h);
    if (!handle) return;
    handle->stopRequested = true;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_clearKvCache(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong h) {
    auto* handle = reinterpret_cast<LlamaCtx*>(h);
    if (!handle || !handle->ctx) {
        ALOGE("clearKvCache(): handle or ctx is null");
        return;
    }
#if LLAMA_STUB_MODE
    ALOGD("clearKvCache(): STUB build active. No-op.");
#else
    std::lock_guard<std::mutex> lock(handle->ctx_mutex);
    ALOGD("clearKvCache(): Clearing KV cache for new session");
    llama_memory_t mem = llama_get_memory(handle->ctx);
    if (mem) {
        llama_memory_clear(mem, true);  // Clear KV cache data
        ALOGD("clearKvCache(): KV cache cleared successfully");
    } else {
        ALOGE("clearKvCache(): Failed to get memory from context");
    }
#endif
}

static void ensureCallbackRefs(JNIEnv* env, jobject callback) {
    if (!g_CallbackClass) {
        jclass local = env->GetObjectClass(callback);
        g_CallbackClass = reinterpret_cast<jclass>(env->NewGlobalRef(local));
        env->DeleteLocalRef(local);
        g_OnToken = env->GetMethodID(g_CallbackClass, "onToken", "(Ljava/lang/String;)V");
        g_OnCompleted = env->GetMethodID(g_CallbackClass, "onCompleted", "()V");
        g_OnError = env->GetMethodID(g_CallbackClass, "onError", "(Ljava/lang/String;)V");
        g_OnLoadProgress = env->GetMethodID(g_CallbackClass, "onLoadProgress", "(I)V");
        g_OnModelMetadata = env->GetMethodID(g_CallbackClass, "onModelMetadata", "(Ljava/lang/String;)V");
        ALOGD("ensureCallbackRefs(): methods cached");
    }
}

static JNIEnv* attachThread() {
	if (!g_JavaVM) return nullptr;
	JNIEnv* env = nullptr;
	jint res = g_JavaVM->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
	if (res == JNI_OK) return env;
	#if defined(__ANDROID__)
	if (g_JavaVM->AttachCurrentThread(&env, nullptr) != 0) return nullptr;
	#else
	if (g_JavaVM->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr) != 0) return nullptr;
	#endif
	return env;
}

static void detachThread() {
	if (!g_JavaVM) return;
	g_JavaVM->DetachCurrentThread();
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_completionStart(
        JNIEnv* env, jobject /*thiz*/, jlong h,
        jstring jPrompt, jint numPredict, jfloat temperature, jfloat topP, jint topK,
        jfloat repeatPenalty, jint repeatLastN,
        jobjectArray jStopSequences, jobject callback) {
    auto* handle = reinterpret_cast<LlamaCtx*>(h);
    if (!handle) return;
    ensureCallbackRefs(env, callback);

    jobject gCallback = env->NewGlobalRef(callback);
    const char* prompt = env->GetStringUTFChars(jPrompt, nullptr);
    std::string promptStr(prompt ? prompt : "");
    env->ReleaseStringUTFChars(jPrompt, prompt);

    // Defaults if invalid values are passed
    // n_predict를 1024로 넉넉하게 설정하여 EOT 토큰이 생성되지 않는 경우를 대비한 안전망
    // EOT 토큰(128009)이 최우선 정지 신호이므로, n_predict는 최종 안전장치 역할만 수행
    // EOT 토큰이 제대로 작동한다면 모델은 n_predict에 도달하기 훨씬 전에 스스로 멈춤
    int n_predict = (numPredict > 0) ? numPredict : 1024;
    float temp = (temperature > 0.0f) ? temperature : 0.7f;  // Llama 3.1 기본값에 가까운 값
    float top_p = (topP > 0.0f) ? topP : 0.9f;  // Llama 3.1 권장값
    int top_k = (topK > 0) ? topK : 40;  // Llama 3.1 기본값
    // Repeat Penalty를 1.2로 설정하여 반복을 줄이면서도 대화 품질 유지
    float rep_penalty = (repeatPenalty > 0.0f) ? repeatPenalty : 1.2f;
    int rep_last_n = (repeatLastN > 0) ? repeatLastN : 256;
    std::vector<std::string> stops;
    if (jStopSequences) {
        jsize len = env->GetArrayLength(jStopSequences);
        for (jsize i = 0; i < len; ++i) {
            jstring s = (jstring) env->GetObjectArrayElement(jStopSequences, i);
            const char* cs = env->GetStringUTFChars(s, nullptr);
            stops.emplace_back(cs ? cs : "");
            env->ReleaseStringUTFChars(s, cs);
            env->DeleteLocalRef(s);
        }
    }

    handle->stopRequested = false;

    std::thread worker([gCallback, handle, promptStr, n_predict, temp, top_p, top_k, rep_penalty, rep_last_n, stops]() {
        ALOGD("completionStart(): worker thread started");
		JNIEnv* threadEnv = attachThread();
		if (!threadEnv) {
            ALOGE("completionStart(): could not attach thread to JVM");
			return;
		}
        ALOGD("completionStart(): worker thread attached to JVM");

        struct CallbackGuard {
            JNIEnv* env;
            jobject ref;
            bool shouldDelete;
            CallbackGuard() : env(nullptr), ref(nullptr), shouldDelete(true) {}
            ~CallbackGuard() {
                if (env && ref && shouldDelete) {
                    // Only delete if thread is still attached (env is valid)
                    // Note: We can't check if thread is attached safely, so we rely on shouldDelete flag
                    // which should be set to false before detachThread() is called
                    env->DeleteGlobalRef(ref);
                }
            }
        } guard;
        guard.env = threadEnv;
        guard.ref = gCallback;

        // Lock mutex to prevent concurrent access to llama_context
        ALOGD("completionStart(): Acquiring mutex lock");
        std::lock_guard<std::mutex> lock(handle->ctx_mutex);
        ALOGD("completionStart(): Mutex lock acquired");

        llama_context* ctx = handle->ctx;
        if (!ctx) {
            ALOGE("completionStart(): ctx is null");
            jstring err = threadEnv->NewStringUTF("Context is null");
            if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
            threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                if (threadEnv->ExceptionCheck()) {
                    ALOGE("completionStart(): Exception in error callback - clearing");
                    threadEnv->ExceptionClear();
                }
            } else {
                ALOGE("completionStart(): Callback validation failed - skipping error callback");
            }
            threadEnv->DeleteLocalRef(err);
            // Prevent CallbackGuard from trying to delete gCallback after detachThread()
            if (guard.env && guard.ref) {
                guard.env->DeleteGlobalRef(guard.ref);
                guard.ref = nullptr;
            }
            guard.shouldDelete = false;
            detachThread();
            return;
        }
        
        // Get actual n_batch from context for optimal chunk size
        // Use llama_n_batch() to get the actual batch size from context
        uint32_t chunk_size = llama_n_batch(ctx);
        ALOGD("completionStart(): Retrieved n_batch=%u from context for chunk size", chunk_size);

        llama_model* model = handle->model;
        if (!model) {
            ALOGE("completionStart(): model is null");
            jstring err = threadEnv->NewStringUTF("Model is null");
            if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
            threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                if (threadEnv->ExceptionCheck()) {
                    ALOGE("completionStart(): Exception in error callback - clearing");
                    threadEnv->ExceptionClear();
                }
            } else {
                ALOGE("completionStart(): Callback validation failed - skipping error callback");
            }
            threadEnv->DeleteLocalRef(err);
            // Prevent CallbackGuard from trying to delete gCallback after detachThread()
            if (guard.env && guard.ref) {
                guard.env->DeleteGlobalRef(guard.ref);
                guard.ref = nullptr;
            }
            guard.shouldDelete = false;
            detachThread();
            return;
        }
        
        const struct llama_vocab * vocab = llama_model_get_vocab(model);

        // tokenize prompt
        std::vector<llama_token> prompt_tokens;
        prompt_tokens.resize(promptStr.size() + 16);
        ALOGD("completionStart(): tokenizing prompt...");
        ALOGD("completionStart(): prompt length=%zu, first 200 chars: %.200s", promptStr.length(), promptStr.c_str());
        // Check if prompt already starts with BOS token
        bool has_bos = (promptStr.length() > 0 && promptStr.find("<|begin_of_text|>") == 0);
        ALOGD("completionStart(): has_bos=%d, add_bos=%d", has_bos ? 1 : 0, !has_bos ? 1 : 0);
        int n_tokens = llama_tokenize(
            vocab,
            promptStr.c_str(),
            (int32_t)promptStr.size(),
            prompt_tokens.data(),
            (int32_t)prompt_tokens.size(),
            !has_bos, // add_bos only if not already present
            false   // special
        );

        if (n_tokens < 0) {
            ALOGE("completionStart(): llama_tokenize failed");
            jstring err = threadEnv->NewStringUTF("Tokenization failed");
            if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
            threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                if (threadEnv->ExceptionCheck()) {
                    ALOGE("completionStart(): Exception in error callback - clearing");
                    threadEnv->ExceptionClear();
                }
            } else {
                ALOGE("completionStart(): Callback validation failed - skipping error callback");
            }
            threadEnv->DeleteLocalRef(err);
            detachThread();
            return;
        }
        prompt_tokens.resize(n_tokens);
        ALOGD("completionStart(): tokenized prompt into %d tokens", n_tokens);
        // Log first and last few tokens for debugging
        if (n_tokens > 0) {
            ALOGD("completionStart(): First 5 tokens: %d %d %d %d %d", 
                  (int)prompt_tokens[0], 
                  n_tokens > 1 ? (int)prompt_tokens[1] : -1,
                  n_tokens > 2 ? (int)prompt_tokens[2] : -1,
                  n_tokens > 3 ? (int)prompt_tokens[3] : -1,
                  n_tokens > 4 ? (int)prompt_tokens[4] : -1);
            if (n_tokens >= 5) {
                ALOGD("completionStart(): Last 5 tokens: %d %d %d %d %d", 
                      (int)prompt_tokens[n_tokens-5], 
                      (int)prompt_tokens[n_tokens-4],
                      (int)prompt_tokens[n_tokens-3],
                      (int)prompt_tokens[n_tokens-2],
                      (int)prompt_tokens[n_tokens-1]);
            }
            // Log actual token text for first and last tokens to verify prompt
            char first_token_text[256];
            int first_len = llama_token_to_piece(vocab, prompt_tokens[0], first_token_text, sizeof(first_token_text), false, false);
            if (first_len > 0 && first_len < 256) {
                first_token_text[first_len] = '\0';
                ALOGD("completionStart(): First token text: '%s' (id=%d)", first_token_text, (int)prompt_tokens[0]);
            }
            char last_token_text[256];
            int last_len = llama_token_to_piece(vocab, prompt_tokens[n_tokens-1], last_token_text, sizeof(last_token_text), false, false);
            if (last_len > 0 && last_len < 256) {
                last_token_text[last_len] = '\0';
                ALOGD("completionStart(): Last token text: '%s' (id=%d)", last_token_text, (int)prompt_tokens[n_tokens-1]);
            }
        }

        // Create sampler chain matching iOS implementation (Llama 3.1 optimized)
        auto sparams = llama_sampler_chain_default_params();
        llama_sampler * smpl = llama_sampler_chain_init(sparams);
        
        // Llama 3.1 optimized sampling chain (matching iOS implementation)
        // Order is critical: Top-P -> Min-P -> Temperature -> Repeat Penalty -> Dist
        
        // 1. Top-K (0 = disabled, Llama 3.1 recommendation: use Top-P + Min-P instead)
        // Top-K is disabled for Llama 3.1 as it works better with Top-P + Min-P combination
        if (top_k > 0) {
            ALOGD("completionStart(): WARNING: Top-K is enabled (%d) but Llama 3.1 recommends Top-K=0 (use Top-P + Min-P)", top_k);
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        }
        
        // 2. Top-P (Nucleus Sampling) - Llama 3.1 recommended: 0.9
        // Keeps tokens with cumulative probability up to top_p
        if (top_p > 0.0f && top_p < 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        } else {
            ALOGD("completionStart(): WARNING: Top-P is disabled or invalid (%.2f), using default 0.9", top_p);
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
        }
        
        // 3. Min-P (Llama 3.1 key setting - exclude low probability tokens)
        // Removes tokens with probability less than min_p * max_probability
        // This is critical for Llama 3.1 quality
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
        
        // 4. Temperature - Llama 3.1 recommended: 0.6
        // Lower temperature = more deterministic, less repetition
        if (temp > 0.0f && temp != 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        } else {
            ALOGD("completionStart(): WARNING: Temperature is disabled or invalid (%.2f), using default 0.6", temp);
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.6f));
        }
        
        // 5. Repeat Penalty (with freq_penalty and presence_penalty)
        // 반복 방지 강화: repeat_penalty 증가, last_n 증가, freq/presence_penalty 증가
        // freq_penalty와 presence_penalty를 0.15로 증가하여 반복을 더 강하게 억제
        if (rep_last_n != 0 && rep_penalty > 0.0f && rep_penalty != 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(rep_last_n, rep_penalty, 0.15f, 0.15f));
        } else {
            ALOGD("completionStart(): WARNING: Repeat penalty is disabled or invalid, using defaults");
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(128, 1.25f, 0.15f, 0.15f));
        }
        
        // 6. Dist sampling (final token selection)
        // Random seed for diversity
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(static_cast<uint32_t>(std::random_device{}())));
        
        ALOGD("completionStart(): Sampling chain configured: Top-K=%d, Top-P=%.2f, Min-P=0.05, Temp=%.2f, Repeat=%d/%.2f",
              top_k, top_p, temp, rep_last_n, rep_penalty);
        ALOGD("completionStart(): Sampling parameters: top_k=%d, top_p=%.3f, temp=%.3f, rep_penalty=%.3f, rep_last_n=%d",
              top_k, top_p, temp, rep_penalty, rep_last_n);

        // Main generation loop - evaluate prompt and generate tokens in streaming mode
        int n_past = 0;
        int n_gen = 0;
        std::string generated;

        // Decode prompt in chunks - use n_batch size for maximum performance
        ALOGD("completionStart(): evaluating prompt...");
        // Performance measurement: Record start time
        auto prompt_eval_start = std::chrono::high_resolution_clock::now();
        ALOGD("completionStart(): Prompt evaluation started, n_tokens=%d", n_tokens);
        
        // Performance optimization: Dynamic chunk size based on prompt length
        // Optimized: Chunk size matches n_batch for maximum GPU memory efficiency
        // This ensures llama_decode uses the allocated VRAM (n_batch) most efficiently
        auto calculate_optimal_chunk_size = [](uint32_t n_tokens, uint32_t n_batch) -> uint32_t {
            if (n_tokens <= n_batch) {
                return n_tokens;  // Process all tokens at once if within n_batch limit
            } else {
                return n_batch;  // Use full n_batch size for optimal GPU utilization
            }
        };
        
        // Use dynamic chunk size for optimal performance
        const uint32_t chunk = calculate_optimal_chunk_size(n_tokens, chunk_size);
        ALOGD("completionStart(): Using dynamic chunk size=%u (n_batch=%u, n_tokens=%d) for prompt evaluation", chunk, chunk_size, n_tokens);
        uint32_t context_size = llama_n_ctx(ctx);
        
        // Evaluate prompt tokens and generate tokens in streaming mode
        // Optimize chunking for speed - use full n_batch size
        // Performance optimization: Reuse batch instead of initializing/freeing for each chunk
        llama_batch batch = llama_batch_init(chunk, 0, 1);
        if (!batch.token || !batch.seq_id || !batch.n_seq_id || !batch.logits) {
            ALOGE("completionStart(): llama_batch_init() returned invalid batch");
            jstring err = threadEnv->NewStringUTF("Failed to initialize batch");
            if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
                threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                if (threadEnv->ExceptionCheck()) {
                    ALOGE("completionStart(): Exception in error callback - clearing");
                    threadEnv->ExceptionClear();
                }
            } else {
                ALOGE("completionStart(): Callback validation failed - skipping error callback");
            }
            threadEnv->DeleteLocalRef(err);
            llama_sampler_free(smpl);
            if (guard.env && guard.ref) {
                guard.env->DeleteGlobalRef(guard.ref);
                guard.ref = nullptr;
            }
            guard.shouldDelete = false;
            detachThread();
            return;
        }
        
        for (int cur = 0; cur < n_tokens; ) {
            int remaining = n_tokens - cur;
            int n_cur = std::min(static_cast<int>(chunk), remaining);
            
            // Limit maximum chunk size to n_batch for stability
            // With V-Cache Q4_0 and full GPU offload, n_batch size chunks are stable
            if (n_cur > static_cast<int>(chunk)) {
                ALOGD("completionStart(): Chunk size %d exceeds maximum %u, limiting to %u", n_cur, chunk, chunk);
                n_cur = static_cast<int>(chunk);
            }
            
            ALOGD("completionStart(): llama_decode chunk start cur=%d n_cur=%d (remaining after=%d)", cur, n_cur, n_tokens - cur - n_cur);
            
            // Reuse batch: only adjust size instead of reinitializing
            batch.n_tokens = n_cur;
            // Set pos to nullptr to let llama_batch_allocr::init() calculate positions from memory
            // This ensures positions are consistent with the KV cache state
            free(batch.pos);
            batch.pos = nullptr;
            
            ALOGD("completionStart(): Batch reused, filling tokens (n_past=%d)", n_past);
            for (int j = 0; j < n_cur; ++j) {
                batch.token   [j] = prompt_tokens[cur + j];
                if (batch.seq_id[j]) {
                batch.seq_id  [j][0] = 0;
                } else {
                    ALOGE("completionStart(): batch.seq_id[%d] is null!", j);
                    llama_batch_free(batch);
                    llama_sampler_free(smpl);
                    detachThread();
                    return;
                }
                batch.n_seq_id[j] = 1;
                batch.logits  [j] = false;
            }
            // NOTE: Do NOT enable logits in the last chunk during prompt evaluation
            // This causes decode to hang on Vulkan backend. Instead, we'll decode the last token
            // separately after prompt evaluation is complete to get logits.
            // Keep all logits disabled during prompt evaluation to avoid Vulkan backend issues
            // We'll handle logits separately after prompt evaluation
            
            bool is_last_chunk = (cur + n_cur == n_tokens);
            ALOGD("completionStart(): Batch filled, calling llama_decode() for chunk cur=%d n_cur=%d (total tokens=%d, is_last_chunk=%d)", 
                  cur, n_cur, n_tokens, is_last_chunk ? 1 : 0);
            ALOGD("completionStart(): About to call llama_decode() for prompt evaluation chunk cur=%d n_cur=%d", cur, n_cur);
            ALOGD("completionStart(): Context size=%u, n_past=%d, n_cur=%d, total will be n_past+n_cur=%d", 
                  context_size, n_past, n_cur, n_past + n_cur);
            
            // Check if we're exceeding context size
            if (n_past + n_cur > static_cast<int>(context_size)) {
                ALOGE("completionStart(): ERROR: n_past (%d) + n_cur (%d) = %d exceeds context_size (%u)!", 
                      n_past, n_cur, n_past + n_cur, context_size);
                jstring err = threadEnv->NewStringUTF("Prompt exceeds context size");
                if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
                    threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                    if (threadEnv->ExceptionCheck()) {
                        ALOGE("completionStart(): Exception in error callback - clearing");
                        threadEnv->ExceptionClear();
                    }
                }
                threadEnv->DeleteLocalRef(err);
                llama_batch_free(batch);
                llama_sampler_free(smpl);
                if (guard.env && guard.ref) {
                    guard.env->DeleteGlobalRef(guard.ref);
                    guard.ref = nullptr;
                }
                guard.shouldDelete = false;
                detachThread();
                return;
            }
            
            int decode_result = llama_decode(ctx, batch);
            ALOGD("completionStart(): llama_decode() returned %d for chunk cur=%d n_cur=%d (is_last_chunk=%d)", 
                  decode_result, cur, n_cur, is_last_chunk ? 1 : 0);
            if (decode_result != 0) {
                ALOGE("completionStart(): llama_decode() failed at chunk cur=%d n_cur=%d", cur, n_cur);
                jstring err = threadEnv->NewStringUTF("Failed to decode prompt");
                if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
                threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                    if (threadEnv->ExceptionCheck()) {
                        ALOGE("completionStart(): Exception in error callback - clearing");
                        threadEnv->ExceptionClear();
                    }
                } else {
                    ALOGE("completionStart(): Callback validation failed - skipping error callback");
                }
                threadEnv->DeleteLocalRef(err);
                llama_batch_free(batch);
                llama_sampler_free(smpl);
                // Prevent CallbackGuard from trying to delete gCallback after detachThread()
                // We need to delete it manually before detaching
                if (guard.env && guard.ref) {
                    guard.env->DeleteGlobalRef(guard.ref);
                    guard.ref = nullptr;
                }
                guard.shouldDelete = false;  // Prevent double deletion
                detachThread();
                return;
            }
            // Don't free batch here - reuse it for next chunk
            n_past += n_cur;
            ALOGD("completionStart(): llama_decode chunk ok cur=%d n_cur=%d, n_past=%d (total tokens=%d, remaining=%d)", 
                  cur, n_cur, n_past, n_tokens, n_tokens - (cur + n_cur));
            
            // Move to next chunk
            cur += n_cur;
        }
        
        // Free batch after all chunks are processed
        llama_batch_free(batch);
        
        ALOGD("completionStart(): Exited prompt evaluation loop. n_past=%d, n_tokens=%d", n_past, n_tokens);
        
        // Performance measurement: Calculate elapsed time
        auto prompt_eval_end = std::chrono::high_resolution_clock::now();
        auto prompt_eval_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(prompt_eval_end - prompt_eval_start);
        long long elapsed_ms = prompt_eval_elapsed.count();
        double avg_time_per_token = (n_tokens > 0) ? (static_cast<double>(elapsed_ms) / n_tokens) : 0.0;
        ALOGD("completionStart(): Prompt evaluation took %lld ms for %d tokens", elapsed_ms, n_tokens);
        ALOGD("completionStart(): Average time per token: %.2f ms", avg_time_per_token);
        
        // After prompt evaluation, decode the last token separately to get logits
        // This avoids the Vulkan backend hang when enabling logits in the last chunk
        if (n_past == n_tokens && n_tokens > 0) {
            ALOGD("completionStart(): Decoding last prompt token separately to get logits...");
            llama_batch last_token_batch = llama_batch_init(1, 0, 1);
            if (last_token_batch.token && last_token_batch.seq_id) {
                last_token_batch.n_tokens = 1;
                last_token_batch.token[0] = prompt_tokens[n_tokens - 1];
                last_token_batch.logits[0] = true;  // Enable logits for the last token
                free(last_token_batch.pos);
                last_token_batch.pos = nullptr;  // Let llama_batch_allocr calculate position
                if (last_token_batch.seq_id[0]) {
                    last_token_batch.seq_id[0][0] = 0;
                    last_token_batch.n_seq_id[0] = 1;
                }
                
                ALOGD("completionStart(): Calling llama_decode() for last token to get logits");
                int last_decode_result = llama_decode(ctx, last_token_batch);
                ALOGD("completionStart(): llama_decode() returned %d for last token", last_decode_result);
                if (last_decode_result != 0) {
                    ALOGE("completionStart(): Failed to decode last token for logits, result=%d", last_decode_result);
                } else {
                    // Verify logits are available after decoding last token
                    const float* last_logits = llama_get_logits_ith(ctx, 0);
                    if (last_logits) {
                        ALOGD("completionStart(): Logits available after last token decode (idx=0)");
                        // Log top 3 logits for verification
                        const llama_model* model = llama_get_model(ctx);
                        const llama_vocab* vocab = llama_model_get_vocab(model);
                        const int n_vocab = llama_vocab_n_tokens(vocab);
                        std::vector<std::pair<float, llama_token>> candidates;
                        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                            candidates.push_back({last_logits[token_id], token_id});
                        }
                        std::sort(candidates.begin(), candidates.end(), 
                            [](const std::pair<float, llama_token>& a, const std::pair<float, llama_token>& b) {
                                return a.first > b.first;
                            });
                        ALOGD("completionStart(): Top 3 logits after last prompt token decode:");
                        for (int i = 0; i < 3 && i < (int)candidates.size(); i++) {
                            char token_text[256];
                            int n_len = llama_token_to_piece(vocab, candidates[i].second, token_text, sizeof(token_text), false, false);
                            token_text[n_len] = '\0';
                            ALOGD("completionStart():   [%d] id=%d, logit=%.3f, text='%s'", 
                                  i, (int)candidates[i].second, candidates[i].first, token_text);
                        }
                    } else {
                        ALOGE("completionStart(): Logits NOT available after last token decode (idx=0)");
                    }
                }
                llama_batch_free(last_token_batch);
            } else {
                ALOGE("completionStart(): Failed to initialize batch for last token");
            }
        }
        
        ALOGD("completionStart(): Prompt evaluation complete. n_past=%d, starting token generation...", n_past);
        
        // UTF-8 바이트 버퍼: 토큰의 바이트를 모아서 완전한 UTF-8 문자만 추출
        // Llama 토크나이저가 한글 한 글자의 3바이트를 여러 토큰으로 분리할 수 있으므로
        // 바이트를 버퍼에 모아서 완전한 UTF-8 시퀀스가 완성될 때까지 기다립니다
        std::string utf8_buffer;
        
        // 문장 완성 감지를 위한 변수
        bool sentence_complete = false;
        int extra_tokens_after_limit = 0;
        const int MAX_EXTRA_TOKENS = 50;  // 최대 추가 토큰 수 (문장 완성을 위해, 한국어 문장 완성을 위해 증가)
        const int MIN_GENERATION_LENGTH = 20;  // 최소 생성 길이: 최소 20개 토큰 생성 보장 (약 10~15자)
        // 열거형 패턴 감지 변수 (while 루프 전체에서 사용)
        bool isEnumerationPattern = false;
        bool incompleteEnumeration = false;
        
        // Continue generating tokens until limit is reached
        // Note: n_gen starts at 0, so we generate tokens 0..(n_predict-1) = n_predict tokens total
        // But we check n_gen < n_predict at the start of loop, so after generating n_predict tokens, n_gen will be n_predict and loop will exit
        while (n_past < context_size && (n_gen < n_predict || (!sentence_complete && extra_tokens_after_limit < MAX_EXTRA_TOKENS))) {
            // Check limit before generating token to ensure we don't exceed n_predict
            if (n_gen >= n_predict) {
                if (!sentence_complete && extra_tokens_after_limit < MAX_EXTRA_TOKENS) {
                    // n_predict에 도달했지만 문장이 완성되지 않았으면 추가 토큰 생성
                    extra_tokens_after_limit++;
                    ALOGD("completionStart(): Reached token limit (n_gen=%d >= n_predict=%d), but sentence incomplete, generating extra token %d/%d", 
                          n_gen, n_predict, extra_tokens_after_limit, MAX_EXTRA_TOKENS);
                } else {
                    ALOGD("completionStart(): Reached token limit (n_gen=%d >= n_predict=%d) and (sentence_complete=%d or extra_tokens=%d >= %d), breaking", 
                          n_gen, n_predict, sentence_complete ? 1 : 0, extra_tokens_after_limit, MAX_EXTRA_TOKENS);
                    break;
                }
            }
            ALOGD("completionStart(): Loop iteration n_gen=%d, n_past=%d", n_gen, n_past);
            if (handle->stopRequested) {
                ALOGD("completionStart(): Stop requested, breaking");
                break;
            }

            // Sample from logits
            // For the first token generation, use idx=0 to get logits from the last decode (last prompt token)
            // For subsequent tokens, use idx=0 to get logits from the most recent decode
            int32_t logits_idx = 0;  // Changed from -1 to 0 since we decoded the last token separately
            ALOGD("completionStart(): Calling llama_sampler_sample() with idx=%d (n_gen=%d, n_past=%d)", logits_idx, n_gen, n_past);
            
            // Check if logits are available before sampling
            const float* logits_check = llama_get_logits_ith(ctx, logits_idx);
            if (!logits_check) {
                ALOGE("completionStart(): logits are null for idx=%d, cannot sample token", logits_idx);
                jstring err = threadEnv->NewStringUTF("Logits not available");
                if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
                    threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                    if (threadEnv->ExceptionCheck()) {
                        ALOGE("completionStart(): Exception in error callback - clearing");
                        threadEnv->ExceptionClear();
                    }
                }
                threadEnv->DeleteLocalRef(err);
                break;
            }
            
            // REMOVED: Korean token boosting - this was interfering with context understanding
            // The model should generate tokens based on context, not forced language preference
            // Let the system prompt and model's natural understanding guide token selection
            
            llama_token id = llama_sampler_sample(smpl, ctx, logits_idx);
            ALOGD("completionStart(): llama_sampler_sample() returned id=%d (n_gen=%d)", (int)id, n_gen);
            
            // Log top 5 candidate tokens for analysis (for debugging context understanding)
            if (logits_check) {
                const llama_model* model = llama_get_model(ctx);
                const llama_vocab* vocab = llama_model_get_vocab(model);
                const int n_vocab = llama_vocab_n_tokens(vocab);
                std::vector<std::pair<float, llama_token>> candidates;
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.push_back({logits_check[token_id], token_id});
                }
                std::sort(candidates.begin(), candidates.end(), 
                    [](const std::pair<float, llama_token>& a, const std::pair<float, llama_token>& b) {
                        return a.first > b.first;
                    });
                ALOGD("completionStart(): Top 5 candidate tokens (context-based):");
                for (int i = 0; i < 5 && i < (int)candidates.size(); i++) {
                    char token_text[256];
                    int n_len = llama_token_to_piece(vocab, candidates[i].second, token_text, sizeof(token_text), false, false);
                    token_text[n_len] = '\0';
                    ALOGD("completionStart():   [%d] id=%d, logit=%.3f, text='%s'", 
                          i, (int)candidates[i].second, candidates[i].first, token_text);
                }
            }
            llama_sampler_accept(smpl, id);
            ALOGD("completionStart(): llama_sampler_accept() completed");

            // [PRIMARY STOP CONDITION - 최우선 정지 신호]
            // EOT 토큰(End-of-Turn, ID: 128009)을 최우선으로 체크
            // Llama 3.1은 대화의 한 턴이 끝났음을 알리기 위해 <|eot_id|>를 생성하도록 훈련됨
            // 이것이 가장 자연스럽고 신뢰할 수 있는 정지 신호이므로 다른 어떤 조건보다 우선하여 실행
            // EOT 토큰이 생성되면 모델이 "내 답변은 여기까지입니다"라고 판단한 것이므로 즉시 생성 중단
            if (id == 128009) {
                ALOGD("completionStart(): EOT token (128009) detected, breaking generation loop gracefully");
                break;  // 루프를 즉시 탈출합니다
            }
            // [FALLBACK STOP CONDITION]
            // EOT 토큰이 생성되지 않은 경우를 대비한 보조 정지 신호
            // 일반적인 End of Sequence (EOS) 토큰도 체크
            if (id == llama_vocab_eos(vocab) || id == 128001 || id == 128008) {
                ALOGD("completionStart(): EOS token detected (id=%d), breaking generation loop", (int)id);
                break;
            }

            // Special token handling (ID >= 128000): skip output but still process
            // Process special tokens through normal flow but skip text output
            bool isSpecialToken = (id >= 128000);
            ALOGD("completionStart(): Token id=%d, isSpecialToken=%d", (int)id, isSpecialToken ? 1 : 0);
            
            // Convert token to piece
            ALOGD("completionStart(): Calling llama_token_to_piece()");
            std::vector<char> piece(16, 0);
            int n_len = llama_token_to_piece(vocab, id, piece.data(), piece.size(), false, false);
            ALOGD("completionStart(): llama_token_to_piece() returned n_len=%d", n_len);
            if (n_len < 0) {
                ALOGE("completionStart(): llama_token_to_piece() failed");
                break;
            }
            if (static_cast<size_t>(n_len) >= piece.size()) {
                ALOGD("completionStart(): Resizing piece buffer from %zu to %d", piece.size(), n_len + 1);
                piece.resize(n_len + 1);
                n_len = llama_token_to_piece(vocab, id, piece.data(), piece.size(), false, false);
                if (n_len < 0) {
                    ALOGE("completionStart(): llama_token_to_piece() retry failed");
                    break;
                }
            }
            piece.resize(n_len);
            
            // [수정 1] 바이트를 즉시 변환하지 말고 버퍼에 추가합니다
            utf8_buffer.append(piece.data(), piece.size());
            ALOGD("completionStart(): Added %d bytes to UTF-8 buffer (total buffer size=%zu)", n_len, utf8_buffer.length());
            
            // [수정 2] 버퍼에서 유효한 UTF-8 문자열을 추출합니다
            size_t valid_utf8_len = 0;
            size_t offset = 0;
            
            while (offset < utf8_buffer.length()) {
                unsigned char first_byte = static_cast<unsigned char>(utf8_buffer[offset]);
                int char_len = 0;
                
                if (first_byte < 0x80) {
                    // 1-byte char (ASCII)
                    char_len = 1;
                } else if ((first_byte & 0xE0) == 0xC0) {
                    // 2-byte char
                    char_len = 2;
                } else if ((first_byte & 0xF0) == 0xE0) {
                    // 3-byte char (대부분의 한글)
                    char_len = 3;
                } else if ((first_byte & 0xF8) == 0xF0) {
                    // 4-byte char
                    char_len = 4;
                } else {
                    // 잘못된 바이트 시퀀스 - 첫 바이트를 건너뛰고 계속
                    ALOGD("completionStart(): Invalid UTF-8 start byte 0x%02x at offset %zu, skipping", first_byte, offset);
                    offset++;
                    continue;
                }
                
                // 버퍼에 완전한 문자가 존재하는지 확인
                if (offset + char_len <= utf8_buffer.length()) {
                    // continuation bytes 검증
                    bool valid = true;
                    for (int i = 1; i < char_len; i++) {
                        unsigned char byte = static_cast<unsigned char>(utf8_buffer[offset + i]);
                        if ((byte & 0xC0) != 0x80) {
                            // continuation byte가 아님
                            valid = false;
                            break;
                        }
                    }
                    
                    if (valid) {
                        // 완전한 UTF-8 문자가 존재합니다
                        valid_utf8_len += char_len;
                        offset += char_len;
                    } else {
                        // continuation byte가 잘못됨 - 첫 바이트를 건너뛰고 계속
                        ALOGD("completionStart(): Invalid continuation byte at offset %zu, skipping first byte", offset);
                        offset++;
                        break;
                    }
                } else {
                    // 버퍼에 불완전한 문자만 남았습니다. 다음 토큰을 기다립니다
                    ALOGD("completionStart(): Incomplete UTF-8 sequence at offset %zu (need %d bytes, have %zu), waiting for next token", 
                          offset, char_len, utf8_buffer.length() - offset);
                    break;
                }
            }
            
            std::string tokenText;
            if (valid_utf8_len > 0) {
                // [수정 3] 유효한 부분만 문자열로 만듭니다
                tokenText = utf8_buffer.substr(0, valid_utf8_len);
                ALOGD("completionStart(): Extracted valid UTF-8 text (length=%zu)", tokenText.length());
                
                // [수정 4] 처리된 부분을 버퍼에서 제거합니다
                utf8_buffer.erase(0, valid_utf8_len);
                ALOGD("completionStart(): Remaining buffer size=%zu", utf8_buffer.length());
            } else {
                // 유효한 UTF-8 문자가 없음 - 다음 토큰을 기다립니다
                ALOGD("completionStart(): No complete UTF-8 sequence yet, waiting for next token (buffer size=%zu)", utf8_buffer.length());
            }
            
            // 1단계: 토큰 레벨 특수 토큰 필터링 (모든 토큰에 대해 적용)
            // 특수 토큰 ID가 아니더라도 텍스트에 특수 토큰 패턴이 포함될 수 있으므로 모든 토큰에 필터링 적용
            if (!tokenText.empty()) {
                tokenText = filterSpecialTokensTokenLevel(tokenText);
                if (tokenText.empty()) {
                    ALOGD("completionStart(): Token filtered out by token-level filter");
                }
            }
            
            // 특수 토큰 ID를 가진 토큰은 절대 출력하지 않음
            // 필터링 후에도 빈 문자열이면 출력하지 않음
            // Only send non-special tokens to callback
            if (!isSpecialToken && !tokenText.empty()) {
                ALOGD("completionStart(): Token text='%s' (length=%zu), creating JNI string and calling callback", 
                      tokenText.c_str(), tokenText.length());
                
                // Convert UTF-8 to JNI string safely
                // NewStringUTF requires Modified UTF-8 which can fail for invalid UTF-8 bytes
                // Use byte array method which handles any UTF-8 bytes safely
                jstring tk = nullptr;
                
                // Create byte array from UTF-8 bytes and convert to String via Java
                jbyteArray byteArray = threadEnv->NewByteArray(tokenText.length());
                if (byteArray) {
                    threadEnv->SetByteArrayRegion(byteArray, 0, tokenText.length(), 
                                                  reinterpret_cast<const jbyte*>(tokenText.data()));
                    // Call Java method to convert byte array to String using UTF-8 charset
                    jclass stringClass = threadEnv->FindClass("java/lang/String");
                    if (stringClass) {
                        jmethodID stringCtor = threadEnv->GetMethodID(stringClass, "<init>", "([BLjava/lang/String;)V");
                        if (stringCtor) {
                            jstring charsetName = threadEnv->NewStringUTF("UTF-8");
                            if (charsetName) {
                                tk = (jstring)threadEnv->NewObject(stringClass, stringCtor, byteArray, charsetName);
                                if (threadEnv->ExceptionCheck()) {
                                    ALOGE("completionStart(): Exception creating String from byte array");
                                    threadEnv->ExceptionClear();
                                    tk = nullptr;
                                }
                                threadEnv->DeleteLocalRef(charsetName);
                            }
                        }
                        threadEnv->DeleteLocalRef(stringClass);
                    }
                    threadEnv->DeleteLocalRef(byteArray);
                }
                
                if (tk) {
                    // For Kotlin interfaces, each implementation is a different anonymous class.
                    // We need to get the method ID from the actual callback object's class.
                    if (gCallback && threadEnv) {
                        // Get method ID from the actual callback object's class
                        jclass callbackClass = nullptr;
                        jmethodID onTokenMethod = nullptr;
                        bool success = false;
                        
                        callbackClass = threadEnv->GetObjectClass(gCallback);
                        if (callbackClass) {
                            onTokenMethod = threadEnv->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");
                            if (onTokenMethod) {
                                ALOGD("completionStart(): Calling onToken callback with token='%s'", tokenText.c_str());
                                threadEnv->CallVoidMethod(gCallback, onTokenMethod, tk);
                                if (threadEnv->ExceptionCheck()) {
                                    ALOGE("completionStart(): Exception in token callback - clearing");
                                    threadEnv->ExceptionClear();
                                } else {
                                    ALOGD("completionStart(): Token callback completed successfully");
                                    success = true;
                                }
                            } else {
                                ALOGE("completionStart(): Failed to get onToken method ID from callback class");
                            }
                        } else {
                            ALOGE("completionStart(): Failed to get callback object class");
                        }
                        
                        // Clean up local references safely
                        if (callbackClass && threadEnv) {
                            threadEnv->DeleteLocalRef(callbackClass);
                        }
                        
                        if (!success) {
                            ALOGE("completionStart(): Token callback failed");
                        }
                    } else {
                        if (!gCallback) {
                            ALOGE("completionStart(): gCallback is null - skipping token callback");
                        }
                        if (!threadEnv) {
                            ALOGE("completionStart(): threadEnv is null - skipping token callback");
                        }
                    }
                    if (threadEnv && tk) {
                        threadEnv->DeleteLocalRef(tk);
                    }
                } else {
                    ALOGE("completionStart(): Failed to create JNI string");
                }
            generated.append(tokenText);
                
                // 2단계: 텍스트 레벨 특수 토큰 필터링 (누적된 텍스트에서 추가 필터링)
                // 매 토큰마다 텍스트 레벨 필터링을 수행하여 여러 토큰이 조합되어 생성된 특수 토큰 패턴도 즉시 제거
                // 이렇게 하면 특수 토큰이 화면에 출력되는 것을 완전히 방지할 수 있음
                std::string filteredGenerated = filterSpecialTokensTextLevel(generated);
                
                // 추가 필터링: "eot>", "eom>", "_id>" 등의 패턴이 텍스트 끝에 있는지 확인하고 제거
                // 이는 여러 토큰으로 분리되어 생성된 경우를 처리하기 위함
                if (filteredGenerated.length() >= 4) {
                    // 텍스트 끝에서 "eot>", "eom>", "_id>" 패턴 확인
                    std::string suffix = filteredGenerated.substr(filteredGenerated.length() - 4);
                    if (suffix == "eot>" || suffix == "eom>" || suffix == "_id>") {
                        filteredGenerated.erase(filteredGenerated.length() - 4);
                        ALOGD("completionStart(): Removed special token pattern '%s' from end of text", suffix.c_str());
                    }
                }
                // 텍스트 끝에서 "eot", "eom" 패턴 확인 (다음 토큰에서 ">"가 올 수 있음)
                // 단, 줄바꿈(단일 또는 두 줄 바꿈) 다음에 오는 경우는 제외 (줄바꿈은 유지)
                if (filteredGenerated.length() >= 3) {
                    std::string suffix = filteredGenerated.substr(filteredGenerated.length() - 3);
                    if (suffix == "eot" || suffix == "eom") {
                        // 줄바꿈(단일 또는 두 줄 바꿈) 다음에 오는 경우는 제외
                        bool shouldRemove = true;
                        if (filteredGenerated.length() > 3) {
                            char prevChar = filteredGenerated[filteredGenerated.length() - 4];
                            // 단일 줄바꿈 또는 두 줄 바꿈 다음에 오는 경우는 제외
                            if (prevChar == '\n') {
                                // 두 줄 바꿈인지 확인
                                if (filteredGenerated.length() > 4 && filteredGenerated[filteredGenerated.length() - 5] == '\n') {
                                    shouldRemove = false; // 두 줄 바꿈 다음
                                } else {
                                    shouldRemove = false; // 단일 줄바꿈 다음
                                }
                            }
                        }
                        if (shouldRemove) {
                            // 다음 토큰이 ">"일 가능성이 높으므로 미리 제거
                            filteredGenerated.erase(filteredGenerated.length() - 3);
                            ALOGD("completionStart(): Removed special token pattern '%s' from end of text (preventing 'eot>' or 'eom>')", suffix.c_str());
                        }
                    }
                }
                // 텍스트 끝에서 단일 문자 "e" 확인 (다음 토큰에서 "ot>"가 올 수 있음)
                // 단, 줄바꿈(단일 또는 두 줄 바꿈) 다음에 오는 경우는 제외
                if (filteredGenerated.length() >= 1 && filteredGenerated.back() == 'e') {
                    // 다음 토큰이 "ot>"일 가능성이 있으므로 제거하지 않지만, 텍스트 레벨 필터링에서 처리됨
                    // 단, 줄바꿈(단일 또는 두 줄 바꿈) 다음에 오는 경우는 제외
                    bool shouldWarn = true;
                    if (filteredGenerated.length() > 1) {
                        char prevChar = filteredGenerated[filteredGenerated.length() - 2];
                        if (prevChar == '\n') {
                            // 두 줄 바꿈인지 확인
                            if (filteredGenerated.length() > 2 && filteredGenerated[filteredGenerated.length() - 3] == '\n') {
                                shouldWarn = false; // 두 줄 바꿈 다음
                            } else {
                                shouldWarn = false; // 단일 줄바꿈 다음
                            }
                        }
                    }
                    if (shouldWarn) {
                        ALOGD("completionStart(): Warning: text ends with 'e', may form 'eot>' pattern");
                    }
                }
                
                if (filteredGenerated != generated) {
                    ALOGD("completionStart(): Text-level filter removed special tokens (before=%zu, after=%zu)", 
                          generated.length(), filteredGenerated.length());
                    generated = filteredGenerated;
                    // 필터링 후 길이가 줄어들었으면, UI에 반영하기 위해 마지막 메시지를 다시 전송해야 할 수도 있음
                    // 하지만 이미 토큰 단위로 전송했으므로, 다음 토큰에서 자연스럽게 수정됨
                }
            } else {
                ALOGD("completionStart(): Skipping token (isSpecialToken=%d, empty=%d)", 
                      isSpecialToken ? 1 : 0, tokenText.empty() ? 1 : 0);
            }
            
            bool hitStop = false;
            // 열거형 패턴이 진행 중인지 먼저 확인 (stop sequence 체크 전에)
            // 열거형 패턴이 진행 중이면 목록 항목이 완성될 때까지 stop sequence를 무시
            bool enumerationInProgress = false;
            if (isEnumerationPattern && incompleteEnumeration) {
                enumerationInProgress = true;
                ALOGD("completionStart(): Enumeration in progress, will ignore stop sequences until enumeration completes");
            }
            
            // 최소 생성 길이를 넘었을 때만 사용자 정의 정지 시퀀스 검사를 시작합니다.
            // 이렇게 하면 모델이 짧은 답변이라도 최소한의 완결성을 갖출 수 있는 "숨 쉴 틈"을 주게 됩니다.
            if (n_gen > MIN_GENERATION_LENGTH && !enumerationInProgress) {
                // 사용자 정의 정지 시퀀스 확인
            for (const auto& stop : stops) {
                if (!stop.empty() && generated.size() >= stop.size()) {
                    if (generated.compare(generated.size() - stop.size(), stop.size(), stop) == 0) {
                            // ".\n\n", "!\n\n", "?\n\n" 패턴의 경우, 목록 시작 기호가 뒤에 오는지 확인
                            // 목록 시작 기호: 숫자(0-9), "-", "*", "•" 등
                            if (stop == ".\n\n" || stop == "!\n\n" || stop == "?\n\n") {
                                // 다음 토큰을 미리 확인할 수 없으므로, 현재 생성된 텍스트의 끝 부분만 확인
                                // 실제로는 다음 토큰이 생성되기 전까지는 알 수 없으므로, 
                                // 이 패턴은 stopSequences에 포함되어 있지만 목록 보호를 위해
                                // 문장 완성 감지 로직에서도 처리합니다.
                                // 여기서는 일단 정지 시퀀스로 처리하되, 로그에 기록합니다.
                                ALOGD("completionStart(): Stop sequence '%s' detected (paragraph end pattern) after generating %d tokens, breaking generation", stop.c_str(), n_gen);
                            } else {
                                ALOGD("completionStart(): Stop sequence '%s' detected after generating %d tokens, breaking generation", stop.c_str(), n_gen);
                            }
                        hitStop = true;
                        break;
                    }
                    }
                }
            } else {
                if (n_gen <= MIN_GENERATION_LENGTH) {
                    ALOGD("completionStart(): Skipping stop sequence check (n_gen=%d <= MIN_GENERATION_LENGTH=%d)", n_gen, MIN_GENERATION_LENGTH);
                } else if (enumerationInProgress) {
                    ALOGD("completionStart(): Skipping stop sequence check (enumeration in progress)");
                }
            }
            // "eotend_header" 패턴 감지 (여러 토큰으로 분리되어 생성될 수 있음)
            // 생성된 텍스트 어디에든 "eotend_header" 패턴이 있으면 종료
            // 로그에서 확인된 패턴: "eotend_header>" (14자)
            if (!hitStop && generated.size() >= 13) {
                // "eotend_header"는 13자, "eotend_header>"는 14자
                size_t pos = generated.find("eotend_header");
                if (pos != std::string::npos) {
                    hitStop = true;
                    ALOGD("completionStart(): Pattern 'eotend_header' detected at position %zu, breaking generation", pos);
                }
            }
            // "<eotend_header>" 패턴도 확인 (완전한 형태)
            if (!hitStop && generated.size() >= 15) {
                // "<eotend_header>"는 15자
                size_t pos = generated.find("<eotend_header>");
                if (pos != std::string::npos) {
                    hitStop = true;
                    ALOGD("completionStart(): Pattern '<eotend_header>' detected at position %zu, breaking generation", pos);
                }
            }

            n_gen++;
            ALOGD("completionStart(): Incremented n_gen to %d", n_gen);
            
            // 문장 완성 감지: 마지막으로 생성된 텍스트에 문장 종료 문자가 있는지 확인
            // 줄바꿈은 제외: 줄바꿈 이후에도 더 출력될 내용이 있을 수 있으므로 종료 신호로 사용하지 않음
            if (!tokenText.empty() && !sentence_complete) {
                // 문장 종료 문자 확인: 마침표, 느낌표, 물음표만 (줄바꿈 제외)
                char last_char = tokenText.back();
                if (last_char == '.' || last_char == '!' || last_char == '?') {
                    sentence_complete = true;
                    ALOGD("completionStart(): Sentence completion detected (last_char='%c'), will finish after current token", last_char);
                }
            }
            
            // 생성된 전체 텍스트의 마지막 부분 확인 (UTF-8 버퍼 처리 후)
            // 줄바꿈 후 문장 완성 여부도 확인하여 미완성 문장으로 종료되는 것을 방지
            if (!generated.empty() && !sentence_complete) {
                // 열거형 패턴 감지: 목록 번호(1., 2., 3.) 또는 불릿(-, •) 패턴 확인
                // 변수는 while 루프 밖에서 선언되었으므로 매번 초기화
                isEnumerationPattern = false;
                incompleteEnumeration = false;
                
                // 전체 텍스트에서 열거형 패턴이 있는지 확인 (목록이 시작되었는지)
                // "1.", "2.", "- ", "* " 등의 패턴이 텍스트에 있는지 확인
                bool hasEnumerationInText = false;
                if (generated.find("1.") != std::string::npos ||
                    generated.find("2.") != std::string::npos ||
                    generated.find("3.") != std::string::npos ||
                    generated.find("- ") != std::string::npos ||
                    generated.find("* ") != std::string::npos) {
                    hasEnumerationInText = true;
                    isEnumerationPattern = true;
                }
                
                // 목록 시작 신호 감지: "다음과 같은", "다음과 같이", "다음은", "예를 들어" 등
                // 이런 표현이 있고 줄바꿈으로 끝나거나 ":" 뒤에 목록이 시작될 것으로 예상
                bool hasListStartSignal = false;
                // 줄바꿈으로 끝나는 경우와 ":" 뒤에 목록이 시작되는 경우 모두 확인
                if (!generated.empty()) {
                    // 마지막 부분 확인 (줄바꿈 또는 ":" 뒤)
                    size_t checkLength = std::min(generated.length(), size_t(100));
                    if (checkLength > 0) {
                        std::string lastPart = generated.substr(generated.length() - checkLength);
                        // 한국어 목록 시작 신호 패턴 (UTF-8 바이트로 확인)
                        // "다음과 같은" = 15바이트, "다음과 같이" = 15바이트, "다음은" = 9바이트, "예를 들어" = 12바이트
                        size_t pos = lastPart.find("다음과 같은");
                        if (pos == std::string::npos) {
                            pos = lastPart.find("다음과 같이");
                        }
                        if (pos == std::string::npos) {
                            pos = lastPart.find("다음은");
                        }
                        if (pos == std::string::npos) {
                            pos = lastPart.find("아래는");
                        }
                        if (pos == std::string::npos) {
                            pos = lastPart.find("예를 들어");
                        }
                        if (pos == std::string::npos) {
                            pos = lastPart.find("예를 들면");
                        }
                        if (pos == std::string::npos) {
                            pos = lastPart.find("예시로는");
                        }
                        if (pos == std::string::npos) {
                            pos = lastPart.find("예시로");
                        }
                        // 패턴이 발견되고, 그 뒤에 ":" 또는 ":" + 짧은 텍스트가 있으면 목록 시작 신호로 간주
                        if (pos != std::string::npos) {
                            // 패턴 뒤의 텍스트 확인
                            std::string afterPattern = lastPart.substr(pos);
                            // ":" 문자가 있는지 확인 (목록 시작 신호)
                            size_t colonPos = afterPattern.find(':');
                            if (colonPos != std::string::npos) {
                                // ":" 뒤의 텍스트 길이 확인
                                size_t textAfterColon = afterPattern.length() - colonPos - 1;
                                // ":" 뒤에 줄바꿈이 오는 경우 (textAfterColon이 0 또는 매우 짧음) 목록 시작 신호로 간주
                                // 또는 ":" 뒤의 텍스트가 짧으면 (최대 50바이트) 목록 시작 신호로 간주
                                if (textAfterColon <= 50) {
                                    hasListStartSignal = true;
                                    isEnumerationPattern = true;
                                    ALOGD("completionStart(): List start signal detected (pattern at pos %zu, colon at %zu, text after colon: %zu bytes)", pos, colonPos, textAfterColon);
                                }
                            } else {
                                // ":"가 없어도 패턴 뒤의 텍스트가 짧으면 (최대 20바이트) 목록 시작 신호로 간주
                                size_t textAfterPattern = afterPattern.length();
                                if (textAfterPattern <= 20) {
                                    hasListStartSignal = true;
                                    isEnumerationPattern = true;
                                    ALOGD("completionStart(): List start signal detected (pattern at pos %zu, text after: %zu bytes)", pos, textAfterPattern);
                                }
                            }
                        }
                    }
                }
                
                // 마지막 줄이 열거형 패턴으로 시작하는지 확인
                size_t lastNewlinePos = generated.rfind('\n');
                if (lastNewlinePos != std::string::npos && lastNewlinePos + 1 < generated.length()) {
                    std::string lastLine = generated.substr(lastNewlinePos + 1);
                    // 목록 번호 패턴 확인: \n\d+\. 또는 \n- 또는 \n• 또는 \n*
                    if (!lastLine.empty()) {
                        // 숫자로 시작하고 마침표가 있는 패턴 (예: "1.", "2.", "10.")
                        if (lastLine.length() >= 2 && 
                            lastLine[0] >= '0' && lastLine[0] <= '9' && 
                            lastLine[1] == '.') {
                            isEnumerationPattern = true;
                            // 마지막 항목이 완성되지 않았는지 확인 (마침표/느낌표/물음표로 끝나지 않음)
                            if (lastLine.length() > 2) {
                                char lastChar = lastLine.back();
                                if (lastChar != '.' && lastChar != '!' && lastChar != '?' && lastChar != '\n') {
                                    incompleteEnumeration = true;
                                }
                            } else {
                                // 항목 번호만 있고 내용이 없음
                                incompleteEnumeration = true;
                            }
                        }
                        // 불릿 패턴 확인: "- " 또는 "* " (•는 UTF-8 멀티바이트이므로 문자열 비교로 처리)
                        else if (lastLine.length() >= 2 && 
                                 (lastLine[0] == '-' || lastLine[0] == '*') &&
                                 lastLine[1] == ' ') {
                            isEnumerationPattern = true;
                            // 마지막 항목이 완성되지 않았는지 확인
                            if (lastLine.length() > 2) {
                                char lastChar = lastLine.back();
                                if (lastChar != '.' && lastChar != '!' && lastChar != '?' && lastChar != '\n') {
                                    incompleteEnumeration = true;
                                }
                            } else {
                                // 불릿만 있고 내용이 없음
                                incompleteEnumeration = true;
                            }
                        }
                        // UTF-8 불릿 문자(•) 패턴 확인: "• " (4바이트: 0xE2 0x80 0xA2 0x20)
                        else if (lastLine.length() >= 4) {
                            // "• "는 UTF-8로 0xE2 0x80 0xA2 0x20 (4바이트)
                            if (static_cast<unsigned char>(lastLine[0]) == 0xE2 &&
                                static_cast<unsigned char>(lastLine[1]) == 0x80 &&
                                static_cast<unsigned char>(lastLine[2]) == 0xA2 &&
                                lastLine[3] == ' ') {
                                isEnumerationPattern = true;
                                // 마지막 항목이 완성되지 않았는지 확인
                                if (lastLine.length() > 4) {
                                    char lastChar = lastLine.back();
                                    if (lastChar != '.' && lastChar != '!' && lastChar != '?' && lastChar != '\n') {
                                        incompleteEnumeration = true;
                                    }
                                } else {
                                    // 불릿만 있고 내용이 없음
                                    incompleteEnumeration = true;
                                }
                            }
                        }
                    }
                }
                
                // 줄바꿈 후 문장이 완성되지 않았는지 확인
                // 마지막 문자가 줄바꿈이고, 그 앞에 문장 종료 패턴이 없으면 미완성으로 간주
                if (generated.back() == '\n') {
                    // 줄바꿈 앞의 텍스트 확인 (최대 20바이트)
                    size_t checkBeforeNewline = std::min(generated.length() - 1, size_t(20));
                    if (checkBeforeNewline > 0) {
                        std::string beforeNewline = generated.substr(generated.length() - 1 - checkBeforeNewline, checkBeforeNewline);
                        // 줄바꿈 앞에 문장 종료 패턴이 있는지 확인
                        bool hasEndingBeforeNewline = false;
                        // 마침표, 느낌표, 물음표 확인
                        if (!beforeNewline.empty()) {
                            char lastBeforeNewline = beforeNewline.back();
                            if (lastBeforeNewline == '.' || lastBeforeNewline == '!' || lastBeforeNewline == '?') {
                                hasEndingBeforeNewline = true;
                            }
                        }
                        // 한국어 종료 패턴 확인
                        if (!hasEndingBeforeNewline && beforeNewline.length() >= 3) {
                            std::string lastThree = beforeNewline.substr(beforeNewline.length() - 3);
                            const std::vector<std::string> koreanEndChars = {
                                "\xEB\x8B\xA4",  // "다"
                                "\xEC\x9A\x94",  // "요"
                                "\xEB\x84\xA4",  // "네"
                                "\xEC\x96\xB4"   // "어"
                            };
                            for (const auto& endChar : koreanEndChars) {
                                if (lastThree == endChar) {
                                    hasEndingBeforeNewline = true;
                                    break;
                                }
                            }
                        }
                        // 줄바꿈 앞에 문장 종료 패턴이 없으면 미완성으로 간주하여 추가 토큰 생성 계속
                        if (!hasEndingBeforeNewline) {
                            ALOGD("completionStart(): Newline detected but no sentence ending before it, continuing generation");
                            // sentence_complete는 false로 유지하여 추가 토큰 생성 계속
                        }
                    }
                }
                
                // 열거형 패턴이 텍스트에 있거나 목록 시작 신호가 있고, 마지막이 줄바꿈으로 끝나면 미완성으로 간주
                // (목록이 시작되었거나 곧 시작될 것으로 예상되지만 마지막 항목이 완성되지 않았을 가능성)
                if ((hasEnumerationInText || hasListStartSignal) && !generated.empty() && generated.back() == '\n') {
                    // 줄바꿈 앞에 문장 종료 패턴이 있는지 확인
                    size_t checkBeforeNewline = std::min(generated.length() - 1, size_t(20));
                    if (checkBeforeNewline > 0) {
                        std::string beforeNewline = generated.substr(generated.length() - 1 - checkBeforeNewline, checkBeforeNewline);
                        bool hasEnding = false;
                        if (!beforeNewline.empty()) {
                            char lastBeforeNewline = beforeNewline.back();
                            if (lastBeforeNewline == '.' || lastBeforeNewline == '!' || lastBeforeNewline == '?') {
                                hasEnding = true;
                            }
                        }
                        // 한국어 종료 패턴 확인
                        if (!hasEnding && beforeNewline.length() >= 3) {
                            std::string lastThree = beforeNewline.substr(beforeNewline.length() - 3);
                            const std::vector<std::string> koreanEndChars = {
                                "\xEB\x8B\xA4", "\xEC\x9A\x94", "\xEB\x84\xA4", "\xEC\x96\xB4"
                            };
                            for (const auto& endChar : koreanEndChars) {
                                if (lastThree == endChar) {
                                    hasEnding = true;
                                    break;
                                }
                            }
                        }
                        // 열거형 패턴이 있고 줄바꿈 앞에 종료 패턴이 없으면 미완성으로 간주
                        if (!hasEnding) {
                            incompleteEnumeration = true;
                            ALOGD("completionStart(): Enumeration pattern detected with newline but no ending, marking as incomplete");
                        }
                    }
                }
                
                // 열거형 패턴이 미완성인 경우 로그 기록
                if (isEnumerationPattern && incompleteEnumeration) {
                    ALOGD("completionStart(): Incomplete enumeration pattern detected, continuing generation");
                }
                // 1. 마지막 문자가 문장 종료 문자인지 확인
                char last_gen_char = generated.back();
                if (last_gen_char == '.' || last_gen_char == '!' || last_gen_char == '?') {
                    sentence_complete = true;
                    ALOGD("completionStart(): Sentence completion detected in generated text (last_char='%c')", last_gen_char);
                } else {
                    // 2. 한국어 문장 종료 패턴 확인 (마지막 몇 글자 확인)
                    // 레벨 1: 가장 안전하고 필수적인 종료 패턴 (마침표/느낌표/물음표 포함)
                    // 레벨 2: 신뢰도 높은 일반 종료 패턴
                    // 더 긴 패턴부터 확인 (예: "습니다." -> "습니다" -> "다")
                    const std::vector<std::string> koreanEndings = {
                        // 레벨 1: 필수 종료 패턴 (문장 종료 기호 포함)
                        "습니다.", "니다.", "요?", "죠?", "가요?", "까요?", "네요!", "군요!",
                        // 레벨 2: 일반 종료 패턴 (문장 종료 기호 포함)
                        "요.", "죠.", "예요.", "에요.",
                        // 레벨 3: 맥락 의존적 종료 패턴
                        "다.",
                        // 기본 종료 패턴 (문장 종료 기호 없이)
                        "습니다", "입니다", "합니다", "네요", "어요", "세요", "까요", "나요", "니요",
                        "요", "다", "네", "어", "지", "게", "까", "나", "니"
                    };
                    
                    // 마지막 12바이트까지 확인 (한국어 문장 종료 패턴은 보통 1~3글자 + 문장 종료 기호, 한 글자는 3바이트)
                    // 예: "습니다." = 9바이트 (3글자) + 1바이트 (마침표) = 10바이트
                    size_t checkLength = std::min(generated.length(), size_t(12));
                    if (checkLength > 0) {
                        std::string lastPart = generated.substr(generated.length() - checkLength);
                        
                        for (const auto& ending : koreanEndings) {
                            // ending의 UTF-8 바이트 길이 계산
                            size_t endingBytes = ending.length();
                            
                            if (lastPart.length() >= endingBytes) {
                                // 마지막 부분이 한국어 종료 패턴으로 끝나는지 확인
                                std::string suffix = lastPart.substr(lastPart.length() - endingBytes);
                                if (suffix == ending) {
                                    // 종료 패턴이 텍스트의 끝이거나, 종료 패턴 뒤에 문장 종료 문자가 오는 경우
                                    // 또는 종료 패턴 앞에 공백이나 줄바꿈이 있는 경우 (완성된 문장)
                                    bool isComplete = false;
                                    
                                    // 1. 종료 패턴이 텍스트의 끝인 경우 (예: "좋습니다")
                                    if (generated.length() == endingBytes) {
                                        isComplete = true;
                                    }
                                    // 2. 종료 패턴 뒤에 문장 종료 문자가 오는 경우 (예: "입니다.", "합니다!")
                                    else if (generated.length() > endingBytes) {
                                        // 종료 패턴 바로 뒤의 문자 확인
                                        size_t posAfterEnding = generated.length() - endingBytes;
                                        if (posAfterEnding < generated.length()) {
                                            char charAfter = generated[posAfterEnding];
                                            if (charAfter == '.' || charAfter == '!' || charAfter == '?') {
                                                isComplete = true;
                                                // 문장 종료 문자 뒤에 "\n\n"이 오는 경우도 확인 (레벨 4 패턴)
                                                if (generated.length() >= endingBytes + 3) {
                                                    std::string afterEnding = generated.substr(posAfterEnding, 3);
                                                    if (afterEnding == ".\n\n" || afterEnding == "!\n\n" || afterEnding == "?\n\n") {
                                                        isComplete = true;
                                                        ALOGD("completionStart(): Paragraph end pattern detected after Korean ending");
                                                    }
                                                }
                                            }
                                            // 3. 종료 패턴 뒤에 공백이나 줄바꿈이 오는 경우도 완성된 문장으로 간주
                                            else if (charAfter == ' ' || charAfter == '\n') {
                                                isComplete = true;
                                            }
                                        }
                                    }
                                    
                                    if (isComplete) {
                                        sentence_complete = true;
                                        ALOGD("completionStart(): Korean sentence completion detected (ending='%s')", ending.c_str());
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    
                    // 3. 마지막 문자가 한국어 종료 패턴으로 끝나는 경우 완성된 문장으로 간주
                    // (예: "좋습니다" -> 완성된 문장)
                    // 한국어는 UTF-8로 인코딩되므로 문자열 비교를 사용
                    if (!sentence_complete && generated.length() >= 3) {
                        // 마지막 3바이트 확인 (한국어 한 글자는 보통 3바이트)
                        std::string lastThreeBytes = generated.substr(generated.length() - 3);
                        // 한국어 종료 문자 패턴 확인 (UTF-8 인코딩)
                        // "다" (0xEB, 0x8B, 0xA4), "요" (0xEC, 0x9A, 0x94), "네" (0xEB, 0x84, 0xA4), "어" (0xEC, 0x96, 0xB4)
                        const std::vector<std::string> koreanEndChars = {
                            "\xEB\x8B\xA4",  // "다"
                            "\xEC\x9A\x94",  // "요"
                            "\xEB\x84\xA4",  // "네"
                            "\xEC\x96\xB4"   // "어"
                        };
                        
                        for (const auto& endChar : koreanEndChars) {
                            if (lastThreeBytes == endChar) {
                                // 종료 문자로 끝나는 경우, 이전 문자가 공백이나 줄바꿈이 아니면 완성된 문장으로 간주
                                // 또는 종료 문자가 텍스트의 끝이면 완성된 문장
                                bool isComplete = false;
                                if (generated.length() == 3) {
                                    // 텍스트가 종료 문자로만 구성된 경우는 제외
                                    isComplete = false;
                                } else {
                                    // 종료 문자 앞의 문자가 공백이나 줄바꿈이 아니면 완성된 문장
                                    char charBefore = generated[generated.length() - 4];
                                    if (charBefore != ' ' && charBefore != '\n' && charBefore != '\t') {
                                        isComplete = true;
                                    }
                                }
                                
                                if (isComplete) {
                                    sentence_complete = true;
                                    ALOGD("completionStart(): Korean sentence completion detected (ends with Korean character)");
                                    break;
                                }
                            }
                        }
                        
                        // 마지막 6바이트 확인 (2글자 패턴, 예: "습니다")
                        if (!sentence_complete && generated.length() >= 6) {
                            std::string lastSixBytes = generated.substr(generated.length() - 6);
                            // "습니다" (0xEC, 0x8A, 0xB5, 0xEB, 0x8B, 0xA4)
                            if (lastSixBytes == "\xEC\x8A\xB5\xEB\x8B\xA4") {
                                // "습니다"로 끝나는 경우, 이전 문자가 공백이나 줄바꿈이 아니면 완성된 문장
                                bool isComplete = false;
                                if (generated.length() == 6) {
                                    isComplete = false;
                                } else {
                                    char charBefore = generated[generated.length() - 7];
                                    if (charBefore != ' ' && charBefore != '\n' && charBefore != '\t') {
                                        isComplete = true;
                                    }
                                }
                                
                                if (isComplete) {
                                    sentence_complete = true;
                                    ALOGD("completionStart(): Korean sentence completion detected (ends with '습니다')");
                                }
                            }
                        }
                    }
                }
            }
            
            // Check if we've reached the token limit after incrementing
            // 문장이 완성되었거나 최대 추가 토큰 수에 도달했으면 종료
            // 단, 줄바꿈 후 문장이 완성되지 않은 경우 또는 열거형 패턴이 미완성인 경우 추가 토큰 생성 계속
            if (n_gen >= n_predict) {
                // 줄바꿈 후 문장이 완성되지 않은 경우 추가 토큰 생성 계속
                bool incompleteAfterNewline = false;
                if (!generated.empty() && generated.back() == '\n') {
                    // 줄바꿈 앞에 문장 종료 패턴이 있는지 확인
                    size_t checkBeforeNewline = std::min(generated.length() - 1, size_t(20));
                    if (checkBeforeNewline > 0) {
                        std::string beforeNewline = generated.substr(generated.length() - 1 - checkBeforeNewline, checkBeforeNewline);
                        bool hasEnding = false;
                        if (!beforeNewline.empty()) {
                            char lastBeforeNewline = beforeNewline.back();
                            if (lastBeforeNewline == '.' || lastBeforeNewline == '!' || lastBeforeNewline == '?') {
                                hasEnding = true;
                            }
                        }
                        if (!hasEnding && beforeNewline.length() >= 3) {
                            std::string lastThree = beforeNewline.substr(beforeNewline.length() - 3);
                            const std::vector<std::string> koreanEndChars = {
                                "\xEB\x8B\xA4", "\xEC\x9A\x94", "\xEB\x84\xA4", "\xEC\x96\xB4"
                            };
                            for (const auto& endChar : koreanEndChars) {
                                if (lastThree == endChar) {
                                    hasEnding = true;
                                    break;
                                }
                            }
                        }
                        if (!hasEnding) {
                            incompleteAfterNewline = true;
                        }
                    }
                }
                
                // 열거형 패턴이 미완성인 경우 추가 토큰 생성 허용
                // 열거형 패턴의 경우 더 많은 추가 토큰을 허용 (MAX_EXTRA_TOKENS * 2)
                const int ENUMERATION_EXTRA_TOKENS = MAX_EXTRA_TOKENS * 2;
                bool allowExtraForEnumeration = (isEnumerationPattern && incompleteEnumeration && 
                                                 extra_tokens_after_limit < ENUMERATION_EXTRA_TOKENS);
                
                if (sentence_complete || 
                    (extra_tokens_after_limit >= MAX_EXTRA_TOKENS && !incompleteAfterNewline && !allowExtraForEnumeration)) {
                    ALOGD("completionStart(): Reached token limit (n_gen=%d >= n_predict=%d) and (sentence_complete=%d or extra_tokens=%d >= %d), breaking before decode", 
                          n_gen, n_predict, sentence_complete ? 1 : 0, extra_tokens_after_limit, MAX_EXTRA_TOKENS);
                    break;
                }
                // 문장이 완성되지 않았고 추가 토큰을 더 생성할 수 있으면 계속 진행
                extra_tokens_after_limit++;
                if (allowExtraForEnumeration) {
                    ALOGD("completionStart(): Reached token limit but incomplete enumeration pattern, continuing with extra token %d/%d", 
                          extra_tokens_after_limit, ENUMERATION_EXTRA_TOKENS);
                } else if (incompleteAfterNewline) {
                    ALOGD("completionStart(): Reached token limit but incomplete after newline, continuing with extra token %d/%d", 
                          extra_tokens_after_limit, MAX_EXTRA_TOKENS);
                } else {
                    ALOGD("completionStart(): Reached token limit but sentence incomplete, continuing with extra token %d/%d", 
                          extra_tokens_after_limit, MAX_EXTRA_TOKENS);
                }
            }

            // Prepare and run next decode with a single token
            ALOGD("completionStart(): Initializing llama_batch");
            llama_batch gen_batch = llama_batch_init(1, 0, 1);
            if (!gen_batch.token || !gen_batch.pos || !gen_batch.seq_id || !gen_batch.n_seq_id || !gen_batch.logits) {
                ALOGE("completionStart(): llama_batch_init() returned invalid gen_batch");
                jstring err = threadEnv->NewStringUTF("Failed to initialize generation batch");
                if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
                    threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                    if (threadEnv->ExceptionCheck()) {
                        ALOGE("completionStart(): Exception in error callback - clearing");
                        threadEnv->ExceptionClear();
                    }
                } else {
                    ALOGE("completionStart(): Callback validation failed - skipping error callback");
                }
                threadEnv->DeleteLocalRef(err);
                llama_batch_free(gen_batch);
                break;
            }
            gen_batch.n_tokens = 1;
            gen_batch.token   [0] = id;
            // For token generation, set pos to nullptr to let llama_batch_allocr::init() calculate position from memory
            // This ensures positions are consistent with the KV cache state, especially after separate last token decode
            free(gen_batch.pos);
            gen_batch.pos = nullptr;  // Let llama_batch_allocr calculate position from memory
            if (gen_batch.seq_id[0]) {
            gen_batch.seq_id  [0][0] = 0;
            } else {
                ALOGE("completionStart(): gen_batch.seq_id[0] is null!");
                llama_batch_free(gen_batch);
                break;
            }
            gen_batch.n_seq_id[0] = 1;
            gen_batch.logits  [0] = true;
            ALOGD("completionStart(): Batch initialized, calling llama_decode() with token=%d, pos=auto (n_past=%d)", (int)id, n_past);
            ALOGD("completionStart(): About to call llama_decode() for token generation, n_past=%d, n_gen=%d", n_past, n_gen);

            int decode_result = llama_decode(ctx, gen_batch);
            ALOGD("completionStart(): llama_decode() returned %d for token generation", decode_result);
            if (decode_result != 0) {
                ALOGE("completionStart(): llama_decode() failed on token, result=%d", decode_result);
                jstring err = threadEnv->NewStringUTF("Failed to decode token");
                if (g_CallbackClass && g_OnError && threadEnv->IsInstanceOf(gCallback, g_CallbackClass)) {
                threadEnv->CallVoidMethod(gCallback, g_OnError, err);
                    if (threadEnv->ExceptionCheck()) {
                        ALOGE("completionStart(): Exception in error callback - clearing");
                        threadEnv->ExceptionClear();
                    }
                } else {
                    ALOGE("completionStart(): Callback validation failed - skipping error callback");
                }
                threadEnv->DeleteLocalRef(err);
                llama_batch_free(gen_batch);
                break;
            }
            ALOGD("completionStart(): Freeing batch");
            llama_batch_free(gen_batch);
            n_past++;
            ALOGD("completionStart(): Incremented n_past to %d", n_past);

            if (hitStop) {
                break;
            }
        }

        llama_sampler_free(smpl);

        // 최종 텍스트 레벨 필터링: 생성 완료 시점에 한 번 더 필터링하여 특수 토큰 완전 제거
        // "eot>", "eom>" 등의 패턴이 최종적으로 남아있을 수 있으므로 한 번 더 필터링
        if (!generated.empty()) {
            std::string finalFiltered = filterSpecialTokensTextLevel(generated);
            if (finalFiltered != generated) {
                ALOGD("completionStart(): Final text-level filter removed special tokens (before=%zu, after=%zu)", 
                      generated.length(), finalFiltered.length());
                generated = finalFiltered;
            }
        }

        // 남은 UTF-8 버퍼 처리 (생성 완료 시점에)
        // 버퍼에 남은 바이트가 있으면, 가능한 한 완전한 UTF-8 문자를 추출하여 출력
        if (!utf8_buffer.empty()) {
            ALOGD("completionStart(): Processing remaining UTF-8 buffer at completion (length=%zu)", utf8_buffer.length());
            
            // 버퍼에서 가능한 한 많은 완전한 UTF-8 문자를 추출
            size_t valid_utf8_len = 0;
            size_t offset = 0;
            
            while (offset < utf8_buffer.length()) {
                unsigned char first_byte = static_cast<unsigned char>(utf8_buffer[offset]);
                int char_len = 0;
                
                if (first_byte < 0x80) {
                    char_len = 1;
                } else if ((first_byte & 0xE0) == 0xC0) {
                    char_len = 2;
                } else if ((first_byte & 0xF0) == 0xE0) {
                    char_len = 3;
                } else if ((first_byte & 0xF8) == 0xF0) {
                    char_len = 4;
                } else {
                    // 잘못된 바이트 - 건너뛰기
                    offset++;
                    continue;
                }
                
                if (offset + char_len <= utf8_buffer.length()) {
                    // continuation bytes 검증
                    bool valid = true;
                    for (int i = 1; i < char_len; i++) {
                        unsigned char byte = static_cast<unsigned char>(utf8_buffer[offset + i]);
                        if ((byte & 0xC0) != 0x80) {
                            valid = false;
                            break;
                        }
                    }
                    
                    if (valid) {
                        valid_utf8_len += char_len;
                        offset += char_len;
                    } else {
                        break;
                    }
                } else {
                    // 불완전한 문자 - 버퍼에 남김
                    break;
                }
            }
            
            if (valid_utf8_len > 0) {
                std::string remainingText = utf8_buffer.substr(0, valid_utf8_len);
                remainingText = filterSpecialTokensTokenLevel(remainingText);
                
                if (!remainingText.empty() && gCallback && threadEnv) {
                    // 남은 텍스트를 콜백으로 전송
                    jbyteArray byteArray = threadEnv->NewByteArray(remainingText.length());
                    if (byteArray) {
                        threadEnv->SetByteArrayRegion(byteArray, 0, remainingText.length(), 
                                                      reinterpret_cast<const jbyte*>(remainingText.data()));
                        jclass stringClass = threadEnv->FindClass("java/lang/String");
                        if (stringClass) {
                            jmethodID stringCtor = threadEnv->GetMethodID(stringClass, "<init>", "([BLjava/lang/String;)V");
                            if (stringCtor) {
                                jstring charsetName = threadEnv->NewStringUTF("UTF-8");
                                if (charsetName) {
                                    jstring tk = (jstring)threadEnv->NewObject(stringClass, stringCtor, byteArray, charsetName);
                                    if (tk && !threadEnv->ExceptionCheck()) {
                                        jclass callbackClass = threadEnv->GetObjectClass(gCallback);
                                        if (callbackClass) {
                                            jmethodID onTokenMethod = threadEnv->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");
                                            if (onTokenMethod) {
                                                threadEnv->CallVoidMethod(gCallback, onTokenMethod, tk);
                                                threadEnv->ExceptionClear();
                                            }
                                            threadEnv->DeleteLocalRef(callbackClass);
                                        }
                                    }
                                    if (tk) threadEnv->DeleteLocalRef(tk);
                                    threadEnv->DeleteLocalRef(charsetName);
                                }
                            }
                            threadEnv->DeleteLocalRef(stringClass);
                        }
                        threadEnv->DeleteLocalRef(byteArray);
                    }
                }
                
                utf8_buffer.erase(0, valid_utf8_len);
            }
            
            // 남은 불완전한 바이트는 버퍼에서 제거
            if (!utf8_buffer.empty()) {
                ALOGD("completionStart(): Discarding incomplete UTF-8 bytes at completion (length=%zu)", utf8_buffer.length());
                utf8_buffer.clear();
            }
        }

        // 최종 텍스트 레벨 필터링 (생성 완료 시점에 한 번 더 필터링)
        std::string finalGenerated = filterSpecialTokensTextLevel(generated);
        if (finalGenerated != generated) {
            ALOGD("completionStart(): Final text-level filter removed special tokens (before=%zu, after=%zu)", 
                  generated.length(), finalGenerated.length());
            generated = finalGenerated;
        }

        // Call completed callback - dynamically get method ID from actual callback object's class
        // This is necessary because Kotlin interfaces are implemented as anonymous classes
        if (gCallback && threadEnv) {
            jclass callbackClass = nullptr;
            jmethodID onCompletedMethod = nullptr;
            bool success = false;
            
            callbackClass = threadEnv->GetObjectClass(gCallback);
            if (callbackClass) {
                onCompletedMethod = threadEnv->GetMethodID(callbackClass, "onCompleted", "()V");
                if (onCompletedMethod) {
                    ALOGD("completionStart(): Calling onCompleted callback");
                    threadEnv->CallVoidMethod(gCallback, onCompletedMethod);
                    if (threadEnv->ExceptionCheck()) {
                        ALOGE("completionStart(): Exception in completed callback - clearing");
                        threadEnv->ExceptionClear();
                    } else {
                        ALOGD("completionStart(): onCompleted callback completed successfully");
                        success = true;
                    }
                } else {
                    ALOGE("completionStart(): Failed to get onCompleted method ID from callback class");
                }
            } else {
                ALOGE("completionStart(): Failed to get callback object class for onCompleted");
            }
            
            // Clean up local references safely
            if (callbackClass && threadEnv) {
                threadEnv->DeleteLocalRef(callbackClass);
            }
            
            if (!success) {
                ALOGE("completionStart(): onCompleted callback failed");
            }
        } else {
            if (!gCallback) {
                ALOGE("completionStart(): gCallback is null - skipping onCompleted callback");
            }
            if (!threadEnv) {
                ALOGE("completionStart(): threadEnv is null - skipping onCompleted callback");
            }
        }
        
        // Clean up gCallback before detaching thread
        // gCallback은 전역 참조이므로 명시적으로 정리해야 합니다
        // detachThread() 전에 정리해야 JNI 환경이 유효한 상태에서 DeleteGlobalRef를 호출할 수 있습니다
        if (guard.env && guard.ref) {
            ALOGD("completionStart(): Cleaning up gCallback before detaching thread");
            guard.env->DeleteGlobalRef(guard.ref);
            guard.ref = nullptr;
            guard.shouldDelete = false;  // 이미 정리했으므로 CallbackGuard가 다시 정리하지 않도록
        }
        
        ALOGD("completionStart(): Worker thread completing, detaching from JVM");
		detachThread();
        ALOGD("completionStart(): Worker thread detached from JVM");
    });
    worker.detach();
    ALOGD("completionStart(): Worker thread detached, function returning (app should remain active)");
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_saveSession(
        JNIEnv* env, jobject /*thiz*/, jlong h, jstring jPath) {
    auto* handle = reinterpret_cast<LlamaCtx*>(h);
    if (!handle) return -1;
    const char* path = env->GetStringUTFChars(jPath, nullptr);
    int result = -3;
#if LLAMA_STUB_MODE
    result = 0;
#else
    if (handle->ctx && path) {
        const size_t stateSize = llama_state_get_size(handle->ctx);
        if (stateSize == 0) {
            result = -4;
        } else {
            std::vector<uint8_t> buffer(stateSize);
            const size_t written = llama_state_get_data(handle->ctx, buffer.data(), buffer.size());
            if (written != buffer.size()) {
                result = -5;
            } else {
                FILE* fp = fopen(path, "wb");
                if (!fp) {
                    result = -6;
                } else {
                    const size_t out = fwrite(buffer.data(), 1, buffer.size(), fp);
                    fclose(fp);
                    result = (out == buffer.size()) ? static_cast<int>(buffer.size()) : -7;
                }
            }
        }
    } else {
        result = -2;
    }
#endif
    env->ReleaseStringUTFChars(jPath, path);
    return result;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_loadSession(
        JNIEnv* env, jobject /*thiz*/, jlong h, jstring jPath) {
    auto* handle = reinterpret_cast<LlamaCtx*>(h);
    if (!handle) return JNI_FALSE;
    const char* path = env->GetStringUTFChars(jPath, nullptr);
    bool ok = false;
#if LLAMA_STUB_MODE
    ok = true;
#else
    if (handle->ctx && path) {
        FILE* fp = fopen(path, "rb");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            const long len = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            if (len > 0) {
                std::vector<uint8_t> buffer(static_cast<size_t>(len));
                const size_t read = fread(buffer.data(), 1, buffer.size(), fp);
                if (read == buffer.size()) {
                    const size_t applied = llama_state_set_data(handle->ctx, buffer.data(), buffer.size());
                    ok = (applied == buffer.size());
                }
            }
            fclose(fp);
        }
    }
#endif
    env->ReleaseStringUTFChars(jPath, path);
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_tokenize(
        JNIEnv* env, jobject /*thiz*/, jlong h, jstring jText) {
    auto* handle = reinterpret_cast<LlamaCtx*>(h);
    if (!handle) return env->NewIntArray(0);
    const char* text = env->GetStringUTFChars(jText, nullptr);
    std::vector<int> out;
#if LLAMA_STUB_MODE
    // Return codepoint count as fake token ids (not meaningful)
    if (text) {
        for (const char* p = text; *p; ++p) {
            out.push_back(static_cast<unsigned char>(*p));
        }
    }
#else
    if (handle->model && text) {
        const llama_vocab* vocab = llama_model_get_vocab(handle->model);
        std::vector<llama_token> toks;
        toks.resize(strlen(text) + 16);
        int n = llama_tokenize(llama_model_get_vocab(handle->model), text, (int)strlen(text), toks.data(), (int)toks.size(), true, false);
        if (n > 0) {
            toks.resize(n);
            out.reserve(n);
            for (int i = 0; i < n; ++i) out.push_back((int)toks[i]);
        }
    }
#endif
    env->ReleaseStringUTFChars(jText, text);
    jintArray arr = env->NewIntArray((jsize)out.size());
    if (!out.empty()) {
        env->SetIntArrayRegion(arr, 0, (jsize)out.size(), reinterpret_cast<const jint*>(out.data()));
    }
    return arr;
}

// RAG 시스템을 위한 동기식 completion 함수
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_llama_nativebridge_LlamaBridge_completion(
        JNIEnv* env, jobject /*thiz*/, jlong h,
        jstring jPrompt, jint numPredict, jfloat temperature, jfloat topP, jint topK,
        jfloat repeatPenalty, jint repeatLastN,
        jobjectArray jStopSequences) {
    ALOGD("completion(): Called with handle=%p", (void*)h);
    auto* handle = reinterpret_cast<LlamaCtx*>(h);
    if (!handle || !handle->ctx || !handle->model) {
        ALOGE("completion(): Invalid handle, ctx, or model (handle=%p, ctx=%p, model=%p)", 
              (void*)handle, handle ? (void*)handle->ctx : nullptr, handle ? (void*)handle->model : nullptr);
        return env->NewStringUTF("");
    }
    ALOGD("completion(): Starting completion, prompt length will be logged");

#if LLAMA_STUB_MODE
    (void) jPrompt; (void) numPredict; (void) temperature; (void) topP; (void) topK;
    (void) repeatPenalty; (void) repeatLastN; (void) jStopSequences;
    return env->NewStringUTF("{\"search_needed\": false, \"search_query\": null}");
#else
    // Lock mutex to prevent concurrent access to llama_context
    ALOGD("completion(): Acquiring mutex lock");
    std::lock_guard<std::mutex> lock(handle->ctx_mutex);
    ALOGD("completion(): Mutex lock acquired");
    
    const char* prompt = env->GetStringUTFChars(jPrompt, nullptr);
    std::string promptStr(prompt ? prompt : "");
    env->ReleaseStringUTFChars(jPrompt, prompt);
    ALOGD("completion(): Prompt length=%zu, first 200 chars: %.200s", promptStr.length(), promptStr.c_str());

    int n_predict = (numPredict > 0) ? numPredict : 256;
    float temp = (temperature > 0.0f) ? temperature : 0.3f;
    float top_p = (topP > 0.0f) ? topP : 0.9f;
    int top_k = (topK > 0) ? topK : 40;
    float rep_penalty = (repeatPenalty > 0.0f) ? repeatPenalty : 1.1f;
    int rep_last_n = (repeatLastN > 0) ? repeatLastN : 64;
    
    std::vector<std::string> stops;
    if (jStopSequences) {
        jsize len = env->GetArrayLength(jStopSequences);
        for (jsize i = 0; i < len; ++i) {
            jstring s = (jstring) env->GetObjectArrayElement(jStopSequences, i);
            const char* cs = env->GetStringUTFChars(s, nullptr);
            stops.emplace_back(cs ? cs : "");
            env->ReleaseStringUTFChars(s, cs);
            env->DeleteLocalRef(s);
        }
    }

    llama_context* ctx = handle->ctx;
    llama_model* model = handle->model;
    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Tokenize prompt
    ALOGD("completion(): Starting tokenization");
    std::vector<llama_token> prompt_tokens;
    prompt_tokens.resize(promptStr.size() + 16);
    bool has_bos = (promptStr.length() > 0 && promptStr.find("<|begin_of_text|>") == 0);
    int n_tokens = llama_tokenize(
        vocab,
        promptStr.c_str(),
        (int32_t)promptStr.size(),
        prompt_tokens.data(),
        (int32_t)prompt_tokens.size(),
        !has_bos,
        false
    );

    if (n_tokens < 0) {
        ALOGE("completion(): Tokenization failed");
        return env->NewStringUTF("");
    }
    ALOGD("completion(): Tokenization complete, n_tokens=%d", n_tokens);
    prompt_tokens.resize(n_tokens);

    // Create sampler
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    
    if (top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    if (temp > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
    }
    if (rep_last_n != 0 && rep_penalty > 0.0f && rep_penalty != 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(rep_last_n, rep_penalty, 0.15f, 0.15f));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(static_cast<uint32_t>(std::random_device{}())));

    // Evaluate prompt
    ALOGD("completion(): Starting prompt evaluation, n_tokens=%d", n_tokens);
    int n_past = 0;
    uint32_t chunk_size = llama_n_batch(ctx);
    ALOGD("completion(): chunk_size=%u", chunk_size);
    
    for (int cur = 0; cur < n_tokens; ) {
        ALOGD("completion(): Evaluating prompt chunk: cur=%d, remaining=%d", cur, n_tokens - cur);
        int remaining = n_tokens - cur;
        int n_cur = std::min(static_cast<int>(chunk_size), remaining);
        
        llama_batch batch = llama_batch_init(n_cur, 0, 1);
        if (!batch.token || !batch.seq_id) {
            llama_batch_free(batch);
            llama_sampler_free(smpl);
            return env->NewStringUTF("");
        }
        
        batch.n_tokens = n_cur;
        free(batch.pos);
        batch.pos = nullptr;
        
        for (int j = 0; j < n_cur; ++j) {
            batch.token[j] = prompt_tokens[cur + j];
            if (batch.seq_id[j]) {
                batch.seq_id[j][0] = 0;
            }
            batch.n_seq_id[j] = 1;
            batch.logits[j] = false;
        }
        
        // Enable logits for last token
        if (cur + n_cur == n_tokens && n_cur > 0) {
            batch.logits[n_cur - 1] = true;
        }
        
        ALOGD("completion(): Calling llama_decode() for prompt evaluation chunk cur=%d n_cur=%d", cur, n_cur);
        int decode_result = llama_decode(ctx, batch);
        ALOGD("completion(): llama_decode() returned %d for prompt evaluation chunk", decode_result);
        if (decode_result != 0) {
            ALOGE("completion(): llama_decode() failed for prompt evaluation chunk cur=%d n_cur=%d", cur, n_cur);
            llama_batch_free(batch);
            llama_sampler_free(smpl);
            return env->NewStringUTF("");
        }
        
        llama_batch_free(batch);
        n_past += n_cur;
        cur += n_cur;
        ALOGD("completion(): Prompt evaluation chunk complete, n_past=%d", n_past);
    }
    ALOGD("completion(): Prompt evaluation complete, n_past=%d", n_past);

    // Decode last token separately to get logits
    ALOGD("completion(): Decoding last token separately to get logits");
    if (n_tokens > 0) {
        llama_batch last_batch = llama_batch_init(1, 0, 1);
        if (last_batch.token && last_batch.seq_id) {
            last_batch.n_tokens = 1;
            last_batch.token[0] = prompt_tokens[n_tokens - 1];
            last_batch.logits[0] = true;
            free(last_batch.pos);
            last_batch.pos = nullptr;
            if (last_batch.seq_id[0]) {
                last_batch.seq_id[0][0] = 0;
                last_batch.n_seq_id[0] = 1;
            }
            ALOGD("completion(): Calling llama_decode() for last token");
            int last_decode_result = llama_decode(ctx, last_batch);
            ALOGD("completion(): llama_decode() returned %d for last token", last_decode_result);
            llama_batch_free(last_batch);
        }
    }
    ALOGD("completion(): Last token decode complete");

    // Generate tokens
    ALOGD("completion(): Starting token generation, n_predict=%d", n_predict);
    std::string generated;
    int n_gen = 0;
    uint32_t context_size = llama_n_ctx(ctx);
    ALOGD("completion(): context_size=%u, n_past=%d", context_size, n_past);

    while (n_past < context_size && n_gen < n_predict) {
        // Sample token
        int32_t logits_idx = 0;
        const float* logits = llama_get_logits_ith(ctx, logits_idx);
        if (!logits) {
            ALOGE("completion(): No logits available");
            break;
        }

        llama_token id = llama_sampler_sample(smpl, ctx, logits_idx);
        llama_sampler_accept(smpl, id);

        // Check for stop tokens
        if (id == 128009 || id == llama_vocab_eos(vocab) || id == 128001 || id == 128008) {
            break;
        }

        // Check stop sequences
        bool hitStop = false;
        for (const auto& stop : stops) {
            if (generated.find(stop) != std::string::npos) {
                hitStop = true;
                ALOGD("completion(): Stop sequence '%s' detected, breaking", stop.c_str());
                break;
            }
        }
        if (hitStop) break;
        
        // Check if JSON is complete (contains both "search_needed" and closing brace)
        if (generated.find("search_needed") != std::string::npos && 
            generated.find("}") != std::string::npos) {
            // Verify JSON completeness: should have both fields or at least closing brace
            size_t bracePos = generated.find_last_of("}");
            if (bracePos != std::string::npos && bracePos > generated.find("search_needed")) {
                ALOGD("completion(): JSON appears complete (contains 'search_needed' and '}'), breaking");
                break;
            }
        }

        // Convert token to text
        std::vector<char> piece(16, 0);
        int n_len = llama_token_to_piece(vocab, id, piece.data(), piece.size(), false, false);
        if (n_len < 0) break;
        if (static_cast<size_t>(n_len) >= piece.size()) {
            piece.resize(n_len + 1);
            n_len = llama_token_to_piece(vocab, id, piece.data(), piece.size(), false, false);
            if (n_len < 0) break;
        }
        piece.resize(n_len);
        
        // Filter special tokens
        std::string tokenText = filterSpecialTokensTokenLevel(std::string(piece.data(), piece.size()));
        if (!tokenText.empty() && id < 128000) {
            generated += tokenText;
        }

        // Decode generated token
        llama_batch gen_batch = llama_batch_init(1, 0, 1);
        if (!gen_batch.token || !gen_batch.seq_id) {
            llama_batch_free(gen_batch);
            break;
        }
        gen_batch.n_tokens = 1;
        gen_batch.token[0] = id;
        gen_batch.logits[0] = true;
        free(gen_batch.pos);
        gen_batch.pos = nullptr;
        if (gen_batch.seq_id[0]) {
            gen_batch.seq_id[0][0] = 0;
            gen_batch.n_seq_id[0] = 1;
        }

        if (llama_decode(ctx, gen_batch) != 0) {
            llama_batch_free(gen_batch);
            break;
        }
        llama_batch_free(gen_batch);
        
        n_past++;
        n_gen++;
    }

    llama_sampler_free(smpl);

    // Final filtering
    std::string finalResult = filterSpecialTokensTextLevel(generated);
    ALOGD("completion(): Generated %zu characters, final result length=%zu", generated.length(), finalResult.length());
    ALOGD("completion(): Final result: %.200s", finalResult.c_str());
    return env->NewStringUTF(finalResult.c_str());
#endif
}


