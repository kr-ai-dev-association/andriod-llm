package com.example.llama.nativebridge

interface TokenCallback {
	fun onToken(token: String)
	fun onCompleted()
	fun onError(message: String)
}


