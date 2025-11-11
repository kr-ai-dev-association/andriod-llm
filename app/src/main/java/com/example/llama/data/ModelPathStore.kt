package com.example.llama.data

import android.content.Context
import android.util.Log

object ModelPathStore {
	private const val PREFS_NAME = "banya_chat_model"
	private const val KEY_MODEL_PATH = "model_path_override"
	private const val KEY_MODEL_METADATA = "model_metadata_json"

	private fun prefs(context: Context) =
		context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

	fun getOverridePath(context: Context): String? {
		val v = prefs(context).getString(KEY_MODEL_PATH, null)
		Log.d("BanyaChat", "ModelPathStore.getOverridePath(): $v")
		return v
	}

	fun setOverridePath(context: Context, path: String?) {
		Log.d("BanyaChat", "ModelPathStore.setOverridePath(): $path")
		prefs(context).edit().putString(KEY_MODEL_PATH, path).apply()
	}

	fun setModelMetadata(context: Context, metadataJson: String?) {
		prefs(context).edit().putString(KEY_MODEL_METADATA, metadataJson).apply()
	}

	fun getModelMetadata(context: Context): String? {
		return prefs(context).getString(KEY_MODEL_METADATA, null)
	}
}


