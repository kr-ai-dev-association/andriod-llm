package com.example.llama.data

import android.content.Context
import android.preference.PreferenceManager
import android.util.Log

object ModelPathStore {
	private const val KEY_MODEL_PATH = "model_path_override"

	fun getOverridePath(context: Context): String? {
		val prefs = PreferenceManager.getDefaultSharedPreferences(context)
		val v = prefs.getString(KEY_MODEL_PATH, null)
		Log.d("BanyaChat", "ModelPathStore.getOverridePath(): $v")
		return v
	}

	fun setOverridePath(context: Context, path: String?) {
		val prefs = PreferenceManager.getDefaultSharedPreferences(context)
		Log.d("BanyaChat", "ModelPathStore.setOverridePath(): $path")
		prefs.edit().putString(KEY_MODEL_PATH, path).apply()
	}
}


