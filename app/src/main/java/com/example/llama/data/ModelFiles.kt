package com.example.llama.data

import android.content.Context
import java.io.File
import java.io.IOException

object ModelFiles {
	fun ensureModelDir(context: Context): File {
		val dir = File(context.filesDir, "models")
		if (!dir.exists()) {
			dir.mkdirs()
		}
		return dir
	}

	@Throws(IOException::class)
	fun copyAssetIfPresent(context: Context, assetName: String, outName: String): File? {
		val am = context.assets
		val candidates = am.list("")?.toList() ?: emptyList()
		if (!candidates.contains(assetName)) return null
		val outDir = ensureModelDir(context)
		val outFile = File(outDir, outName)
		if (outFile.exists() && outFile.length() > 0) return outFile
		am.open(assetName).use { input ->
			outFile.outputStream().use { output ->
				input.copyTo(output)
			}
		}
		return outFile
	}
}


