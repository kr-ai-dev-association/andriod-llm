package com.example.llama

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import com.example.llama.ui.ChatScreen
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat

class MainActivity : ComponentActivity() {
	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)
		enableEdgeToEdge()
		// Fullscreen immersive: hide system bars (navigation/home), allow swipe to reveal temporarily
		WindowCompat.setDecorFitsSystemWindows(window, false)
		hideSystemBars()
		setContent {
			MaterialTheme {
				SafeAreaSurface {
					ChatScreen()
				}
			}
		}
	}

	override fun onWindowFocusChanged(hasFocus: Boolean) {
		super.onWindowFocusChanged(hasFocus)
		if (hasFocus) {
			hideSystemBars()
		}
	}

	private fun hideSystemBars() {
		val controller = WindowInsetsControllerCompat(window, window.decorView)
		controller.systemBarsBehavior =
			WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
		controller.hide(WindowInsetsCompat.Type.systemBars())
	}
}

@Composable
private fun SafeAreaSurface(content: @Composable () -> Unit) {
	Surface(
		color = MaterialTheme.colorScheme.background,
		content = content
	)
}


