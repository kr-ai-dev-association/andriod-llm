package com.example.llama

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.res.painterResource
import com.example.llama.ui.ChatScreen
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import kotlinx.coroutines.delay

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
					SplashScreen {
						ChatScreen()
					}
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

@Composable
private fun SplashScreen(content: @Composable () -> Unit) {
	var showSplash by remember { mutableStateOf(true) }
	
	LaunchedEffect(Unit) {
		delay(2500) // 페이드 인(500ms) + 표시(1500ms) + 페이드 아웃(500ms) = 2.5초
		showSplash = false
	}
	
	if (showSplash) {
		SplashContent()
	} else {
		content()
	}
}

@Composable
private fun SplashContent() {
	var alpha by remember { mutableStateOf(0f) }
	
	LaunchedEffect(Unit) {
		// 페이드 인 (0.5초)
		alpha = 1f
		delay(1500) // 1.5초 표시
		// 페이드 아웃 (0.5초)
		alpha = 0f
	}
	
	val animatedAlpha by animateFloatAsState(
		targetValue = alpha,
		animationSpec = tween(durationMillis = 500),
		label = "splash_alpha"
	)
	
	Box(
		modifier = Modifier
			.fillMaxSize()
			.alpha(animatedAlpha),
		contentAlignment = Alignment.Center
	) {
		Image(
			painter = painterResource(id = R.drawable.banya_logo_square_mint),
			contentDescription = "Banya Logo",
			modifier = Modifier.fillMaxSize(0.5f)
		)
	}
}


