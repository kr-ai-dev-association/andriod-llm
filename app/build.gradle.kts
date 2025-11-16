plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

import org.gradle.api.GradleException
import org.gradle.internal.os.OperatingSystem
import java.util.Properties
import java.io.FileInputStream

val hostTag = when {
    OperatingSystem.current().isMacOsX -> "darwin-x86_64"
    OperatingSystem.current().isWindows -> "windows-x86_64"
    OperatingSystem.current().isLinux -> "linux-x86_64"
    else -> throw GradleException("Unsupported OS for shader toolchain")
}

// local.properties에서 API 키 읽기
val localProperties = Properties()
val localPropertiesFile = rootProject.file("local.properties")
if (localPropertiesFile.exists()) {
    localProperties.load(FileInputStream(localPropertiesFile))
}
val tavilyApiKey = localProperties.getProperty("TAVILY_API_KEY") ?: ""

android {
    namespace = "com.example.llama"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.llama"
		minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        
        // Tavily API 키를 BuildConfig에 추가
        buildConfigField("String", "TAVILY_API_KEY", "\"$tavilyApiKey\"")

        externalNativeBuild {
            cmake {
                cppFlags("-O3", "-DNDEBUG", "-fvisibility=hidden")
                // Use GGML_ prefixed flags for subproject options
                arguments(
                    "-DUSE_LLAMA=ON",
                    "-DGGML_VULKAN=OFF",  // Disabled: Testing OpenCL with android_dlopen_ext
                    "-DGGML_OPENCL=ON",   // Enabled: Using android_dlopen_ext to bypass namespace restrictions
                    "-DGGML_K_QUANTS=ON",
                    "-DOPENCL_INCLUDE_DIR=${project.projectDir}/../third_party/OpenCL-Headers",
                    "-DCMAKE_MAKE_PROGRAM=${android.sdkDirectory}/cmake/3.22.1/bin/ninja",
                    "-DCMAKE_PROGRAM_PATH=${android.sdkDirectory}/cmake/3.22.1/bin"
                )
            }
        }
        ndk {
            abiFilters.add("arm64-v8a")
        }
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
        }
    }

    buildFeatures {
        compose = true
        prefab = false
        buildConfig = true
    }

	compileOptions {
		sourceCompatibility = JavaVersion.VERSION_17
		targetCompatibility = JavaVersion.VERSION_17
	}

    // composeOptions no longer needed with Kotlin 2.0 Compose plugin

    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    kotlinOptions {
        jvmTarget = "17"
        freeCompilerArgs = freeCompilerArgs + listOf("-Xjvm-default=all")
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2024.10.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
    implementation("androidx.compose.material3:material3:1.3.0")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.6")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    
    // RAG 시스템을 위한 네트워크 라이브러리
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")
    implementation("com.google.code.gson:gson:2.10.1")
}


