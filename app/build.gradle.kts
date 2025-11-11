plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

import org.gradle.api.GradleException
import org.gradle.internal.os.OperatingSystem

val hostTag = when {
    OperatingSystem.current().isMacOsX -> "darwin-x86_64"
    OperatingSystem.current().isWindows -> "windows-x86_64"
    OperatingSystem.current().isLinux -> "linux-x86_64"
    else -> throw GradleException("Unsupported OS for shader toolchain")
}

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

        externalNativeBuild {
            cmake {
                cppFlags("-O3", "-DNDEBUG", "-fvisibility=hidden")
                // Use GGML_ prefixed flags for subproject options
                arguments(
                    "-DUSE_LLAMA=ON",
                    "-DGGML_VULKAN=ON",
                    "-DGGML_VULKAN_USE_VOLK=ON",
                    "-DGGML_K_QUANTS=ON",
                    "-DCMAKE_MAKE_PROGRAM=${android.sdkDirectory}/cmake/3.22.1/bin/ninja",
                    "-DCMAKE_PROGRAM_PATH=${android.sdkDirectory}/cmake/3.22.1/bin",
                    "-DVulkan_GLSLC_EXECUTABLE=${android.ndkDirectory}/shader-tools/$hostTag/glslc",
                    "-DVulkan_INCLUDE_DIR=${project.projectDir}/third_party/Vulkan-Headers/include",
                    "-DCMAKE_INCLUDE_PATH=${project.projectDir}/third_party/Vulkan-Headers/include"
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
}


