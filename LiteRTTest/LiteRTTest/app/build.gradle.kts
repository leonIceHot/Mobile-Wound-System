plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.ksp)
}

android {
    namespace = "com.example.literttest"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.literttest"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
}

dependencies {
    // LiteRT 推論引擎
    // 更新到最新的穩定版本
    implementation("com.google.ai.edge.litert:litert:2.1.5") // 最新穩定版

    // Android Core
    implementation("androidx.core:core-ktx:1.13.1") // 最新穩定版
    implementation("androidx.appcompat:appcompat:1.7.0") // 最新穩定版
    implementation("com.google.android.material:material:1.12.0") // 最新穩定版
    implementation("androidx.activity:activity-ktx:1.10.1") // 最新穩定版

    // CameraX (建議所有 CameraX 庫使用相同的主版本號)
    // 更新到最新的穩定版本
    implementation("androidx.camera:camera-core:1.3.3") // 最新穩定版
    implementation("androidx.camera:camera-camera2:1.3.3") // 最新穩定版
    implementation("androidx.camera:camera-lifecycle:1.3.3") // 最新穩定版
    implementation("androidx.camera:camera-view:1.3.3") // 最新穩定版

    // Compose 基礎 (假設你已在 version catalog 中定義了這些版本，此處使用佔位符)
    // 建議定期更新 Compose BOM 以保持所有 Compose 庫版本一致。
    implementation(libs.androidx.core.ktx) // 假設在 libs.versions.toml 中定義了最新版本
    implementation(libs.androidx.lifecycle.runtime.ktx) // 假設在 libs.versions.toml 中定義了最新版本
    implementation(libs.androidx.activity.compose) // 假設在 libs.versions.toml 中定義了最新版本
    implementation(platform("androidx.compose:compose-bom:2026.05.00")) // 請查詢官方最新 BOM
    implementation(libs.androidx.compose.ui) // 假設在 libs.versions.toml 中定義了最新版本
    implementation(libs.androidx.compose.ui.graphics) // 假設在 libs.versions.toml 中定義了最新版本
    implementation(libs.androidx.compose.ui.tooling.preview) // 假設在 libs.versions.toml 中定義了最新版本
    implementation(libs.androidx.compose.material3) // 假設在 libs.versions.toml 中定義了最新版本

    // Navigation Compose
    implementation("androidx.navigation:navigation-compose:2.8.1") // 最新穩定版

    // ViewModel & StateFlow
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7") // 最新穩定版
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7") // 最新穩定版

    // Room
    // 使用最新的穩定版本
    implementation("androidx.room:room-runtime:2.6.1") // 最新穩定版
    implementation("androidx.room:room-ktx:2.6.1") // 最新穩定版
    ksp("androidx.room:room-compiler:2.6.1") // 與 room-runtime 版本匹配

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1") // 最新穩定版
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1") // 最新穩定版

    // Testing (假設你已在 version catalog 中定義了這些版本)
    testImplementation(libs.junit) // 假設在 libs.versions.toml 中定義了最新版本
    androidTestImplementation(libs.androidx.junit) // 假設在 libs.versions.toml 中定義了最新版本
    androidTestImplementation(libs.androidx.espresso.core) // 假設在 libs.versions.toml 中定義了最新版本
    androidTestImplementation(platform("androidx.compose:compose-bom:2026.05.00")) // 與 Compose BOM 保持一致
    androidTestImplementation(libs.androidx.compose.ui.test.junit4) // 假設在 libs.versions.toml 中定義了最新版本
    debugImplementation(libs.androidx.compose.ui.tooling) // 假設在 libs.versions.toml 中定義了最新版本
    debugImplementation(libs.androidx.compose.ui.test.manifest) // 假設在 libs.versions.toml 中定義了最新版本
    // Coil 圖片加載
    implementation("io.coil-kt:coil-compose:2.4.0")
// Material 3
    implementation("androidx.compose.material3:material3:1.1.1")
    implementation("androidx.compose.material:material-icons-core:1.5.4")
    implementation("androidx.compose.material:material-icons-extended:1.5.4")
}
