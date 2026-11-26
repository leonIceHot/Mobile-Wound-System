package com.example.literttest.ui.theme
import androidx.compose.ui.graphics.Color
import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// ===== 暗色主題（主要使用）=====
private val DarkColorScheme = darkColorScheme(
    primary = NeonGreen,
    onPrimary = TextOnGreen,
    primaryContainer = NeonGreenDark,
    onPrimaryContainer = TextPrimary,

    secondary = NeonGreenLight,
    onSecondary = TextOnGreen,

    tertiary = StatusWarning,
    onTertiary = TextOnGreen,

    background = DarkBackground,
    onBackground = TextPrimary,

    surface = DarkSurface,
    onSurface = TextPrimary,
    surfaceVariant = DarkSurfaceVariant,
    onSurfaceVariant = TextSecondary,

    error = StatusDanger,
    onError = TextPrimary,

    outline = DividerColor,
    outlineVariant = BorderDashed,
)

// ===== 亮色主題（備用，但本 APP 主要走暗色）=====
private val LightColorScheme = lightColorScheme(
    primary = NeonGreenDark,
    onPrimary = TextPrimary,
    background = Color(0xFFF5F7FA),
    onBackground = Color(0xFF0A1929),
    surface = Color(0xFFFFFFFF),
    onSurface = Color(0xFF0A1929),
    error = StatusDanger,
)

@Composable
fun LiteRTTestTheme(
    darkTheme: Boolean = true,   // 🔥 強制使用暗色主題（符合設計圖）
    // 關閉動態顏色（Android 12+ 會吃掉我們的自訂色），確保設計一致
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalView.current.context
            if (darkTheme) {
                androidx.compose.material3.dynamicDarkColorScheme(context)
            } else {
                androidx.compose.material3.dynamicLightColorScheme(context)
            }
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    // 設定狀態列顏色 & 圖示顏色
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.background.toArgb()
            window.navigationBarColor = colorScheme.background.toArgb()
            WindowCompat.getInsetsController(window, view).apply {
                isAppearanceLightStatusBars = !darkTheme
                isAppearanceLightNavigationBars = !darkTheme
            }
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}