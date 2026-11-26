package com.example.literttest.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.example.literttest.ui.viewmodel.WoundDetectionViewModel

private val DarkRed = Color(0xFFB71C1C)
private val LightGray = Color(0xFFF5F5F5)
private val DarkGray = Color(0xFF333333)
private val GradeColors = mapOf(
    "WCC_0" to Pair(Color(0xFF4CAF50), "正常皮膚"),
    "WCC_1" to Pair(Color(0xFFFFC107), "輕微傷口"),
    "WCC_2" to Pair(Color(0xFFFF9800), "中等傷口"),
    "WCC_3" to Pair(Color(0xFFF44336), "嚴重傷口"),
    "No Wound" to Pair(Color(0xFF2196F3), "無傷口")
)

val SuggestionMap = mapOf(
    "WCC_0" to listOf(
        "✓ 皮膚正常，無明顯傷口",
        "✓ 繼續保持日常護理",
        "✓ 定期檢查皮膚狀況"
    ),
    "WCC_1" to listOf(
        "⚠ 檢測到輕微傷口",
        "• 保持傷口清潔乾燥",
        "• 使用溫和的清潔產品",
        "• 若症狀加重請就醫"
    ),
    "WCC_2" to listOf(
        "⚠ 檢測到中等傷口",
        "• 定期清潔和消毒傷口",
        "• 使用適當的傷口敷料",
        "• 避免摩擦和刺激",
        "• 建議諮詢醫療專業人士"
    ),
    "WCC_3" to listOf(
        "🚨 檢測到嚴重傷口",
        "• 立即停止使用該區域",
        "• 進行專業醫療評估",
        "• 可能需要特殊治療",
        "• 請儘快聯繫醫療人員"
    ),
    "No Wound" to listOf(
        "✓ 未檢測到明顯傷口",
        "✓ 皮膚狀況良好",
        "✓ 繼續做好日常護理"
    )
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SuggestionScreen(
    navController: NavController,
    viewModel: WoundDetectionViewModel = viewModel()
) {
    val result = viewModel.detectionResult

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("健康建議") },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "返回",
                            tint = Color.White
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkRed,
                    titleContentColor = Color.White,
                    navigationIconContentColor = Color.White
                )
            )
        }
    ) { paddingValues ->
        if (result == null) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues),
                contentAlignment = Alignment.Center
            ) {
                Text("沒有檢測結果")
            }
        } else {
            SuggestionContent(
                result = result,
                navController = navController,
                modifier = Modifier.padding(paddingValues)
            )
        }
    }
}

@Composable
private fun SuggestionContent(
    result: com.example.literttest.ml.DetectionResult,
    navController: NavController,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .background(LightGray)
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // ── 1. 大標題 ─────────────────────────────────
        Text(
            text = "個性化健康建議",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            color = DarkGray,
            modifier = Modifier.padding(bottom = 24.dp)
        )

        // ── 2. 傷口等級摘要 ────────────────────────────
        val (color, statusText) = GradeColors[result.woundGrade] ?:
        (Color.Gray to "未知")

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 20.dp),
            colors = CardDefaults.cardColors(containerColor = color),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = statusText,
                    fontSize = 18.sp,
                    color = Color.White.copy(alpha = 0.9f),
                    modifier = Modifier.padding(bottom = 8.dp)
                )

                Text(
                    text = result.woundGrade,
                    fontSize = 32.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
            }
        }

        // ── 3. 建議內容 ────────────────────────────────
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 20.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(modifier = Modifier.padding(20.dp)) {
                Text(
                    text = "建議清單",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = DarkGray,
                    modifier = Modifier.padding(bottom = 16.dp)
                )

                val suggestions = SuggestionMap[result.woundGrade] ?: listOf("無相關建議")

                suggestions.forEach { suggestion ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 10.dp),
                        verticalAlignment = Alignment.Top
                    ) {
                        Text(
                            text = "•",
                            fontSize = 20.sp,
                            color = color,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier
                                .padding(end = 12.dp)
                                .padding(top = 2.dp)
                        )

                        Text(
                            text = suggestion,
                            fontSize = 14.sp,
                            color = DarkGray,
                            lineHeight = 20.sp
                        )
                    }
                }
            }
        }

        // ── 4. 注意事項 ────────────────────────────────
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 20.dp),
            colors = CardDefaults.cardColors(
                containerColor = Color(0xFFFFF3CD)
            ),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "⚠ 重要提示",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF856404),
                    modifier = Modifier.padding(bottom = 8.dp)
                )

                Text(
                    text = "本應用的分析結果僅供參考，不能替代專業醫療診斷。若傷口情況嚴重或症狀持續惡化，請立即就醫。",
                    fontSize = 13.sp,
                    color = Color(0xFF856404),
                    lineHeight = 19.sp
                )
            }
        }

        // ── 5. 按鈕組 ─────────────────────────────────
        Spacer(modifier = Modifier.height(12.dp))

        Button(
            onClick = { navController.navigate("camera") },
            modifier = Modifier
                .fillMaxWidth()
                .height(48.dp)
                .padding(bottom = 12.dp),
            colors = ButtonDefaults.buttonColors(containerColor = DarkRed),
            shape = RoundedCornerShape(8.dp)
        ) {
            Text("重新檢測", color = Color.White, fontWeight = FontWeight.Bold)
        }

        OutlinedButton(
            onClick = { navController.navigate("home") },
            modifier = Modifier
                .fillMaxWidth()
                .height(48.dp),
            shape = RoundedCornerShape(8.dp)
        ) {
            Text("返回首頁")
        }

        Spacer(modifier = Modifier.height(16.dp))
    }
}
