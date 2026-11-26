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
import com.example.literttest.ml.DetectionResult
import com.example.literttest.ui.viewmodel.WoundDetectionViewModel

private val DarkRed = Color(0xFFB71C1C)
private val LightGray = Color(0xFFF5F5F5)
private val DarkGray = Color(0xFF333333)
private val GradeColors = mapOf(
    "WCC_0" to Color(0xFF4CAF50),  // 綠色 - 正常
    "WCC_1" to Color(0xFFFFC107),  // 黃色 - 輕微
    "WCC_2" to Color(0xFFFF9800),  // 橙色 - 中等
    "WCC_3" to Color(0xFFF44336),  // 紅色 - 嚴重
    "No Wound" to Color(0xFF2196F3) // 藍色 - 無傷口
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AnalysisScreen(
    navController: NavController,
    viewModel: WoundDetectionViewModel = viewModel()
) {
    val result = viewModel.detectionResult

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("分析結果") },
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
            AnalysisContent(
                result = result,
                navController = navController,
                modifier = Modifier.padding(paddingValues)
            )
        }
    }
}

@Composable
private fun AnalysisContent(
    result: DetectionResult,
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
            text = "傷口分析報告",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            color = DarkGray,
            modifier = Modifier.padding(bottom = 24.dp)
        )

        // ── 2. 主結果卡片 ──────────────────────────────
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            colors = CardDefaults.cardColors(
                containerColor = GradeColors[result.woundGrade] ?: Color.Gray
            ),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "傷口等級",
                    fontSize = 14.sp,
                    color = Color.White.copy(alpha = 0.8f),
                    modifier = Modifier.padding(bottom = 8.dp)
                )

                Text(
                    text = result.woundGrade,
                    fontSize = 36.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                    modifier = Modifier.padding(bottom = 12.dp)
                )

                Text(
                    text = "信心度: ${String.format("%.1f%%", result.confidence * 100)}",
                    fontSize = 16.sp,
                    color = Color.White.copy(alpha = 0.9f)
                )
            }
        }

        // ── 3. 檢測框資訊 ──────────────────────────────
        if (result.hasWound && result.detectedBoxes.isNotEmpty()) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "檢測框數量",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = DarkGray
                    )
                    Text(
                        text = "${result.detectedBoxes.size} 個傷口",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = DarkRed,
                        modifier = Modifier.padding(top = 8.dp)
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    // 顯示每個框的詳細資訊
                    result.detectedBoxes.forEachIndexed { index, box ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 6.dp)
                                .background(
                                    LightGray,
                                    RoundedCornerShape(6.dp)
                                )
                                .padding(8.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "框 ${index + 1}",
                                fontSize = 12.sp,
                                color = DarkGray
                            )
                            Text(
                                text = "信心度: ${String.format("%.2f", box.score)}",
                                fontSize = 12.sp,
                                color = DarkRed,
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                    }
                }
            }
        }

        // ── 4. 等級概率分佈 ────────────────────────────
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "等級概率分佈",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = DarkGray,
                    modifier = Modifier.padding(bottom = 12.dp)
                )

                result.gradeScores.forEach { (grade, score) ->
                    Column(modifier = Modifier.padding(vertical = 8.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = grade,
                                fontSize = 12.sp,
                                color = DarkGray,
                                fontWeight = FontWeight.SemiBold,
                                modifier = Modifier.weight(1f)
                            )
                            Text(
                                text = "${String.format("%.1f%%", score * 100)}",
                                fontSize = 12.sp,
                                color = DarkRed,
                                fontWeight = FontWeight.Bold
                            )
                        }
                        LinearProgressIndicator(
                            progress = score,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(6.dp)
                                .padding(top = 4.dp),
                            color = GradeColors[grade] ?: Color.Gray,
                            trackColor = LightGray,
                            strokeCap = androidx.compose.ui.graphics.StrokeCap.Round
                        )
                    }
                }
            }
        }

        // ── 5. 推論時間 ────────────────────────────────
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 24.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            shape = RoundedCornerShape(12.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "推論時間",
                    fontSize = 14.sp,
                    color = DarkGray
                )
                Text(
                    text = "${result.latencyMs}ms",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = DarkRed
                )
            }
        }

        // ── 6. 按鈕組 ─────────────────────────────────
        Button(
            onClick = { navController.navigate("suggestion") },
            modifier = Modifier
                .fillMaxWidth()
                .height(48.dp)
                .padding(bottom = 12.dp),
            colors = ButtonDefaults.buttonColors(containerColor = DarkRed),
            shape = RoundedCornerShape(8.dp)
        ) {
            Text("查看健康建議", color = Color.White, fontWeight = FontWeight.Bold)
        }

        OutlinedButton(
            onClick = { navController.navigate("camera") },
            modifier = Modifier
                .fillMaxWidth()
                .height(48.dp),
            shape = RoundedCornerShape(8.dp)
        ) {
            Text("重新檢測")
        }

        Spacer(modifier = Modifier.height(16.dp))
    }
}
