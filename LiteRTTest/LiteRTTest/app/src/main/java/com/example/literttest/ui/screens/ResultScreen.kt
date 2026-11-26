package com.example.literttest.ui.screens

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.literttest.ml.DetectionResult
import com.example.literttest.ml.WoundBox
import android.graphics.Paint as AndroidPaint
import androidx.compose.ui.graphics.nativeCanvas

// ── 等級顏色對應 ──────────────────────────────────────────
private fun gradeColor(grade: String): Color = when (grade) {
    "WCC_0" -> Color(0xFF4CAF50)   // 綠
    "WCC_1" -> Color(0xFFFFEB3B)   // 黃
    "WCC_2" -> Color(0xFFFF9800)   // 橙
    "WCC_3" -> Color(0xFFF44336)   // 紅
    else    -> Color(0xFF9E9E9E)   // 灰
}

// ── 等級中文說明 ──────────────────────────────────────────
private fun gradeLabel(grade: String): String = when (grade) {
    "WCC_0" -> "WCC 0"
    "WCC_1" -> "WCC 1"
    "WCC_2" -> "WCC 2"
    "WCC_3" -> "WCC 3"
    "WCC_4" -> "WCC 4"
    "OTHER" -> "其他類別"
    "No Wound" -> "未偵測到傷口（建議人工確認）"
    else -> grade
}

// ── 主畫面 ────────────────────────────────────────────────
@Composable
fun ResultScreen(
    bitmap: Bitmap,
    result: DetectionResult,
    onRetake: () -> Unit,
    onHome: () -> Unit
) {
    val scrollState = rememberScrollState()
    val mainColor = gradeColor(result.woundGrade)

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF121212))
            .verticalScroll(scrollState)
            .padding(horizontal = 20.dp, vertical = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        // ── 標題 ──────────────────────────────────────────
        Text(
            text = "分析結果",
            fontSize = 26.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )

        Spacer(modifier = Modifier.height(20.dp))

        // ── 圖片 + Bounding Box ───────────────────────────
        ImageWithBoxes(
            bitmap = bitmap,
            boxes = result.detectedBoxes
        )

        Spacer(modifier = Modifier.height(20.dp))

        // ── 等級卡片 ──────────────────────────────────────
        GradeCard(
            grade = result.woundGrade,
            confidence = result.confidence,
            color = mainColor
        )

        Spacer(modifier = Modifier.height(16.dp))

        // ── hasWound 狀態列 ───────────────────────────────
        WoundStatusRow(hasWound = result.hasWound)

        Spacer(modifier = Modifier.height(16.dp))

        // ── 各等級分數條 ──────────────────────────────────
        ScoreBarSection(gradeScores = result.gradeScores)

        Spacer(modifier = Modifier.height(16.dp))

        // ── 偵測框統計 ────────────────────────────────────
        if (result.detectedBoxes.isNotEmpty()) {
            BoxStatsSection(boxes = result.detectedBoxes)
            Spacer(modifier = Modifier.height(16.dp))
        }

        // ── 推論時間 ──────────────────────────────────────
        LatencyRow(latencyMs = result.latencyMs)

        Spacer(modifier = Modifier.height(28.dp))

        // ── 按鈕區 ────────────────────────────────────────
        ButtonRow(onRetake = onRetake, onHome = onHome)
    }
}

// ── 圖片 + 繪製 Bounding Box ──────────────────────────────


@Composable
private fun ImageWithBoxes(
    bitmap: Bitmap,
    boxes: List<WoundBox>
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            .clip(RoundedCornerShape(16.dp))
    ) {
        Image(
            bitmap = bitmap.asImageBitmap(),
            contentDescription = "傷口圖片",
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        if (boxes.isNotEmpty()) {
            Canvas(modifier = Modifier.fillMaxSize()) {

                // 對應 ContentScale.Crop 的實際縮放比例
                val scale = maxOf(
                    size.width / bitmap.width.toFloat(),
                    size.height / bitmap.height.toFloat()
                )

                // 置中裁切所產生的偏移量
                val offsetX =
                    (size.width - bitmap.width * scale) / 2f
                val offsetY =
                    (size.height - bitmap.height * scale) / 2f

                val labelHeight = 22.dp.toPx()
                val labelPadding = 6.dp.toPx()

                val textPaint = AndroidPaint(
                    AndroidPaint.ANTI_ALIAS_FLAG
                ).apply {
                    color = android.graphics.Color.WHITE
                    textSize = 11.sp.toPx()
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                }

                boxes.forEach { box ->
                    val left = box.left * scale + offsetX
                    val top = box.top * scale + offsetY
                    val width = box.width * scale
                    val height = box.height * scale

                    // 畫出傷口 Bounding Box
                    drawRect(
                        color = Color(0xFFF44336),
                        topLeft = Offset(left, top),
                        size = Size(width, height),
                        style = Stroke(width = 3.dp.toPx())
                    )

                    val label = "${"%.0f".format(box.score * 100f)}%"
                    val labelWidth = maxOf(
                        48.dp.toPx(),
                        textPaint.measureText(label) + 2f * labelPadding
                    )

                    val labelX = left.coerceIn(
                        0f,
                        (size.width - labelWidth).coerceAtLeast(0f)
                    )
                    val labelY = (top - labelHeight).coerceIn(
                        0f,
                        (size.height - labelHeight).coerceAtLeast(0f)
                    )

                    drawRect(
                        color = Color(0xCCF44336),
                        topLeft = Offset(labelX, labelY),
                        size = Size(labelWidth, labelHeight)
                    )

                    drawContext.canvas.nativeCanvas.drawText(
                        label,
                        labelX + labelPadding,
                        labelY + labelHeight * 0.72f,
                        textPaint
                    )
                }
            }
        }
    }
}


// ── 等級卡片 ──────────────────────────────────────────────
@Composable
private fun GradeCard(grade: String, confidence: Float, color: Color) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1E1E1E))
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "傷口等級",
                fontSize = 13.sp,
                color = Color(0xFF9E9E9E)
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = gradeLabel(grade),
                fontSize = 22.sp,
                fontWeight = FontWeight.Bold,
                color = color,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(12.dp))

            // 信心度進度條
            Text(
                text = "信心度：${"%.1f".format(confidence * 100)}%",
                fontSize = 13.sp,
                color = Color(0xFFBDBDBD)
            )

            Spacer(modifier = Modifier.height(6.dp))

            LinearProgressIndicator(
                progress = { confidence },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp)),
                color = color,
                trackColor = Color(0xFF333333)
            )
        }
    }
}

// ── hasWound 狀態列 ───────────────────────────────────────
@Composable
private fun WoundStatusRow(hasWound: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Color(0xFF1E1E1E))
            .padding(horizontal = 20.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = "傷口偵測",
            fontSize = 14.sp,
            color = Color(0xFF9E9E9E),
            modifier = Modifier.weight(1f)
        )
        val (statusText, statusColor) = if (hasWound)
            "✅  偵測到傷口" to Color(0xFFF44336)
        else
            "✅  未偵測到傷口" to Color(0xFF4CAF50)

        Text(
            text = statusText,
            fontSize = 14.sp,
            fontWeight = FontWeight.SemiBold,
            color = statusColor
        )
    }
}

// ── 各等級分數條 ──────────────────────────────────────────
@Composable
private fun ScoreBarSection(gradeScores: Map<String, Float>) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1E1E1E))
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp)
        ) {
            Text(
                text = "各等級分數",
                fontSize = 14.sp,
                fontWeight = FontWeight.SemiBold,
                color = Color.White
            )

            Spacer(modifier = Modifier.height(12.dp))

            val order = listOf(
                "OTHER",
                "WCC_0",
                "WCC_1",
                "WCC_2",
                "WCC_3",
                "WCC_4"
            )

            order.forEach { key ->
                val score = gradeScores[key] ?: 0f
                val barColor = gradeColor(key)

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = key,
                        fontSize = 12.sp,
                        color = Color(0xFFBDBDBD),
                        modifier = Modifier.width(52.dp)
                    )

                    LinearProgressIndicator(
                        progress = { score },
                        modifier = Modifier
                            .weight(1f)
                            .height(8.dp)
                            .clip(RoundedCornerShape(4.dp)),
                        color = barColor,
                        trackColor = Color(0xFF333333)
                    )

                    Spacer(modifier = Modifier.width(8.dp))

                    Text(
                        text = "${"%.1f".format(score * 100)}%",
                        fontSize = 12.sp,
                        color = Color(0xFFBDBDBD),
                        modifier = Modifier.width(44.dp),
                        textAlign = TextAlign.End
                    )
                }
            }
        }
    }
}

// ── 偵測框統計 ────────────────────────────────────────────
@Composable
private fun BoxStatsSection(boxes: List<WoundBox>) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1E1E1E))
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp)
        ) {
            Text(
                text = "偵測框資訊（共 ${boxes.size} 個）",
                fontSize = 14.sp,
                fontWeight = FontWeight.SemiBold,
                color = Color.White
            )

            Spacer(modifier = Modifier.height(10.dp))

            boxes.forEachIndexed { idx, box ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 3.dp),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "框 ${idx + 1}",
                        fontSize = 12.sp,
                        color = Color(0xFF9E9E9E)
                    )
                    Text(
                        text = "信心：${"%.1f".format(box.score * 100)}%  " +
                                "W:${"%.0f".format(box.width)}  " +
                                "H:${"%.0f".format(box.height)}",
                        fontSize = 12.sp,
                        color = Color(0xFFBDBDBD)
                    )
                }
                if (idx < boxes.size - 1) {
                    Divider(color = Color(0xFF2C2C2C), thickness = 0.5.dp)
                }
            }
        }
    }
}

// ── 推論時間列 ────────────────────────────────────────────
@Composable
private fun LatencyRow(latencyMs: Long) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Color(0xFF1E1E1E))
            .padding(horizontal = 20.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = "⚡ 推論時間",
            fontSize = 14.sp,
            color = Color(0xFF9E9E9E),
            modifier = Modifier.weight(1f)
        )
        Text(
            text = "${latencyMs} ms",
            fontSize = 14.sp,
            fontWeight = FontWeight.SemiBold,
            color = Color(0xFF64B5F6)
        )
    }
}

// ── 按鈕區 ────────────────────────────────────────────────
@Composable
private fun ButtonRow(onRetake: () -> Unit, onHome: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        OutlinedButton(
            onClick = onRetake,
            modifier = Modifier
                .weight(1f)
                .height(52.dp),
            shape = RoundedCornerShape(14.dp),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.White)
        ) {
            Text("📷  重新拍攝", fontSize = 15.sp)
        }

        Button(
            onClick = onHome,
            modifier = Modifier
                .weight(1f)
                .height(52.dp),
            shape = RoundedCornerShape(14.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1976D2))
        ) {
            Text("🏠  回首頁", fontSize = 15.sp, color = Color.White)
        }
    }
}
