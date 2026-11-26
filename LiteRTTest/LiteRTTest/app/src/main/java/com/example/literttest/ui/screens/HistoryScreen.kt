
package com.example.literttest.ui.screens

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import com.example.literttest.ml.DetectionResult
import com.example.literttest.ml.WoundBox
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

private val HistoryRed = Color(0xFFB71C1C)

// ── 歷史紀錄資料結構 ─────────────────────────────────
data class HistoryRecord(
    val id: String,
    val date: String,
    val source: String,
    val woundGrade: String,
    val confidence: Float,
    val hasWound: Boolean,
    val imagePath: String,
    val gradeScores: Map<String, Float>,
    val detectedBoxes: List<WoundBox>,
    val latencyMs: Long
)

// ── 歷史紀錄永久儲存管理 ─────────────────────────────
object WoundHistoryStore {

    private const val PREF_NAME = "wound_history"
    private const val KEY_RECORDS = "records"

    private fun imageDir(context: Context): File {
        return File(context.applicationContext.filesDir, "wound_history")
    }

    private fun recordsToJson(records: List<HistoryRecord>): String {
        val array = JSONArray()

        records.forEach { record ->
            val scores = JSONObject()
            record.gradeScores.forEach { (name, score) ->
                scores.put(name, score.toDouble())
            }

            val boxes = JSONArray()
            record.detectedBoxes.forEach { box ->
                boxes.put(JSONObject().apply {
                    put("left", box.left.toDouble())
                    put("top", box.top.toDouble())
                    put("right", box.right.toDouble())
                    put("bottom", box.bottom.toDouble())
                    put("score", box.score.toDouble())
                })
            }

            array.put(JSONObject().apply {
                put("id", record.id)
                put("date", record.date)
                put("source", record.source)
                put("woundGrade", record.woundGrade)
                put("confidence", record.confidence.toDouble())
                put("hasWound", record.hasWound)
                put("imagePath", record.imagePath)
                put("gradeScores", scores)
                put("detectedBoxes", boxes)
                put("latencyMs", record.latencyMs)
            })
        }

        return array.toString()
    }

    @Synchronized
    fun getAll(context: Context): List<HistoryRecord> {
        val prefs = context.applicationContext.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        val json = prefs.getString(KEY_RECORDS, "[]") ?: "[]"
        val records = mutableListOf<HistoryRecord>()

        try {
            val array = JSONArray(json)

            for (i in 0 until array.length()) {
                try {
                    val obj = array.getJSONObject(i)
                    val scores = mutableMapOf<String, Float>()
                    val scoresObj = obj.optJSONObject("gradeScores") ?: JSONObject()
                    val keys = scoresObj.keys()

                    while (keys.hasNext()) {
                        val key = keys.next()
                        scores[key] = scoresObj.optDouble(key, 0.0).toFloat()
                    }

                    val boxes = mutableListOf<WoundBox>()
                    val boxesArray = obj.optJSONArray("detectedBoxes") ?: JSONArray()

                    for (j in 0 until boxesArray.length()) {
                        val box = boxesArray.getJSONObject(j)

                        boxes.add(WoundBox(
                            left = box.optDouble("left").toFloat(),
                            top = box.optDouble("top").toFloat(),
                            right = box.optDouble("right").toFloat(),
                            bottom = box.optDouble("bottom").toFloat(),
                            score = box.optDouble("score").toFloat()
                        ))
                    }

                    records.add(HistoryRecord(
                        id = obj.getString("id"),
                        date = obj.optString("date", ""),
                        source = obj.optString("source", "未知"),
                        woundGrade = obj.optString("woundGrade", "Unknown"),
                        confidence = obj.optDouble("confidence").toFloat(),
                        hasWound = obj.optBoolean("hasWound", false),
                        imagePath = obj.optString("imagePath", ""),
                        gradeScores = scores,
                        detectedBoxes = boxes,
                        latencyMs = obj.optLong("latencyMs", 0L)
                    ))
                } catch (e: Exception) {
                    Log.e("WoundHistory", "單筆紀錄解析失敗", e)
                }
            }
        } catch (e: Exception) {
            Log.e("WoundHistory", "歷史紀錄讀取失敗", e)
        }

        return records
    }

    private fun saveAll(context: Context, records: List<HistoryRecord>): Boolean {
        return context.applicationContext.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_RECORDS, recordsToJson(records))
            .commit()
    }

    // ── 新增檢測紀錄 ─────────────────────────────────
    @Synchronized
    fun add(
        context: Context,
        bitmap: Bitmap,
        result: DetectionResult,
        source: String
    ): HistoryRecord {

        val id = UUID.randomUUID().toString()
        val date = SimpleDateFormat("yyyy/MM/dd HH:mm:ss", Locale.TAIWAN).format(Date())

        val folder = imageDir(context)
        if (!folder.exists() && !folder.mkdirs()) {
            throw IllegalStateException("無法建立歷史圖片資料夾")
        }

        val imageFile = File(folder, "$id.jpg")

        try {
            FileOutputStream(imageFile).use { stream ->
                if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 92, stream)) {
                    throw IllegalStateException("圖片儲存失敗")
                }
            }

            val record = HistoryRecord(
                id = id,
                date = date,
                source = source,
                woundGrade = result.woundGrade,
                confidence = result.confidence,
                hasWound = result.hasWound,
                imagePath = imageFile.absolutePath,
                gradeScores = result.gradeScores.toMap(),
                detectedBoxes = result.detectedBoxes.toList(),
                latencyMs = result.latencyMs
            )

            val records = listOf(record) + getAll(context)

            if (!saveAll(context, records)) {
                throw IllegalStateException("歷史紀錄資料寫入失敗")
            }

            Log.d("WoundHistory", "歷史紀錄新增成功：${record.woundGrade}")
            return record

        } catch (e: Exception) {
            imageFile.delete()
            throw e
        }
    }

    // ── 刪除指定紀錄 ─────────────────────────────────
    @Synchronized
    fun delete(context: Context, record: HistoryRecord): Boolean {
        val records = getAll(context).filterNot { it.id == record.id }

        if (!saveAll(context, records)) return false

        try {
            val file = File(record.imagePath).canonicalFile
            if (file.parentFile == imageDir(context).canonicalFile) {
                if (file.exists() && !file.delete()) {
                    Log.w("WoundHistory", "紀錄已移除，但圖片刪除失敗：${file.path}")
                }
            }
        } catch (e: Exception) {
            Log.e("WoundHistory", "歷史圖片刪除失敗", e)
        }

        return true
    }
}

// ── 圖片讀取，避免一次載入過大圖片 ────────────────────
private fun loadHistoryImage(path: String, maxSize: Int): Bitmap? {
    if (path.isBlank()) return null

    return try {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)

        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        var sampleSize = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sampleSize > maxSize) {
            sampleSize *= 2
        }

        BitmapFactory.decodeFile(path, BitmapFactory.Options().apply {
            inSampleSize = sampleSize
        })
    } catch (e: Exception) {
        Log.e("WoundHistory", "圖片讀取失敗：$path", e)
        null
    }
}

private fun historyPercent(value: Float): String {
    return String.format(Locale.US, "%.1f%%", value * 100f)
}

// ── 歷史紀錄主畫面 ────────────────────────────────────
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HistoryScreen(navController: NavHostController) {

    val context = LocalContext.current
    var records by remember { mutableStateOf<List<HistoryRecord>>(emptyList()) }
    var selectedRecord by remember { mutableStateOf<HistoryRecord?>(null) }
    var deleteTarget by remember { mutableStateOf<HistoryRecord?>(null) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        records = withContext(Dispatchers.IO) {
            WoundHistoryStore.getAll(context)
        }
    }

    BackHandler(enabled = selectedRecord != null) {
        selectedRecord = null
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        if (selectedRecord == null) "歷史紀錄" else "檢測詳細報告",
                        fontWeight = FontWeight.Bold
                    )
                },
                navigationIcon = {
                    IconButton(onClick = {
                        if (selectedRecord != null) {
                            selectedRecord = null
                        } else {
                            navController.popBackStack()
                        }
                    }) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "返回")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = HistoryRed,
                    titleContentColor = Color.White,
                    navigationIconContentColor = Color.White
                )
            )
        }
    ) { paddingValues ->

        if (selectedRecord == null) {

            if (records.isEmpty()) {
                Box(
                    modifier = Modifier.fillMaxSize().padding(paddingValues),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("目前沒有歷史紀錄", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(8.dp))
                        Text("完成傷口辨識後，紀錄會顯示在這裡", color = Color.Gray)
                    }
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize().padding(paddingValues).padding(horizontal = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                    contentPadding = PaddingValues(vertical = 16.dp)
                ) {
                    item {
                        Text(
                            "共 ${records.size} 筆紀錄",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    items(records, key = { it.id }) { record ->
                        HistoryCard(
                            record = record,
                            onClick = { selectedRecord = record },
                            onDelete = { deleteTarget = record }
                        )
                    }
                }
            }

        } else {
            HistoryDetail(
                record = selectedRecord!!,
                modifier = Modifier.padding(paddingValues),
                onDelete = { deleteTarget = selectedRecord }
            )
        }
    }

    // ── 刪除確認視窗 ──────────────────────────────────
    deleteTarget?.let { target ->
        AlertDialog(
            onDismissRequest = { deleteTarget = null },
            title = { Text("刪除歷史紀錄") },
            text = { Text("確定刪除 ${target.date} 的檢測紀錄嗎？刪除後無法復原。") },
            confirmButton = {
                TextButton(onClick = {
                    val success = WoundHistoryStore.delete(context, target)

                    if (success) {
                        records = WoundHistoryStore.getAll(context)
                        if (selectedRecord?.id == target.id) selectedRecord = null
                    } else {
                        errorMessage = "刪除失敗，請稍後再試"
                    }

                    deleteTarget = null
                }) {
                    Text("刪除", color = HistoryRed)
                }
            },
            dismissButton = {
                TextButton(onClick = { deleteTarget = null }) {
                    Text("取消")
                }
            }
        )
    }

    errorMessage?.let { message ->
        AlertDialog(
            onDismissRequest = { errorMessage = null },
            title = { Text("錯誤") },
            text = { Text(message) },
            confirmButton = {
                TextButton(onClick = { errorMessage = null }) {
                    Text("確定")
                }
            }
        )
    }
}

// ── 歷史紀錄卡片 ──────────────────────────────────────
@Composable
private fun HistoryCard(
    record: HistoryRecord,
    onClick: () -> Unit,
    onDelete: () -> Unit
) {
    val preview by produceState<Bitmap?>(initialValue = null, key1 = record.imagePath) {
        value = withContext(Dispatchers.IO) {
            loadHistoryImage(record.imagePath, 200)
        }
    }

    Card(
        modifier = Modifier.fillMaxWidth().clickable(onClick = onClick),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (preview != null) {
                Image(
                    bitmap = preview!!.asImageBitmap(),
                    contentDescription = "傷口照片",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.size(82.dp).clip(RoundedCornerShape(10.dp))
                )
            } else {
                Box(
                    modifier = Modifier.size(82.dp).background(Color(0xFFE0E0E0), RoundedCornerShape(10.dp)),
                    contentAlignment = Alignment.Center
                ) {
                    Text("無圖片", color = Color.Gray)
                }
            }

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    record.woundGrade,
                    fontWeight = FontWeight.Bold,
                    style = MaterialTheme.typography.titleMedium,
                    color = if (record.hasWound) HistoryRed else Color(0xFF388E3C)
                )

                Spacer(modifier = Modifier.height(4.dp))
                Text("信心度：${historyPercent(record.confidence)}", style = MaterialTheme.typography.bodySmall)
                Text("來源：${record.source}", style = MaterialTheme.typography.bodySmall)
                Text(record.date, style = MaterialTheme.typography.bodySmall, color = Color.Gray)
            }

            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                IconButton(onClick = onDelete) {
                    Icon(Icons.Default.Delete, contentDescription = "刪除紀錄", tint = HistoryRed)
                }

                Icon(
                    Icons.Default.ChevronRight,
                    contentDescription = "查看詳細報告",
                    tint = Color.Gray
                )
            }
        }
    }
}

// ── 詳細檢測報告 ──────────────────────────────────────
@Composable
private fun HistoryDetail(
    record: HistoryRecord,
    modifier: Modifier = Modifier,
    onDelete: () -> Unit
) {
    val image by produceState<Bitmap?>(initialValue = null, key1 = record.imagePath) {
        value = withContext(Dispatchers.IO) {
            loadHistoryImage(record.imagePath, 1200)
        }
    }

    Column(
        modifier = modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {

        // ── 原始檢測照片 ──────────────────────────────
        Card(shape = RoundedCornerShape(16.dp)) {
            if (image != null) {
                Image(
                    bitmap = image!!.asImageBitmap(),
                    contentDescription = "檢測照片",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxWidth().height(280.dp).background(Color.Black)
                )
            } else {
                Box(
                    modifier = Modifier.fillMaxWidth().height(200.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text("照片無法讀取")
                }
            }
        }

        // ── 基本檢測資訊 ──────────────────────────────
        Card(shape = RoundedCornerShape(16.dp)) {
            Column(
                modifier = Modifier.fillMaxWidth().padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text("檢測結果", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                HorizontalDivider()

                HistoryReportRow("傷口判定", if (record.hasWound) "有偵測到傷口" else "未偵測到傷口")
                HistoryReportRow("WCC 等級", record.woundGrade)
                HistoryReportRow("分類信心度", historyPercent(record.confidence))
                HistoryReportRow("檢測時間", record.date)
                HistoryReportRow("圖片來源", record.source)
                HistoryReportRow("YOLO 偵測框", "${record.detectedBoxes.size} 個")
                HistoryReportRow("總推論耗時", if (record.latencyMs > 0) "${record.latencyMs} ms" else "未記錄")
            }
        }

        // ── MobileNet 分類詳細機率 ────────────────────
        Card(shape = RoundedCornerShape(16.dp)) {
            Column(
                modifier = Modifier.fillMaxWidth().padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text("MobileNet 分類機率", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                HorizontalDivider()

                if (record.gradeScores.isEmpty()) {
                    Text("沒有分類機率資料", color = Color.Gray)
                } else {
                    val order = listOf("WCC_0", "WCC_1", "WCC_2", "WCC_3", "WCC_4", "other", "No Wound")
                    val classes = order.filter { it in record.gradeScores } +
                            record.gradeScores.keys.filter { it !in order }.sorted()

                    classes.forEach { className ->
                        val score = record.gradeScores[className] ?: 0f

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Text(className, modifier = Modifier.width(70.dp), style = MaterialTheme.typography.bodySmall)

                            LinearProgressIndicator(
                                progress = { score.coerceIn(0f, 1f) },
                                modifier = Modifier.weight(1f).height(8.dp),
                                color = if (className == record.woundGrade) HistoryRed else Color(0xFF90A4AE),
                                trackColor = MaterialTheme.colorScheme.surfaceVariant
                            )

                            Text(
                                historyPercent(score),
                                modifier = Modifier.width(52.dp),
                                style = MaterialTheme.typography.bodySmall,
                                textAlign = TextAlign.End
                            )
                        }
                    }
                }
            }
        }

        // ── YOLO 偵測框詳細資訊 ───────────────────────
        Card(shape = RoundedCornerShape(16.dp)) {
            Column(
                modifier = Modifier.fillMaxWidth().padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text("YOLO 偵測詳細資訊", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                HorizontalDivider()

                if (record.detectedBoxes.isEmpty()) {
                    Text("本次沒有偵測到傷口框")
                } else {
                    record.detectedBoxes.forEachIndexed { index, box ->
                        Text("偵測框 ${index + 1}", fontWeight = FontWeight.Bold)
                        HistoryReportRow("信心度", historyPercent(box.score))
                        HistoryReportRow("左上座標", "(${box.left.toInt()}, ${box.top.toInt()})")
                        HistoryReportRow("右下座標", "(${box.right.toInt()}, ${box.bottom.toInt()})")
                        HistoryReportRow("寬 × 高", "${(box.right - box.left).toInt()} × ${(box.bottom - box.top).toInt()} px")

                        if (index < record.detectedBoxes.lastIndex) {
                            HorizontalDivider()
                        }
                    }
                }
            }
        }

        OutlinedButton(
            onClick = onDelete,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = HistoryRed)
        ) {
            Icon(Icons.Default.Delete, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text("刪除此筆紀錄")
        }
    }
}

// ── 詳細報告共用欄位 ─────────────────────────────────
@Composable
private fun HistoryReportRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.Top
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)

        Spacer(modifier = Modifier.width(12.dp))

        Text(
            value,
            modifier = Modifier.weight(1f, fill = false),
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium,
            textAlign = TextAlign.End
        )
    }
}
