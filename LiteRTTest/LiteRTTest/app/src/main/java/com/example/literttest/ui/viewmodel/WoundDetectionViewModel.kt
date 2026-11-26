
package com.example.literttest.ui.viewmodel

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.literttest.ml.DetectionResult
import com.example.literttest.ml.WoundBox
import com.example.literttest.ml.WoundClassifier
import com.example.literttest.ml.YoloDetector
import com.example.literttest.ui.screens.WoundHistoryStore
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class WoundDetectionViewModel : ViewModel() {

    companion object {
        private const val TAG = "WoundDetectionVM"
    }

    // ── State ─────────────────────────────────────────────
    var capturedBitmap by mutableStateOf<Bitmap?>(null)
        private set

    var detectionResult by mutableStateOf<DetectionResult?>(null)
        private set

    var isProcessing by mutableStateOf(false)
        private set

    var errorMessage by mutableStateOf<String?>(null)
        private set

    // ── 推論引擎 ──────────────────────────────────────────
    private var yoloDetector: YoloDetector? = null
    private var woundClassifier: WoundClassifier? = null
    private var appContext: Context? = null

    // ── 工作管理 ──────────────────────────────────────────
    private var initializationJob: Job? = null
    private var detectionJob: Job? = null
    private val inferenceMutex = Mutex()
    private var requestVersion = 0L

    // ── 初始化模型（避免重複初始化） ───────────────────────
    fun initializeModels(context: Context) {
        appContext = context.applicationContext

        if (yoloDetector != null && woundClassifier != null) {
            Log.d(TAG, "模型已初始化，跳過重複初始化")
            return
        }

        if (initializationJob?.isActive == true) {
            Log.d(TAG, "模型正在初始化，跳過重複請求")
            return
        }

        initializationJob = viewModelScope.launch {
            try {
                Log.d(TAG, "開始初始化模型...")

                withContext(Dispatchers.Default) {
                    val detector = YoloDetector(context.applicationContext)

                    try {
                        val classifier = WoundClassifier(context.applicationContext)
                        yoloDetector = detector
                        woundClassifier = classifier
                    } catch (e: Exception) {
                        detector.close()
                        throw e
                    }
                }

                Log.d(TAG, "YOLO 與 MobileNet 初始化完成")

            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "模型初始化失敗", e)
                errorMessage = "模型初始化失敗: ${e.message}"
            }
        }
    }

    // ── 主流程：檢測 → 分類 → 儲存歷史 ───────────────────
    // 預設來源為「上傳」，相機呼叫時指定 source = "相機"
    fun detectWoundFromBitmap(bitmap: Bitmap, source: String = "上傳") {

        detectionJob = viewModelScope.launch(Dispatchers.Main.immediate) {
            if (isProcessing) {
                Log.w(TAG, "已有推論正在執行，跳過重複請求")
                return@launch
            }

            val currentRequest = ++requestVersion

            isProcessing = true
            errorMessage = null
            detectionResult = null
            capturedBitmap = bitmap

            try {
                val context = appContext
                    ?: throw IllegalStateException("尚未取得 App Context，請先初始化模型")

                // 等待模型初始化完成
                if (yoloDetector == null || woundClassifier == null) {
                    initializeModels(context)
                }

                initializationJob?.join()

                if (yoloDetector == null || woundClassifier == null) {
                    throw IllegalStateException("YOLO 或 MobileNet 模型尚未成功初始化")
                }

                Log.d(TAG, "開始新一輪 AI 推論，來源=$source")

                // 同時間只允許一個推論操作模型
                val newResult = inferenceMutex.withLock {

                    val startNs = SystemClock.elapsedRealtimeNanos()

                    // ① YOLO 偵測
                    val detectedBoxes = runDetection(bitmap)
                    Log.d(TAG, "YOLO 檢測完成，找到 ${detectedBoxes.size} 個傷口框")

                    if (detectedBoxes.isEmpty()) {

                        // 沒有傷口，仍建立檢測結果
                        val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000

                        DetectionResult(
                            woundGrade = "No Wound",
                            confidence = 0f,
                            hasWound = false,
                            gradeScores = mapOf("No Wound" to 1.0f),
                            latencyMs = elapsedMs,
                            detectedBoxes = emptyList()
                        )

                    } else {

                        // ② MobileNet 分類
                        val (grade, scores) = runClassification(bitmap)

                        if (grade == "Error" || scores.isEmpty()) {
                            throw IllegalStateException("MobileNet 分類失敗，沒有取得有效結果")
                        }

                        Log.d(TAG, "MobileNet 分類完成：$grade")

                        // ③ 組合結果
                        val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000

                        DetectionResult(
                            woundGrade = grade,
                            confidence = scores[grade] ?: 0f,
                            hasWound = true,
                            gradeScores = scores,
                            latencyMs = elapsedMs,
                            detectedBoxes = detectedBoxes
                        )
                    }
                }

                // ④ 正式儲存歷史紀錄
                // 寫入完成後才顯示結果，避免導航太快導致沒有紀錄
                withContext(Dispatchers.IO) {
                    WoundHistoryStore.add(
                        context = context,
                        bitmap = bitmap,
                        result = newResult,
                        source = source
                    )
                }

                Log.d(TAG, "歷史紀錄儲存成功：${newResult.woundGrade}")

                // ⑤ 只更新本次有效的推論結果
                if (currentRequest == requestVersion) {
                    detectionResult = newResult
                    Log.d(TAG, "本次推論完成：$newResult")
                }

            } catch (e: CancellationException) {
                Log.d(TAG, "推論工作已取消")
                throw e

            } catch (e: Exception) {
                Log.e(TAG, "推論或歷史紀錄儲存失敗", e)

                if (currentRequest == requestVersion) {
                    errorMessage = "處理失敗: ${e.message}"
                    detectionResult = null
                }

            } finally {
                if (currentRequest == requestVersion) {
                    isProcessing = false
                }
            }
        }
    }

    // ── YOLO 檢測 ────────────────────────────────────────
    private suspend fun runDetection(bitmap: Bitmap): List<WoundBox> {
        return withContext(Dispatchers.Default) {

            val detector = yoloDetector
                ?: throw IllegalStateException("YOLO 檢測器未初始化")

            val startTime = SystemClock.elapsedRealtimeNanos()

            val boxes = detector.detect(bitmap)

            val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startTime) / 1_000_000.0

            Log.d(TAG, "YOLO 推論耗時: $elapsedMs ms")

            boxes
        }
    }

    // ── MobileNet 分類 ───────────────────────────────────
    private suspend fun runClassification(bitmap: Bitmap): Pair<String, Map<String, Float>> {
        return withContext(Dispatchers.Default) {

            val classifier = woundClassifier
                ?: throw IllegalStateException("MobileNet 分類器未初始化")

            val startTime = SystemClock.elapsedRealtimeNanos()

            val (grade, scores) = classifier.classify(bitmap)

            val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startTime) / 1_000_000.0

            Log.d(TAG, "MobileNet 推論耗時: $elapsedMs ms")

            Pair(grade, scores)
        }
    }

    // ── 清除舊結果，準備下一次檢測 ──────────────────────
    fun resetResult() {
        Log.d(TAG, "重置檢測結果")

        // 讓前一次工作不能再更新畫面
        requestVersion++

        // 取消正在執行的舊工作
        detectionJob?.cancel()
        detectionJob = null

        detectionResult = null
        capturedBitmap = null
        errorMessage = null
        isProcessing = false

        // 不刪除歷史紀錄，也不重新初始化模型
    }

    // ── 錯誤重置 ─────────────────────────────────────────
    fun clearError() {
        errorMessage = null
    }

    // ── 清理模型資源 ─────────────────────────────────────
    override fun onCleared() {
        super.onCleared()

        detectionJob?.cancel()
        initializationJob?.cancel()

        yoloDetector?.close()
        woundClassifier?.close()

        yoloDetector = null
        woundClassifier = null
        appContext = null
        capturedBitmap = null

        Log.d(TAG, "ViewModel 已清理")
    }
}
