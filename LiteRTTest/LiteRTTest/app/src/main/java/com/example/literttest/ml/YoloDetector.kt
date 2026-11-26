
package com.example.literttest.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class YoloDetector(context: Context) {

    private val assets = context.assets
    private var model: CompiledModel? = null

    companion object {
        private const val TAG = "YoloDetector"
        private const val MODEL_NAME = "yolo_no_nms_best_float32.tflite"
        private const val INPUT_SIZE = 640

        // 優先降低漏偵測，後續依測試結果調整
        private const val CONF_THRESHOLD = 0.15f
        private const val IOU_THRESHOLD = 0.45f

        // 第二次偵測使用中央 65% 區域
        private const val ENABLE_SECOND_PASS = true
        private const val CENTER_CROP_RATIO = 0.65f

        // 目前按照單類 YOLO 的 [1, 5, N] 輸出解析
        // 若實際模型輸出是 [1, N, 5]，需改成 false
        private const val CHANNEL_FIRST_OUTPUT = true
        private const val ATTRS = 5
    }

    init {
        model = try {
            CompiledModel.create(
                assets,
                MODEL_NAME,
                CompiledModel.Options(Accelerator.GPU)
            ).also {
                Log.d(TAG, "YOLO GPU 初始化成功")
            }
        } catch (e: Exception) {
            Log.w(TAG, "YOLO GPU 初始化失敗，改用 CPU", e)
            CompiledModel.create(
                assets,
                MODEL_NAME,
                CompiledModel.Options(Accelerator.CPU)
            ).also {
                Log.d(TAG, "YOLO CPU 初始化成功")
            }
        }
    }

    // ── 主流程：完整圖片 → 必要時中央放大 ──────────────
    fun detect(bitmap: Bitmap): List<WoundBox> {
        require(!bitmap.isRecycled) { "YOLO 輸入圖片已被釋放" }

        val firstPass = detectOnce(bitmap, "完整照片")

        if (firstPass.isNotEmpty()) {
            Log.d(TAG, "完整照片偵測成功：${firstPass.size} 個傷口框")
            return firstPass
        }

        if (!ENABLE_SECOND_PASS) {
            return emptyList()
        }

        Log.d(TAG, "完整照片未偵測到傷口，開始中央放大偵測")

        val cropWidth = (bitmap.width * CENTER_CROP_RATIO)
            .roundToInt().coerceIn(1, bitmap.width)

        val cropHeight = (bitmap.height * CENTER_CROP_RATIO)
            .roundToInt().coerceIn(1, bitmap.height)

        val cropLeft = (bitmap.width - cropWidth) / 2
        val cropTop = (bitmap.height - cropHeight) / 2

        val cropped = Bitmap.createBitmap(
            bitmap,
            cropLeft,
            cropTop,
            cropWidth,
            cropHeight
        )

        val secondPass = try {
            detectOnce(cropped, "中央放大")
        } finally {
            if (cropped !== bitmap) cropped.recycle()
        }

        if (secondPass.isEmpty()) {
            Log.w(TAG, "兩次 YOLO 偵測都未找到符合門檻的傷口")
            return emptyList()
        }

        // 將中央放大後的框座標還原到原始照片
        val restored = secondPass.map { box ->
            WoundBox(
                left = (box.left + cropLeft).coerceIn(0f, bitmap.width.toFloat()),
                top = (box.top + cropTop).coerceIn(0f, bitmap.height.toFloat()),
                right = (box.right + cropLeft).coerceIn(0f, bitmap.width.toFloat()),
                bottom = (box.bottom + cropTop).coerceIn(0f, bitmap.height.toFloat()),
                score = box.score
            )
        }

        Log.d(TAG, "中央放大偵測成功：${restored.size} 個傷口框")
        return restored
    }

    // ── 單次 YOLO 推論 ─────────────────────────────────
    private fun detectOnce(
        bitmap: Bitmap,
        passName: String
    ): List<WoundBox> {

        val m = model ?: throw IllegalStateException("YOLO 模型尚未初始化")
        val letterbox = letterboxBitmap(bitmap, INPUT_SIZE)

        try {
            val inputArray = bitmapToNhwcFloatArray(
                letterbox.bitmap,
                INPUT_SIZE
            )

            val inputBuffers = m.createInputBuffers()

            try {
                val outputBuffers = m.createOutputBuffers()

                try {
                    inputBuffers[0].writeFloat(inputArray)

                    val startNs = SystemClock.elapsedRealtimeNanos()

                    m.run(inputBuffers, outputBuffers)

                    val elapsedMs =
                        (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000.0

                    Log.d(TAG, "$passName YOLO 耗時: $elapsedMs ms")

                    val output = outputBuffers[0].readFloat()
                    Log.d(TAG, "$passName 原始輸出最小值: ${output.minOrNull()}")
                    Log.d(TAG, "$passName 原始輸出最大值: ${output.maxOrNull()}")
                    Log.d(TAG, "$passName 原始輸出前20個數值: ${output.take(20)}")
                    return parseYoloOutput(
                        output = output,
                        sourceWidth = bitmap.width,
                        sourceHeight = bitmap.height,
                        letterbox = letterbox,
                        passName = passName
                    )

                } finally {
                    outputBuffers.forEach { it.close() }
                }

            } finally {
                inputBuffers.forEach { it.close() }
            }

        } finally {
            letterbox.bitmap.recycle()
        }
    }

    // ── 解析 YOLO 原始輸出 ────────────────────────────
    private fun parseYoloOutput(
        output: FloatArray,
        sourceWidth: Int,
        sourceHeight: Int,
        letterbox: LetterboxResult,
        passName: String
    ): List<WoundBox> {

        require(output.isNotEmpty() && output.size % ATTRS == 0) {
            "YOLO 輸出長度不符合單類五欄格式：${output.size}"
        }

        val numBoxes = output.size / ATTRS

        fun getValue(boxIndex: Int, attrIndex: Int): Float {
            return if (CHANNEL_FIRST_OUTPUT) {
                output[attrIndex * numBoxes + boxIndex]
            } else {
                output[boxIndex * ATTRS + attrIndex]
            }
        }

        // 一次檢查整個模型的座標尺度，避免每個框使用不同尺度
        var maxCoordinate = 0f
        var maxScore = 0f
        var validScores = 0
        var scoresAboveThreshold = 0

        for (i in 0 until numBoxes) {
            val score = getValue(i, 4)

            if (score.isFinite() && score in 0f..1f) {
                validScores++
                if (score > maxScore) maxScore = score
                if (score >= CONF_THRESHOLD) scoresAboveThreshold++
            }

            for (j in 0..3) {
                val value = getValue(i, j)
                if (value.isFinite()) {
                    maxCoordinate = max(maxCoordinate, abs(value))
                }
            }
        }

        if (validScores < numBoxes / 2) {
            throw IllegalStateException(
                "YOLO 信心度輸出異常，請確認輸出 Shape 與 CHANNEL_FIRST_OUTPUT 設定"
            )
        }

        val normalizedCoordinates = maxCoordinate <= 2f

        Log.d(TAG, "========== $passName ==========")
        Log.d(TAG, "輸出元素數量: ${output.size}")
        Log.d(TAG, "預測框數: $numBoxes")
        Log.d(TAG, "排列: ${if (CHANNEL_FIRST_OUTPUT) "[5,N]" else "[N,5]"}")
        Log.d(TAG, "座標: ${if (normalizedCoordinates) "Normalized" else "Pixel"}")
        Log.d(TAG, "最高信心度: $maxScore")
        Log.d(TAG, "門檻: $CONF_THRESHOLD")
        Log.d(TAG, "達門檻預測數: $scoresAboveThreshold")

        val candidates = mutableListOf<WoundBox>()

        for (i in 0 until numBoxes) {
            var cx = getValue(i, 0)
            var cy = getValue(i, 1)
            var w = getValue(i, 2)
            var h = getValue(i, 3)
            val score = getValue(i, 4)

            if (!score.isFinite() || score < CONF_THRESHOLD || score > 1f) continue

            if (!cx.isFinite() || !cy.isFinite() ||
                !w.isFinite() || !h.isFinite()
            ) continue

            if (w <= 0f || h <= 0f) continue

            if (normalizedCoordinates) {
                cx *= INPUT_SIZE
                cy *= INPUT_SIZE
                w *= INPUT_SIZE
                h *= INPUT_SIZE
            }

            val x1 = cx - w / 2f
            val y1 = cy - h / 2f
            val x2 = cx + w / 2f
            val y2 = cy + h / 2f

            // Letterbox 座標 → 原圖座標
            val left = (x1 - letterbox.padX) / letterbox.scaleX
            val top = (y1 - letterbox.padY) / letterbox.scaleY
            val right = (x2 - letterbox.padX) / letterbox.scaleX
            val bottom = (y2 - letterbox.padY) / letterbox.scaleY

            val box = WoundBox(
                left = left.coerceIn(0f, sourceWidth.toFloat()),
                top = top.coerceIn(0f, sourceHeight.toFloat()),
                right = right.coerceIn(0f, sourceWidth.toFloat()),
                bottom = bottom.coerceIn(0f, sourceHeight.toFloat()),
                score = score
            )

            if (box.right > box.left && box.bottom > box.top) {
                candidates.add(box)
            }
        }

        val result = nms(
            candidates.sortedByDescending { it.score },
            IOU_THRESHOLD
        )

        Log.d(TAG, "有效候選框: ${candidates.size}")
        Log.d(TAG, "NMS 後框數: ${result.size}")

        result.take(5).forEachIndexed { index, box ->
            Log.d(
                TAG,
                "框${index + 1}: score=${box.score}, " +
                        "(${box.left},${box.top})-(${box.right},${box.bottom})"
            )
        }

        Log.d(TAG, "================================")

        return result
    }

    // ── NMS ───────────────────────────────────────────
    private fun nms(
        boxes: List<WoundBox>,
        iouThreshold: Float
    ): List<WoundBox> {

        val selected = mutableListOf<WoundBox>()

        for (box in boxes) {
            var keep = true

            for (chosen in selected) {
                if (iou(box, chosen) > iouThreshold) {
                    keep = false
                    break
                }
            }

            if (keep) selected.add(box)
        }

        return selected
    }

    private fun iou(a: WoundBox, b: WoundBox): Float {

        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)

        val width = max(0f, right - left)
        val height = max(0f, bottom - top)

        val intersection = width * height

        val areaA =
            max(0f, a.right - a.left) *
                    max(0f, a.bottom - a.top)

        val areaB =
            max(0f, b.right - b.left) *
                    max(0f, b.bottom - b.top)

        val union = areaA + areaB - intersection

        return if (union <= 0f) 0f else intersection / union
    }

    // ── YOLO Letterbox ────────────────────────────────
    private fun letterboxBitmap(
        src: Bitmap,
        inputSize: Int
    ): LetterboxResult {

        val scale = min(
            inputSize / src.width.toFloat(),
            inputSize / src.height.toFloat()
        )

        val newWidth = (src.width * scale)
            .roundToInt().coerceIn(1, inputSize)

        val newHeight = (src.height * scale)
            .roundToInt().coerceIn(1, inputSize)

        val resized = Bitmap.createScaledBitmap(
            src,
            newWidth,
            newHeight,
            true
        )

        val output = Bitmap.createBitmap(
            inputSize,
            inputSize,
            Bitmap.Config.ARGB_8888
        )

        val padX = (inputSize - newWidth) / 2
        val padY = (inputSize - newHeight) / 2

        val canvas = Canvas(output)
        canvas.drawColor(Color.rgb(114, 114, 114))

        val paint = Paint(Paint.FILTER_BITMAP_FLAG)

        canvas.drawBitmap(
            resized,
            padX.toFloat(),
            padY.toFloat(),
            paint
        )

        if (resized !== src) {
            resized.recycle()
        }

        return LetterboxResult(
            bitmap = output,
            scaleX = newWidth.toFloat() / src.width,
            scaleY = newHeight.toFloat() / src.height,
            padX = padX.toFloat(),
            padY = padY.toFloat()
        )
    }

    // ── Bitmap → NHWC Float32 ─────────────────────────
    private fun bitmapToNhwcFloatArray(
        bitmap: Bitmap,
        inputSize: Int
    ): FloatArray {

        val pixels = IntArray(inputSize * inputSize)

        bitmap.getPixels(
            pixels, 0, inputSize, 0, 0, inputSize, inputSize
        )

        val output = FloatArray(inputSize * inputSize * 3)

        for (i in pixels.indices) {
            val pixel = pixels[i]

            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f

            val index = i * 3

            output[index] = r
            output[index + 1] = g
            output[index + 2] = b
        }

        return output
    }

    private data class LetterboxResult(
        val bitmap: Bitmap,
        val scaleX: Float,
        val scaleY: Float,
        val padX: Float,
        val padY: Float
    )

    fun close() {
        model?.close()
        model = null
        Log.d(TAG, "YOLO 模型已關閉")
    }
}
