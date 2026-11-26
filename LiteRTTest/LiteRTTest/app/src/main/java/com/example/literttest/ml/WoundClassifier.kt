
package com.example.literttest.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlin.math.exp
import kotlin.math.roundToInt

class WoundClassifier(context: Context) {
    private val assets = context.assets
    private var model: CompiledModel? = null

    companion object {
        private const val TAG = "WoundClassifier"
        private const val MODEL_NAME = "mobilenetv3_large_6class_float32.tflite"
        private const val INPUT_SIZE = 224
        private const val RESIZE_SIZE = 256
        private const val NUM_CLASSES = 6
    }

    // 必須與模型訓練時的 class_to_idx 索引順序一致
    private val classNames = listOf(
        "OTHER",
        "WCC_0",
        "WCC_1",
        "WCC_2",
        "WCC_3",
        "WCC_4"
    )

    init {
        model = CompiledModel.create(
            assets,
            MODEL_NAME,
            CompiledModel.Options(Accelerator.CPU)
        )
        Log.d(TAG, "MobileNetV3 Large 使用 CPU 初始化成功")
    }

    // ── 主分類流程 ─────────────────────────────────────
    fun classify(bitmap: Bitmap): Pair<String, Map<String, Float>> {
        val m = model ?: return Pair("Error", emptyMap())

        var inputBuffers: List<*>? = null
        var outputBuffers: List<*>? = null

        return try {
            // ① 圖片預處理：Resize 256 → CenterCrop 224 → Normalize
            // ② 使用 NHWC 格式：[1, 224, 224, 3]
            val inputArray = bitmapToNhwcFloatArray(bitmap)

            inputBuffers = m.createInputBuffers()
            outputBuffers = m.createOutputBuffers()

            inputBuffers[0].writeFloat(inputArray)

            // ③ LiteRT 推論
            val startNs = SystemClock.elapsedRealtimeNanos()
            m.run(inputBuffers, outputBuffers)
            val endNs = SystemClock.elapsedRealtimeNanos()

            val elapsedMs = (endNs - startNs) / 1_000_000.0
            Log.d(TAG, "MobileNetV3 Large 分類耗時: $elapsedMs ms")

            // ④ 取得六個原始輸出
            val output = outputBuffers[0].readFloat()

            require(output.size == NUM_CLASSES) {
                "分類器輸出錯誤：預期 $NUM_CLASSES 類，實際 ${output.size} 類"
            }

            Log.d(TAG, "模型原始輸出: ${output.contentToString()}")

            // ⑤ Logits 轉換成機率
            val probs = softmax(output)

            // ⑥ 找出最高機率類別
            val bestIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            val grade = classNames[bestIdx]

            // ⑦ 儲存全部六類機率
            val scores = classNames.indices.associate { i ->
                classNames[i] to probs[i]
            }

            Log.d(TAG, "分類結果: $grade")
            Log.d(TAG, "分類信心度: ${probs[bestIdx] * 100f}%")

            scores.forEach { (name, score) ->
                Log.d(TAG, "$name: ${String.format("%.2f", score * 100f)}%")
            }

            Pair(grade, scores)

        } finally {
            inputBuffers?.forEach { if (it is AutoCloseable) it.close() }
            outputBuffers?.forEach { if (it is AutoCloseable) it.close() }
        }
    }

    // ── Bitmap → NHWC Float32 ───────────────────────────
    private fun bitmapToNhwcFloatArray(bitmap: Bitmap): FloatArray {

        // 配合 PyTorch Resize(256)：將較短的一邊調整到 256
        val scale = RESIZE_SIZE.toFloat() / minOf(bitmap.width, bitmap.height)
        val scaledWidth = (bitmap.width * scale).roundToInt()
        val scaledHeight = (bitmap.height * scale).roundToInt()

        val scaled = Bitmap.createScaledBitmap(
            bitmap,
            scaledWidth,
            scaledHeight,
            true
        )

        try {
            // CenterCrop(224)
            val cropX = (scaledWidth - INPUT_SIZE) / 2
            val cropY = (scaledHeight - INPUT_SIZE) / 2

            val cropped = Bitmap.createBitmap(
                scaled,
                cropX,
                cropY,
                INPUT_SIZE,
                INPUT_SIZE
            )

            try {
                val w = INPUT_SIZE
                val h = INPUT_SIZE

                val pixels = IntArray(w * h)
                cropped.getPixels(pixels, 0, w, 0, 0, w, h)

                // ImageNet Normalize
                val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
                val std = floatArrayOf(0.229f, 0.224f, 0.225f)

                // NHWC：RGB 交錯排列
                val out = FloatArray(w * h * 3)

                for (i in pixels.indices) {
                    val p = pixels[i]

                    val r = ((p shr 16) and 0xFF) / 255f
                    val g = ((p shr 8) and 0xFF) / 255f
                    val b = (p and 0xFF) / 255f

                    val offset = i * 3

                    out[offset] = (r - mean[0]) / std[0]
                    out[offset + 1] = (g - mean[1]) / std[1]
                    out[offset + 2] = (b - mean[2]) / std[2]
                }

                return out

            } finally {
                if (cropped !== bitmap && cropped !== scaled) {
                    cropped.recycle()
                }
            }

        } finally {
            if (scaled !== bitmap) {
                scaled.recycle()
            }
        }
    }

    // ── Softmax：Logits → 機率 ──────────────────────────
    private fun softmax(arr: FloatArray): FloatArray {
        val maxVal = arr.maxOrNull() ?: 0f
        val exps = FloatArray(arr.size)
        var sum = 0f

        for (i in arr.indices) {
            exps[i] = exp((arr[i] - maxVal).toDouble()).toFloat()
            sum += exps[i]
        }

        for (i in exps.indices) {
            exps[i] /= sum
        }

        return exps
    }

    // ── 清理模型 ───────────────────────────────────────
    fun close() {
        model?.close()
        model = null
        Log.d(TAG, "MobileNetV3 Large 模型已關閉")
    }
}
