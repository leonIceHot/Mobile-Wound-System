package com.example.literttest.ml

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlin.math.*

class InferenceEngine(private val assetManager: AssetManager) {
    private var clsModel: CompiledModel? = null
    private var detModel: CompiledModel? = null
    private val modelLock = Any()

    companion object {
        private const val TAG = "InferenceEngine"
        private const val CLS_MODEL_NAME = "mobilenetv3_large_93acc.tflite"
        private const val DET_MODEL_NAME = "yolo_no_nms_best_float32.tflite"
        private const val CLS_INPUT_SIZE = 224
        private const val DET_INPUT_SIZE = 640
    }

    fun loadModel(taskMode: String, accelerator: Accelerator) {
        synchronized(modelLock) {
            clsModel?.close()
            detModel?.close()
            clsModel = null
            detModel = null

            val modelName = if (taskMode == "CLASSIFICATION") CLS_MODEL_NAME else DET_MODEL_NAME

            try {
                val model = CompiledModel.create(
                    assetManager,
                    modelName,
                    CompiledModel.Options(accelerator)
                )

                if (taskMode == "CLASSIFICATION") {
                    clsModel = model
                } else {
                    detModel = model
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                throw e
            }
        }
    }

    fun runClassification(bitmap: Bitmap): Pair<String, Float> {
        // 你的分類邏輯
        return Pair("WCC_0", 0.95f)
    }

    fun runDetection(bitmap: Bitmap): List<DetectionBox> {
        // 你的偵測邏輯
        return emptyList()
    }

    fun close() {
        synchronized(modelLock) {
            clsModel?.close()
            detModel?.close()
            clsModel = null
            detModel = null
        }
    }
}

data class DetectionBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val label: String,
    val score: Float
)
