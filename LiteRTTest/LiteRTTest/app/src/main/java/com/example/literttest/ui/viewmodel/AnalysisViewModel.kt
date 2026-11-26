package com.example.literttest.ui.viewmodel

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import com.example.literttest.ml.InferenceEngine
import android.content.res.AssetManager

data class AnalysisResult(
    val prediction: String = "",
    val confidence: Float = 0f,
    val latencyMs: Double = 0.0,
    val cameraFps: Double = 0.0,
    val inferenceFps: Double = 0.0
)

class AnalysisViewModel(private val assetManager: AssetManager) : ViewModel() {
    private val _analysisResult = MutableStateFlow(AnalysisResult())
    val analysisResult: StateFlow<AnalysisResult> = _analysisResult

    private val inferenceEngine = InferenceEngine(assetManager)

    fun updateResult(result: AnalysisResult) {
        _analysisResult.value = result
    }

    fun switchModel(taskMode: String, accelerator: String) {
        // 切換模型邏輯
    }

    override fun onCleared() {
        super.onCleared()
        inferenceEngine.close()
    }
}
