package com.example.literttest.ml

data class DetectionResult(
    val woundGrade: String,              // "WCC_0" / "WCC_1" / "WCC_2" / "WCC_3" / "other"
    val confidence: Float,               // 0.0f ~ 1.0f
    val hasWound: Boolean,               // true / false
    val gradeScores: Map<String, Float>, // 每個等級的分數
    val latencyMs: Long,                 // 推論時間（毫秒）
    val detectedBoxes: List<WoundBox> = emptyList()  // 偵測到的傷口框
)
