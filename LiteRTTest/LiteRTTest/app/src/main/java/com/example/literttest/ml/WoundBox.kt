package com.example.literttest.ml

data class WoundBox(
    val left: Float,      // 左邊界
    val top: Float,       // 上邊界
    val right: Float,     // 右邊界
    val bottom: Float,    // 下邊界
    val score: Float      // 信心度 (0.0 ~ 1.0)
) {
    // 寬度
    val width: Float
        get() = right - left

    // 高度
    val height: Float
        get() = bottom - top

    // 中心 X
    val centerX: Float
        get() = (left + right) / 2f

    // 中心 Y
    val centerY: Float
        get() = (top + bottom) / 2f

    // 面積
    val area: Float
        get() = width * height
}
