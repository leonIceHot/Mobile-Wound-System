import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

data class DetectionBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val label: String,
    val score: Float
)

class BoxOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val boxes = mutableListOf<DetectionBox>()

    private val boxPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.YELLOW
        textSize = 42f
        isAntiAlias = true
    }

    private val textBgPaint = Paint().apply {
        color = Color.argb(180, 0, 0, 0)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    fun setBoxes(newBoxes: List<DetectionBox>) {
        boxes.clear()
        boxes.addAll(newBoxes)
        invalidate()
    }

    fun clearBoxes() {
        boxes.clear()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (box in boxes) {
            val left = box.left
            val top = box.top
            val right = box.right
            val bottom = box.bottom

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val text = "${box.label} ${"%.1f".format(box.score * 100f)}%"
            val textX = left
            val textY = if (top - 12f > 42f) top - 12f else top + 48f

            val textWidth = textPaint.measureText(text)
            val bgLeft = textX
            val bgTop = textY - 42f
            val bgRight = textX + textWidth + 18f
            val bgBottom = textY + 10f

            canvas.drawRect(bgLeft, bgTop, bgRight, bgBottom, textBgPaint)
            canvas.drawText(text, textX + 8f, textY, textPaint)
        }
    }
}