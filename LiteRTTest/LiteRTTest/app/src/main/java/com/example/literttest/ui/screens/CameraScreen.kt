
package com.example.literttest.ui.screens

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.provider.MediaStore
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Camera
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Observer
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.example.literttest.ui.viewmodel.WoundDetectionViewModel
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

private const val CAMERA_TAG = "CameraScreen"
private val DarkRed = Color(0xFFB71C1C)
private val DarkOverlay = Color(0xAA000000)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CameraScreen(
    navController: NavController,
    viewModel: WoundDetectionViewModel = viewModel()
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var imageCaptureUseCase by remember { mutableStateOf<ImageCapture?>(null) }
    var isCapturing by remember { mutableStateOf(false) }
    var hasNewCapture by remember { mutableStateOf(false) }
    var cameraError by remember { mutableStateOf<String?>(null) }
    var cameraStatus by remember { mutableStateOf("正在啟動相機...") }
    var cameraInitKey by remember { mutableIntStateOf(0) }

    val isProcessing = viewModel.isProcessing
    val errorMessage = viewModel.errorMessage
    val result = viewModel.detectionResult

    // ── 相機權限 ──────────────────────────────────────────
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        if (!granted) {
            cameraError = "尚未取得相機權限，請允許 APP 使用相機"
        } else {
            cameraError = null
            cameraInitKey++
        }
    }

    LaunchedEffect(Unit) {
        viewModel.resetResult()
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    // ── PreviewView，只建立一次 ────────────────────────────
    val previewView = remember(context) {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    // ── 共用拍照執行緒 ──────────────────────────────────
    val cameraExecutor = remember {
        Executors.newSingleThreadExecutor()
    }

    DisposableEffect(cameraExecutor) {
        onDispose {
            cameraExecutor.shutdown()
        }
    }

    // ── CameraX 初始化與生命週期管理 ────────────────────
    DisposableEffect(
        hasCameraPermission,
        lifecycleOwner,
        cameraInitKey,
        previewView
    ) {
        var disposed = false
        var boundProvider: ProcessCameraProvider? = null
        var boundPreview: Preview? = null
        var boundCapture: ImageCapture? = null
        var streamObserver: Observer<PreviewView.StreamState>? = null
        var boundCamera: Camera? = null
        var stateObserver: Observer<CameraState>? = null

        imageCaptureUseCase = null

        if (hasCameraPermission) {
            cameraError = null
            cameraStatus = "正在連接後置相機..."

            try {
                val providerFuture = ProcessCameraProvider.getInstance(context)

                providerFuture.addListener({
                    if (!disposed) {
                        try {
                            val cameraProvider = providerFuture.get()

                            if (!cameraProvider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)) {
                                throw IllegalStateException("裝置找不到後置相機")
                            }

                            val preview = Preview.Builder()
                                .build()
                                .also {
                                    it.setSurfaceProvider(previewView.surfaceProvider)
                                }

                            val imageCapture = ImageCapture.Builder()
                                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                                .build()

                            // 移除上一次可能殘留的 CameraX UseCases
                            cameraProvider.unbindAll()

                            val camera = cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                CameraSelector.DEFAULT_BACK_CAMERA,
                                preview,
                                imageCapture
                            )

                            boundProvider = cameraProvider
                            boundPreview = preview
                            boundCapture = imageCapture
                            boundCamera = camera

                            // 追蹤相機開啟狀態，方便找出黑屏原因
                            val cameraObserver = Observer<CameraState> { state ->
                                if (!disposed) {
                                    Log.d(CAMERA_TAG, "CameraState: ${state.type}")

                                    state.error?.let { error ->
                                        cameraError = "相機開啟失敗，錯誤碼：${error.code}"
                                        Log.e(
                                            CAMERA_TAG,
                                            "CameraState error: ${error.code}",
                                            error.cause
                                        )
                                    }
                                }
                            }

                            camera.cameraInfo.cameraState.observe(
                                lifecycleOwner,
                                cameraObserver
                            )
                            stateObserver = cameraObserver

                            // 追蹤預覽是否真正開始輸出畫面
                            val observer = Observer<PreviewView.StreamState> { state ->
                                if (!disposed) {
                                    Log.d(CAMERA_TAG, "PreviewStreamState: $state")

                                    cameraStatus = when (state) {
                                        PreviewView.StreamState.STREAMING -> "預覽正常"
                                        else -> "正在等待相機畫面..."
                                    }
                                }
                            }

                            previewView.previewStreamState.observe(
                                lifecycleOwner,
                                observer
                            )
                            streamObserver = observer

                            imageCaptureUseCase = imageCapture

                            Log.d(CAMERA_TAG, "CameraX 綁定成功")

                        } catch (e: Exception) {
                            Log.e(CAMERA_TAG, "CameraX 初始化失敗", e)
                            cameraError = "相機初始化失敗：${e.message}"
                            cameraStatus = "初始化失敗"
                            imageCaptureUseCase = null
                        }
                    }
                }, ContextCompat.getMainExecutor(context))

            } catch (e: Exception) {
                Log.e(CAMERA_TAG, "無法取得 CameraProvider", e)
                cameraError = "無法啟動相機：${e.message}"
            }
        }

        onDispose {
            disposed = true
            imageCaptureUseCase = null

            streamObserver?.let {
                previewView.previewStreamState.removeObserver(it)
            }

            stateObserver?.let { observer ->
                boundCamera?.cameraInfo?.cameraState?.removeObserver(observer)
            }

            val provider = boundProvider
            val preview = boundPreview
            val capture = boundCapture

            if (provider != null && preview != null && capture != null) {
                try {
                    provider.unbind(preview, capture)
                    Log.d(CAMERA_TAG, "CameraX 資源已釋放")
                } catch (e: Exception) {
                    Log.e(CAMERA_TAG, "CameraX 釋放失敗", e)
                }
            }
        }
    }

    // ── 本次推論完成才跳轉 ──────────────────────────────
    LaunchedEffect(result, isProcessing, hasNewCapture) {
        if (hasNewCapture && result != null && !isProcessing) {
            hasNewCapture = false

            Log.d(CAMERA_TAG, "本次檢測完成，跳轉結果頁")

            navController.navigate("result") {
                popUpTo("camera") { saveState = true }
                launchSingleTop = true
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("拍攝傷口") },
                navigationIcon = {
                    IconButton(
                        onClick = {
                            if (!isCapturing && !isProcessing) {
                                navController.popBackStack()
                            }
                        }
                    ) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "返回"
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkRed,
                    titleContentColor = Color.White,
                    navigationIconContentColor = Color.White
                )
            )
        }
    ) { paddingValues ->

        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .background(Color.Black)
        ) {

            // ── 1. 相機即時預覽 ───────────────────────────
            if (hasCameraPermission) {
                AndroidView(
                    factory = { previewView },
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text(
                        text = "需要相機權限才能拍攝傷口",
                        color = Color.White
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    Button(
                        onClick = {
                            permissionLauncher.launch(Manifest.permission.CAMERA)
                        },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = DarkRed
                        )
                    ) {
                        Text("允許相機權限")
                    }
                }
            }

            // ── 2. 相機狀態提示 ───────────────────────────
            if (hasCameraPermission &&
                cameraStatus != "預覽正常" &&
                cameraError == null
            ) {
                Text(
                    text = cameraStatus,
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .padding(16.dp)
                        .background(
                            Color(0x99000000),
                            MaterialTheme.shapes.small
                        )
                        .padding(12.dp),
                    color = Color.White
                )
            }

            // ── 3. 相機錯誤與重試 ─────────────────────────
            cameraError?.let { error ->
                Column(
                    modifier = Modifier
                        .align(Alignment.Center)
                        .padding(20.dp)
                        .background(
                            Color(0xDD222222),
                            MaterialTheme.shapes.medium
                        )
                        .padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = error,
                        color = Color.White
                    )

                    if (hasCameraPermission) {
                        Spacer(modifier = Modifier.height(12.dp))

                        Button(
                            onClick = {
                                cameraError = null
                                cameraInitKey++
                            },
                            colors = ButtonDefaults.buttonColors(
                                containerColor = DarkRed
                            )
                        ) {
                            Text("重新啟動相機")
                        }
                    }
                }
            }

            // ── 4. 拍攝 / AI 處理中遮罩 ────────────────────
            if (isProcessing || isCapturing) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(DarkOverlay),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        CircularProgressIndicator(color = Color.White)

                        Spacer(modifier = Modifier.height(12.dp))

                        Text(
                            text = if (isCapturing) "拍攝中…" else "AI 分析中…",
                            color = Color.White,
                            style = MaterialTheme.typography.bodyLarge
                        )
                    }
                }
            }

            // ── 5. AI 推論錯誤提示 ────────────────────────
            errorMessage?.let { msg ->
                Snackbar(
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .padding(16.dp),
                    containerColor = DarkRed,
                    contentColor = Color.White
                ) {
                    Text(msg)
                }
            }

            // ── 6. 拍照按鈕 ───────────────────────────────
            Button(
                onClick = {
                    if (!isCapturing && !isProcessing) {
                        viewModel.resetResult()
                        hasNewCapture = false

                        takePhotoAndProcess(
                            context = context,
                            imageCapture = imageCaptureUseCase,
                            executor = cameraExecutor,
                            onCapturingStart = {
                                isCapturing = true
                            },
                            onCapturingEnd = {
                                isCapturing = false
                            },
                            onBitmapReady = { bitmap ->
                                hasNewCapture = true

                                viewModel.detectWoundFromBitmap(
                                    bitmap,
                                    source = "相機"
                                )
                            },
                            onError = { err ->
                                hasNewCapture = false
                                cameraError = err
                                Log.e(CAMERA_TAG, "拍照失敗：$err")
                            }
                        )
                    }
                },
                enabled = hasCameraPermission &&
                        imageCaptureUseCase != null &&
                        cameraError == null &&
                        !isCapturing &&
                        !isProcessing,
                shape = CircleShape,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 48.dp)
                    .size(72.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = DarkRed,
                    disabledContainerColor = Color.Gray
                ),
                contentPadding = PaddingValues(0.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Camera,
                    contentDescription = "拍照",
                    tint = Color.White,
                    modifier = Modifier.size(32.dp)
                )
            }
        }
    }
}

// ── 拍照 → Bitmap → 儲存相簿 ─────────────────────────
private fun takePhotoAndProcess(
    context: Context,
    imageCapture: ImageCapture?,
    executor: ExecutorService,
    onCapturingStart: () -> Unit,
    onCapturingEnd: () -> Unit,
    onBitmapReady: (Bitmap) -> Unit,
    onError: (String) -> Unit
) {
    val capture = imageCapture ?: run {
        onError("相機尚未初始化完成")
        return
    }

    val mainExecutor = ContextCompat.getMainExecutor(context)

    onCapturingStart()

    try {
        capture.takePicture(
            executor,
            object : ImageCapture.OnImageCapturedCallback() {

                override fun onCaptureSuccess(image: ImageProxy) {
                    try {
                        val buffer = image.planes[0].buffer
                        val bytes = ByteArray(buffer.remaining())
                        buffer.get(bytes)

                        val raw = BitmapFactory.decodeByteArray(
                            bytes, 0, bytes.size
                        ) ?: throw IllegalStateException("照片解碼失敗")

                        val rotated = try {
                            val degrees = image.imageInfo.rotationDegrees

                            if (degrees == 0) {
                                raw
                            } else {
                                val matrix = Matrix().apply {
                                    postRotate(degrees.toFloat())
                                }

                                Bitmap.createBitmap(
                                    raw,
                                    0,
                                    0,
                                    raw.width,
                                    raw.height,
                                    matrix,
                                    true
                                )
                            }
                        } finally {
                            // 若旋轉後產生新的 Bitmap，釋放原圖
                            // 不在這裡直接 recycle，避免旋轉為 0 時誤刪
                        }

                        if (rotated !== raw) {
                            raw.recycle()
                        }

                        // 相簿儲存放在背景執行緒
                        saveBitmapToGallery(context, rotated)

                        // Compose 狀態與 ViewModel 呼叫回到主執行緒
                        mainExecutor.execute {
                            onCapturingEnd()
                            onBitmapReady(rotated)
                        }

                    } catch (e: Exception) {
                        Log.e(CAMERA_TAG, "拍照影像處理失敗", e)

                        mainExecutor.execute {
                            onCapturingEnd()
                            onError("照片處理失敗：${e.message}")
                        }

                    } finally {
                        image.close()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(CAMERA_TAG, "CameraX 拍照失敗", exception)

                    mainExecutor.execute {
                        onCapturingEnd()
                        onError("拍照失敗：${exception.message}")
                    }
                }
            }
        )
    } catch (e: Exception) {
        onCapturingEnd()
        onError("無法執行拍照：${e.message}")
    }
}

// ── 儲存照片到手機相簿 ───────────────────────────────
private fun saveBitmapToGallery(context: Context, bitmap: Bitmap) {
    val filename = "WoundPhoto_${
        SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
    }.jpg"

    val values = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, filename)
        put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/WoundDetection")
    }

    var outputStream: OutputStream? = null

    try {
        val uri = context.contentResolver.insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            values
        ) ?: return

        outputStream = context.contentResolver.openOutputStream(uri)
        if (outputStream == null) {
            Log.e(CAMERA_TAG, "無法開啟相簿輸出串流")
            return
        }

        bitmap.compress(
            Bitmap.CompressFormat.JPEG,
            90,
            outputStream
        )

        Log.d(CAMERA_TAG, "照片已儲存：$uri")

    } catch (e: Exception) {
        Log.e(CAMERA_TAG, "照片儲存失敗", e)

    } finally {
        try {
            outputStream?.close()
        } catch (e: Exception) {
            Log.e(CAMERA_TAG, "關閉相簿串流失敗", e)
        }
    }
}
