
package com.example.literttest.ui.screens

import android.net.Uri
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.example.literttest.ui.viewmodel.WoundDetectionViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private val DarkRed = Color(0xFFB71C1C)
private val LightGray = Color(0xFFF5F5F5)
private val DarkGray = Color(0xFF333333)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun UploadScreen(
    navController: NavController,
    viewModel: WoundDetectionViewModel = viewModel()
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    var previewBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isLoadingImage by remember { mutableStateOf(false) }
    var hasNewUpload by remember { mutableStateOf(false) }
    var uploadMessage by remember { mutableStateOf("") }

    val isDetectionProcessing = viewModel.isProcessing
    val detectionResult = viewModel.detectionResult
    val detectionError = viewModel.errorMessage

    val isBusy = isLoadingImage || isDetectionProcessing || hasNewUpload

    // ── 每次進入上傳頁面，清除之前的 AI 結果 ─────────────
    LaunchedEffect(Unit) {
        viewModel.resetResult()
    }

    // ── 圖片選擇器 ──────────────────────────────────────
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null && !isBusy) {
            selectedImageUri = uri
            previewBitmap = null
            uploadMessage = ""
            hasNewUpload = false
            viewModel.resetResult()
        }
    }

    // ── 顯示選擇圖片的預覽 ──────────────────────────────
    LaunchedEffect(selectedImageUri) {
        val uri = selectedImageUri

        previewBitmap = null

        if (uri != null) {
            try {
                previewBitmap = withContext(Dispatchers.IO) {
                    val options = BitmapFactory.Options().apply {
                        inJustDecodeBounds = true
                    }

                    context.contentResolver.openInputStream(uri)?.use {
                        BitmapFactory.decodeStream(it, null, options)
                    }

                    if (options.outWidth <= 0 || options.outHeight <= 0) {
                        return@withContext null
                    }

                    var sampleSize = 1

                    while (maxOf(options.outWidth, options.outHeight) / sampleSize > 1024) {
                        sampleSize *= 2
                    }

                    val decodeOptions = BitmapFactory.Options().apply {
                        inSampleSize = sampleSize
                    }

                    context.contentResolver.openInputStream(uri)?.use {
                        BitmapFactory.decodeStream(it, null, decodeOptions)
                    }
                }
            } catch (e: Exception) {
                Log.e("UploadScreen", "圖片預覽讀取失敗", e)
                uploadMessage = "圖片預覽失敗: ${e.message}"
            }
        }
    }

    // ── 本次推論完成後，才允許跳轉結果頁 ────────────────
    LaunchedEffect(detectionResult, isDetectionProcessing, hasNewUpload, detectionError) {
        if (hasNewUpload && !isDetectionProcessing) {

            if (detectionResult != null) {
                Log.d("UploadScreen", "本次上傳分析完成，跳轉 ResultScreen")

                hasNewUpload = false

                navController.navigate("result") {
                    popUpTo("upload") { saveState = true }
                    launchSingleTop = true
                }

            } else if (detectionError != null) {
                Log.e("UploadScreen", "AI 分析失敗：$detectionError")

                uploadMessage = detectionError
                hasNewUpload = false
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("上傳圖片") },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "返回",
                            tint = Color.White
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

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .background(LightGray)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Text(
                text = "上傳傷口圖片",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = DarkGray,
                modifier = Modifier.padding(bottom = 24.dp)
            )

            // ── 圖片預覽 ──────────────────────────────────
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(250.dp)
                    .padding(bottom = 16.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                when {
                    previewBitmap != null -> {
                        Image(
                            bitmap = previewBitmap!!.asImageBitmap(),
                            contentDescription = "已選擇的傷口圖片",
                            contentScale = ContentScale.Fit,
                            modifier = Modifier
                                .fillMaxSize()
                                .background(Color.Black)
                        )
                    }

                    selectedImageUri != null -> {
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(LightGray),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                "圖片已選擇",
                                color = DarkGray
                            )
                        }
                    }

                    else -> {
                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .padding(16.dp),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.CloudUpload,
                                contentDescription = "上傳",
                                modifier = Modifier.size(64.dp),
                                tint = DarkRed
                            )

                            Text(
                                text = "未選擇圖片",
                                color = DarkGray,
                                fontSize = 14.sp,
                                modifier = Modifier.padding(top = 8.dp)
                            )
                        }
                    }
                }
            }

            // ── 選擇圖片按鈕 ──────────────────────────────
            Button(
                onClick = {
                    imagePickerLauncher.launch("image/*")
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = DarkRed,
                    disabledContainerColor = Color.Gray
                ),
                shape = RoundedCornerShape(8.dp),
                enabled = !isBusy
            ) {
                Icon(
                    imageVector = Icons.Default.CloudUpload,
                    contentDescription = "選擇",
                    modifier = Modifier.padding(end = 8.dp),
                    tint = Color.White
                )

                Text(
                    "選擇圖片",
                    color = Color.White,
                    fontWeight = FontWeight.Bold
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // ── 開始 AI 分析 ───────────────────────────────
            Button(
                onClick = {
                    val uri = selectedImageUri

                    if (uri == null) {
                        uploadMessage = "請先選擇圖片"
                    } else if (!isBusy) {

                        scope.launch {
                            isLoadingImage = true
                            uploadMessage = ""

                            // 清除前一次結果
                            viewModel.resetResult()

                            try {
                                // 在背景執行緒讀取圖片，避免主畫面卡住
                                val bitmap = withContext(Dispatchers.IO) {
                                    context.contentResolver.openInputStream(uri)?.use { stream ->
                                        BitmapFactory.decodeStream(stream)
                                    }
                                }

                                if (bitmap != null) {
                                    Log.d("UploadScreen", "開始分析新圖片")

                                    // 標記本次確實有新的上傳任務
                                    hasNewUpload = true

                                    // 送進 YOLO + MobileNet，並儲存歷史紀錄
                                    viewModel.detectWoundFromBitmap(
                                        bitmap,
                                        source = "上傳"
                                    )

                                    uploadMessage = "圖片已上傳，AI 分析中..."

                                } else {
                                    hasNewUpload = false
                                    uploadMessage = "無法讀取圖片，請重新選擇"
                                }

                            } catch (e: Exception) {
                                Log.e("UploadScreen", "讀取圖片失敗", e)

                                hasNewUpload = false
                                uploadMessage = "讀取失敗: ${e.message}"

                            } finally {
                                isLoadingImage = false
                            }
                        }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF4CAF50),
                    disabledContainerColor = Color.Gray
                ),
                shape = RoundedCornerShape(8.dp),
                enabled = selectedImageUri != null && !isBusy
            ) {
                if (isBusy) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        color = Color.White,
                        strokeWidth = 2.dp
                    )

                    Spacer(modifier = Modifier.width(8.dp))

                    Text("處理中...", color = Color.White)

                } else {
                    Text(
                        "開始分析",
                        color = Color.White,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            // ── 狀態訊息 ──────────────────────────────────
            val displayMessage = detectionError ?: uploadMessage
            val hasError = detectionError != null ||
                    "失敗" in displayMessage ||
                    "無法" in displayMessage ||
                    "請先" in displayMessage

            if (displayMessage.isNotEmpty()) {
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 16.dp),
                    color = if (hasError) Color(0xFFFFEBEE) else Color(0xFFE8F5E9),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(
                        text = displayMessage,
                        modifier = Modifier.padding(12.dp),
                        color = if (hasError) DarkRed else Color(0xFF2E7D32),
                        fontSize = 12.sp
                    )
                }
            }

            Spacer(modifier = Modifier.weight(1f))
        }
    }
}
