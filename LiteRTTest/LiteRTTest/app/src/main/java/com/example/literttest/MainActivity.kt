
package com.example.literttest

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.literttest.ui.screens.*
import com.example.literttest.ui.viewmodel.WoundDetectionViewModel

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val navController = rememberNavController()
                    val viewModel: WoundDetectionViewModel = viewModel()

                    LaunchedEffect(Unit) {
                        viewModel.initializeModels(this@MainActivity)
                    }

                    NavHost(
                        navController = navController,
                        startDestination = "home"
                    ) {

                        // ════════════════════════════════════════════
                        // 首頁
                        // ════════════════════════════════════════════
                        composable("home") {

                            BackHandler {
                                // 首頁按返回鍵，不退出 APP
                            }

                            HomeScreen(navController)
                        }

                        // ════════════════════════════════════════════
                        // 相機偵測
                        // ════════════════════════════════════════════
                        composable("camera") {
                            CameraScreen(navController, viewModel)
                        }

                        // ════════════════════════════════════════════
                        // 上傳圖片
                        // ════════════════════════════════════════════
                        composable("upload") {
                            UploadScreen(navController, viewModel)
                        }

                        // ════════════════════════════════════════════
                        // AI 檢測結果
                        // ════════════════════════════════════════════
                        composable("result") {

                            val bitmap = viewModel.capturedBitmap
                            val result = viewModel.detectionResult

                            if (bitmap != null && result != null) {

                                ResultScreen(
                                    bitmap = bitmap,
                                    result = result,

                                    onRetake = {
                                        // 先返回上一個頁面
                                        val returned = navController.popBackStack()

                                        if (!returned) {
                                            navController.navigate("home") {
                                                popUpTo("home") {
                                                    inclusive = false
                                                }
                                                launchSingleTop = true
                                            }
                                        }

                                        // 返回操作後，清除前一次結果
                                        viewModel.resetResult()
                                    },

                                    onHome = {
                                        // 返回既有首頁，不刪除首頁
                                        navController.navigate("home") {
                                            popUpTo("home") {
                                                inclusive = false
                                            }
                                            launchSingleTop = true
                                        }

                                        // 導覽操作後才清除結果
                                        viewModel.resetResult()
                                    }
                                )

                            } else {

                                // 沒有結果時顯示提示，不自動 popBackStack
                                Box(
                                    modifier = Modifier.fillMaxSize(),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Column(
                                        horizontalAlignment = Alignment.CenterHorizontally,
                                        verticalArrangement = Arrangement.spacedBy(16.dp)
                                    ) {
                                        Text(
                                            text = "目前沒有可顯示的檢測結果",
                                            style = MaterialTheme.typography.titleMedium
                                        )

                                        Button(
                                            onClick = {
                                                navController.navigate("home") {
                                                    popUpTo("home") {
                                                        inclusive = false
                                                    }
                                                    launchSingleTop = true
                                                }

                                                viewModel.resetResult()
                                            }
                                        ) {
                                            Text("返回首頁")
                                        }
                                    }
                                }
                            }
                        }

                        // ════════════════════════════════════════════
                        // 病患管理
                        // ════════════════════════════════════════════
                        composable("patients") {
                            PatientScreen(navController)
                        }

                        // ════════════════════════════════════════════
                        // 歷史紀錄
                        // ════════════════════════════════════════════
                        composable("history") {
                            HistoryScreen(navController)
                        }

                        // ════════════════════════════════════════════
                        // 設定頁面
                        // ════════════════════════════════════════════
                        composable("settings") {
                            SettingsScreen(navController)
                        }
                    }
                }
            }
        }
    }
}
