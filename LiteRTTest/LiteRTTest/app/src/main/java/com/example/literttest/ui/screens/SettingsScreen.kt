package com.example.literttest.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(navController: NavHostController) {

    var confidenceThreshold by remember { mutableStateOf(0.5f) }
    var darkMode by remember { mutableStateOf(false) }
    var autoSave by remember { mutableStateOf(true) }
    var language by remember { mutableStateOf("繁體中文") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("設定", fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "返回")
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {

            // ── 模型參數 ──────────────────────────────────
            SettingsSection(title = "模型參數") {
                Column {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            "信心閾值",
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            "%.0f%%".format(confidenceThreshold * 100),
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                    Slider(
                        value = confidenceThreshold,
                        onValueChange = { confidenceThreshold = it },
                        valueRange = 0.1f..0.9f,
                        steps = 7,
                        modifier = Modifier.fillMaxWidth()
                    )
                    Text(
                        text = "低於此閾值的偵測結果將被忽略",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                    )
                }
            }

            // ── 顯示設定 ──────────────────────────────────
            SettingsSection(title = "顯示設定") {
                SettingsToggleItem(
                    icon = Icons.Default.DarkMode,
                    iconColor = Color(0xFF5C6BC0),
                    title = "深色模式",
                    subtitle = "切換深色/淺色主題",
                    checked = darkMode,
                    onCheckedChange = { darkMode = it }
                )
                HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
                SettingsToggleItem(
                    icon = Icons.Default.Save,
                    iconColor = Color(0xFF388E3C),
                    title = "自動儲存結果",
                    subtitle = "偵測後自動存入歷史紀錄",
                    checked = autoSave,
                    onCheckedChange = { autoSave = it }
                )
            }

            // ── 語言 ──────────────────────────────────────
            SettingsSection(title = "語言") {
                listOf("繁體中文", "English").forEach { lang ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        RadioButton(
                            selected = language == lang,
                            onClick = { language = lang }
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(lang, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }

            // ── 關於 ──────────────────────────────────────
            SettingsSection(title = "關於") {
                SettingsInfoItem(
                    icon = Icons.Default.Info,
                    iconColor = Color(0xFF0288D1),
                    title = "版本",
                    value = "v1.0.0"
                )
                HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
                SettingsInfoItem(
                    icon = Icons.Default.Build,
                    iconColor = Color(0xFF795548),
                    title = "模型版本",
                    value = "YOLOv8-wound-v2"
                )
            }

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

// ── 設定區塊容器 ──────────────────────────────────────────────
@Composable
private fun SettingsSection(
    title: String,
    content: @Composable ColumnScope.() -> Unit
) {
    Column {
        Text(
            text = title,
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                content = content
            )
        }
    }
}

// ── Toggle 項目 ───────────────────────────────────────────────
@Composable
private fun SettingsToggleItem(
    icon: ImageVector,
    iconColor: Color,
    title: String,
    subtitle: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = iconColor,
            modifier = Modifier.size(24.dp)
        )
        Spacer(modifier = Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                title,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium
            )
            Text(
                subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
            )
        }
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

// ── 資訊項目 ──────────────────────────────────────────────────
@Composable
private fun SettingsInfoItem(
    icon: ImageVector,
    iconColor: Color,
    title: String,
    value: String
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = iconColor,
            modifier = Modifier.size(24.dp)
        )
        Spacer(modifier = Modifier.width(12.dp))
        Text(
            text = title,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.weight(1f)
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary
        )
    }
}
