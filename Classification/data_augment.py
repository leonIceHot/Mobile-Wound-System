import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from tqdm import tqdm # 引入 tqdm 庫

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# ========== 擴增函式 ==========
def augment_image(img):
    """對單張圖片隨機套用一種或多種擴增方式

    包含以下方法:
    1. Flipping (翻轉)
    2. ColorJitter - Brightness (亮度調整)
    3. ColorJitter - Contrast (對比度調整)
    4. GaussianBlur (高斯模糊)
    5. Shifting (平移)
    6. Border (邊界填充, 搭配 Rotate / Shifting 使用)
    7. Rotate (旋轉)
    """

    BORDER_COLOR = (255, 255, 255)  # Border 填充顏色

    # 1. Flipping (水平翻轉)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. Rotate (旋轉) + Border (邊界填充)
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, expand=True, fillcolor=BORDER_COLOR)

    # 3. ColorJitter - Brightness (亮度調整)
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)

    # 4. ColorJitter - Contrast (對比度調整)
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # 5. GaussianBlur (高斯模糊)
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # 6. Shifting (平移) + Border (邊界填充)
    if random.random() < 0.3:
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        img = img.transform(
            img.size, Image.AFFINE,
            (1, 0, dx, 0, 1, dy),
            fillcolor=BORDER_COLOR
        )

    return img


# ========== 複製原圖到目的資料夾 ==========
def copy_original_images(src_folder: Path, dst_folder: Path):
    dst_folder.mkdir(parents=True, exist_ok=True)
    images = [f for f in src_folder.iterdir() if f.suffix.lower() in IMG_EXTENSIONS]

    for img_path in images:
        dst_path = dst_folder / img_path.name
        if not dst_path.exists():
            shutil.copy2(img_path, dst_path)

    return images


# ========== 針對單一類別資料夾進行擴增 ==========
def augment_class_folder(src_folder: Path, dst_folder: Path, target_count: int):
    print(f"  📁 類別: {src_folder.name}")

    original_images = copy_original_images(src_folder, dst_folder)
    current_count = len(list(dst_folder.iterdir()))
    print(f"     原始: {len(original_images)}, 目前: {current_count}, 目標: {target_count}")

    if current_count >= target_count:
        print("     ✅ 已達目標數量, 不需擴增")
        return

    if len(original_images) == 0:
        print("     ⚠️ 沒有原始圖片, 略過")
        return

    need = target_count - current_count
    print(f"     ➕ 需要擴增 {need} 張")

    idx = 0
    # 使用 tqdm 顯示擴增進度
    for _ in tqdm(range(need), desc=f"       擴增 {src_folder.name}", unit="張"):
        if original_images: # 檢查列表是否為空
            src_file = original_images[idx % len(original_images)]
            try:
                img = Image.open(src_file).convert("RGB")
                aug_img = augment_image(img)

                new_name = f"{src_file.stem}_aug{idx}{src_file.suffix}"
                save_path = dst_folder / new_name

                # 確保文件名唯一，避免覆蓋
                attempt_idx = 0
                while save_path.exists():
                    attempt_idx += 1
                    new_name = f"{src_file.stem}_aug{idx}_{attempt_idx}{src_file.suffix}"
                    save_path = dst_folder / new_name

                aug_img.save(save_path)
                idx += 1 # 只有成功保存後才增加原始圖片的索引
            except Exception as e:
                # 這裡不應該減少 need，因為這張圖片的擴增失敗了，還需要繼續擴增
                # 打印錯誤並嘗試下一張圖片
                print(f"\n     ⚠️ 錯誤: {src_file.name} -> {e}")
                idx += 1 # 即使失敗也嘗試下一張原始圖片
        else:
            # 如果 original_images 為空，說明沒有圖片可以擴增，直接跳出
            print("\n     ⚠️ 沒有原始圖片可以擴增，終止擴增。")
            break

    final_count = len(list(dst_folder.iterdir()))
    print(f"     ✅ 完成, 現有 {final_count} 張")


# ========== 只複製、不擴增 (用於 val / test) ==========
def copy_only_class_folder(src_folder: Path, dst_folder: Path):
    print(f"  📁 類別: {src_folder.name}")
    images = copy_original_images(src_folder, dst_folder)
    print(f"     ✅ 已複製 {len(images)} 張 (不擴增)")


# ========== 處理單一 split (train/val/test) ==========
# ========== 處理單一 split (train/val/test) ==========
def process_split(split_name, src_root: Path, dst_root: Path, target_counts: dict, do_augment: bool):
    src_split = src_root / split_name
    dst_split = dst_root / split_name

    if not src_split.exists():
        print(f"⚠️ 找不到 {src_split}, 略過")
        return

    print(f"\n===== 處理 {split_name.upper()} =====")

    # 修正：將 'f f' 改為 'f in'
    class_folders = [f for f in src_split.iterdir() if f.is_dir()]

    # 使用 tqdm 顯示類別資料夾的處理進度
    for src_folder in tqdm(class_folders, desc=f"處理 {split_name} 類別資料夾", unit="個"):
        class_name = src_folder.name
        dst_folder = dst_split / class_name

        if do_augment:
            target_count = target_counts.get(class_name)
            if target_count is None:
                # 沒設定目標數量的類別, 就只複製
                copy_only_class_folder(src_folder, dst_folder)
            else:
                augment_class_folder(src_folder, dst_folder, target_count)
        else:
            copy_only_class_folder(src_folder, dst_folder)


# ========== 主程式 ==========
def main():
    # ---------- 這裡設定 ----------
    SOURCE_DIR = r"./new_wound5_files/new_wound5_8_mix_aug"        # 來源根目錄 (內含 train/val/test)
    DEST_DIR   = r"./data_augment10"    # 目的根目錄 (輸出 train/val/test)

    # 每個類別要擴增到的目標數量 (只套用在 train)
    TARGET_COUNTS = {
        "other": 1000,
        "WCC_0": 1000,
        "WCC_1": 1000,
        "WCC_2": 1000,
        "WCC_3": 1000,
        "WCC_4": 1000,
    }

    # 是否對各個 split 做擴增
    SPLIT_AUGMENT = {
        "train": False,   # train 需要擴增
        "val":   False,  # val 只複製, 不擴增
        "test":  True,  # test 只複製, 不擴增
    }
    # -------------------------------

    source_root = Path(SOURCE_DIR)
    dest_root = Path(DEST_DIR)
    dest_root.mkdir(parents=True, exist_ok=True)

    for split_name, do_augment in SPLIT_AUGMENT.items():
        process_split(split_name, source_root, dest_root, TARGET_COUNTS, do_augment)

    print("\n🎉 全部處理完成!")


if __name__ == "__main__":
    main()
