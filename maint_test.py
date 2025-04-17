import os
import numpy as np
import cv2
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def calculate_metrics(original, compressed, grayscale=True):
    """Вычисляет PSNR и SSIM между оригиналом и сжатым изображением."""
    if grayscale:
        psnr = compare_psnr(original, compressed, data_range=1.0)
        ssim = compare_ssim(original, compressed, data_range=1.0)
    else:
        psnr = compare_psnr(original, compressed, data_range=1.0)
        ssim = compare_ssim(original, compressed, multichannel=True, data_range=1.0)
    return psnr, ssim

# Попробуем подключить CuPy для GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None  # заглушка

def load_and_convert_image(image_dir, image_name, grayscale=True):
    """Загружает изображение из указанной директории."""
    full_path = os.path.join(image_dir, image_name)
    if grayscale:
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {full_path}")
        return image / 255.0
    else:
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {full_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image / 255.0

def svd_compress(image, k, use_gpu=False):
    """Выполняет SVD с использованием CPU или GPU."""
    if use_gpu and GPU_AVAILABLE:
        image_gpu = cp.asarray(image)
        U, S, Vt = cp.linalg.svd(image_gpu, full_matrices=False)
        S_k = cp.diag(S[:k])
        compressed_gpu = U[:, :k] @ S_k @ Vt[:k, :]
        return cp.asnumpy(compressed_gpu)
    else:
        U, S, Vt = np.linalg.svd(image, full_matrices=False)
        S_k = np.diag(S[:k])
        return U[:, :k] @ S_k @ Vt[:k, :]

def compress_color_image(image, k, use_gpu=False):
    """Применяет SVD сжатие отдельно к каждому каналу."""
    channels = []
    for i in range(3):
        compressed = svd_compress(image[:, :, i], k, use_gpu=use_gpu)
        channels.append(compressed)
    return np.stack(channels, axis=2)

def save_compressed_images(original, compressed_images, k_values, output_dir, image_name, grayscale=True):
    """Сохраняет оригинальное и сжатые изображения с сохранением формата."""
    os.makedirs(output_dir, exist_ok=True)
    _, ext = os.path.splitext(image_name)

    if grayscale:
        original_path = os.path.join(output_dir, f"original{ext}")
        cv2.imwrite(original_path, (original * 255).astype(np.uint8))
        for img, k in zip(compressed_images, k_values):
            output_path = os.path.join(output_dir, f"compressed_k{k}{ext}")
            cv2.imwrite(output_path, (img * 255).astype(np.uint8))
    else:
        original_bgr = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"original{ext}"), original_bgr)
        for img, k in zip(compressed_images, k_values):
            bgr_img = cv2.cvtColor((img * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            output_path = os.path.join(output_dir, f"compressed_k{k}{ext}")
            cv2.imwrite(output_path, bgr_img)

# === НАСТРОЙКИ ===
image_dir = 'media'
image_name = 'image1.jpg'
output_dir = 'resized'
k_values = [5, 20, 50, 100]
gray = False
use_gpu = False  # <-- Включи True для GPU, если установлен CuPy

original_image = load_and_convert_image(image_dir, image_name, grayscale=gray)

print(f"Сжатие изображений... Используется {'GPU' if use_gpu and GPU_AVAILABLE else 'CPU'}")

if gray:
    compressed_images = [
        svd_compress(original_image, k, use_gpu=use_gpu)
        for k in tqdm(k_values, desc="Обработка SVD")
    ]
else:
    compressed_images = [
        compress_color_image(original_image, k, use_gpu=use_gpu)
        for k in tqdm(k_values, desc="Обработка SVD (цвет)")
    ]

print("Сохранение изображений...")
save_compressed_images(original_image, compressed_images, k_values, output_dir, image_name, grayscale=gray)

print(f"Изображения сохранены в директории: {output_dir}")
print("\nМетрики качества:")
for img, k in zip(compressed_images, k_values):
    psnr, ssim = calculate_metrics(original_image, img, grayscale=gray)
    print(f"k = {k}: PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}")