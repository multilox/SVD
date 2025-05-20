import os
import numpy as np
import cv2
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt

def calculate_metrics(original, compressed, grayscale=True):
    """Вычисляет PSNR и SSIM между оригиналом и сжатым изображением."""
    if grayscale:
        psnr = compare_psnr(original, compressed, data_range=1.0)

        win_size = min(7, original.shape[0], original.shape[1])
        win_size = win_size if win_size % 2 == 1 else win_size - 1  # делаем нечётным
        ssim = compare_ssim(original, compressed, data_range=1.0, win_size=win_size)
    else:
        psnr = compare_psnr(original, compressed, data_range=1.0)

        win_size = min(7, original.shape[0], original.shape[1])
        win_size = win_size if win_size % 2 == 1 else win_size - 1  # делаем нечётным
        ssim = compare_ssim(original, compressed, data_range=1.0,
                           channel_axis=2, win_size=win_size)
    return psnr, ssim


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


def apply_gaussian_filter_and_save(compressed_images, k_values, output_dir, image_name, grayscale=True, sigma=1.0):
    """
    Применяет Гауссов фильтр к сжатым изображениям и сохраняет их в подпапку 'denoised'

    Параметры:
        compressed_images: список сжатых изображений (numpy arrays)
        k_values: список значений k, соответствующих изображениям
        output_dir: основная директория для сохранения
        image_name: имя оригинального файла (для определения формата)
        grayscale: флаг серого изображения
        sigma: параметр размытия для Гауссова фильтра
    """
    denoised_dir = os.path.join(output_dir, 'denoised')
    os.makedirs(denoised_dir, exist_ok=True)
    _, ext = os.path.splitext(image_name)

    print("\nПрименение Гауссова фильтра...")
    denoised_images = []

    for img, k in zip(tqdm(compressed_images, desc="Фильтрация"), k_values):
        if grayscale:
            # Для серых изображений
            filtered = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
            output_path = os.path.join(denoised_dir, f"denoised_k{k}{ext}")
            cv2.imwrite(output_path, (filtered * 255).clip(0, 255).astype(np.uint8))
        else:
            # Для цветных - фильтруем каждый канал отдельно
            filtered = np.zeros_like(img)
            for i in range(3):
                filtered[:, :, i] = cv2.GaussianBlur(img[:, :, i], (0, 0), sigmaX=sigma)
            output_path = os.path.join(denoised_dir, f"denoised_k{k}{ext}")
            bgr_img = cv2.cvtColor((filtered * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, bgr_img)

        denoised_images.append(filtered)

    print(f"Отфильтрованные изображения сохранены в: {denoised_dir}")
    return denoised_images


def plot_singular_values(image, grayscale=True, max_values=1200):
    """
    Строит график сингулярных значений. При большом числе — обрезает до max_values.

    grayscale — флаг для ч/б или цветного изображения.
    max_values — максимум отображаемых сингулярных значений.
    """
    if grayscale:
        U, S, Vt = np.linalg.svd(image, full_matrices=False)
        singular_values = S
        label = "Grayscale"
    else:
        singular_values = []
        for i in range(3):
            _, S, _ = np.linalg.svd(image[:, :, i], full_matrices=False)
            singular_values.append(S[:max_values])
        label = ["Red", "Green", "Blue"]

    plt.figure(figsize=(10, 5))
    if grayscale:
        plt.plot(singular_values[:max_values], label=label)
    else:
        colors = ['r', 'g', 'b']
        for i in range(3):
            plt.plot(singular_values[i], color=colors[i], label=label[i])

    plt.title(f"Сингулярные значения (макс. {max_values})")
    plt.xlabel("Индекс")
    plt.ylabel("Сингулярное значение")
    plt.yscale("log")  # логарифмическая шкала для лучшей читаемости
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def find_optimal_k(image, energy_threshold=0.99, grayscale=True):
    """Находит оптимальное k, при котором сохраняется заданная доля энергии."""
    if not grayscale:
        image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0

    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2)
    k_opt = np.searchsorted(cumulative_energy, energy_threshold * total_energy) + 1
    print(f"Оптимальное k для сохранения {energy_threshold*100:.1f}% энергии: {k_opt}")
    return k_opt
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
image_name = 'image3.jpg'
output_dir = 'resized'
k_values = [512]
gray = False
use_gpu = True  # <-- Включи True для GPU, если установлен CuPy
# Дополнительные функции
enable_plot_singular = True
enable_find_optimal_k = True
energy_threshold = 0.999  # Сколько энергии хотим сохранить

original_image = load_and_convert_image(image_dir, image_name, grayscale=gray)
print(f"Сжатие изображений... Используется {'GPU' if use_gpu and GPU_AVAILABLE else 'CPU'}")

# Поиск оптимального k
if enable_find_optimal_k:
    optimal_k = find_optimal_k(original_image, energy_threshold=energy_threshold, grayscale=gray)
    k_values = [optimal_k]  # Заменяем список значений на один — оптимальный

# Отображение сингулярных значений
if enable_plot_singular:
    plot_singular_values(original_image, grayscale=gray)


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

# Метрики качества
# print("\nМетрики качества:")
# for img, k in zip(compressed_images, k_values):
#     psnr, ssim = calculate_metrics(original_image, img, grayscale=gray)
#     print(f"k = {k}: PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}")


# sigma = 1.5
# denoised_images = apply_gaussian_filter_and_save(
#     compressed_images,
#     k_values,
#     output_dir,
#     image_name,
#     grayscale=gray,
#     sigma=sigma
# )

#
# print("\nМетрики качества после фильтрации Гауссом:")
# for img, k in zip(denoised_images, k_values):
#     psnr, ssim = calculate_metrics(original_image, img, grayscale=gray)
#     print(f"k = {k}: PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}")