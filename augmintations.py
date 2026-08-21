import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations import (PiecewiseAffine, ElasticTransform, Perspective, ShiftScaleRotate, 
                                            RandomBrightnessContrast, RandomGamma, GaussNoise,
                                            MotionBlur, HueSaturationValue, RGBShift, RandomSunFlare,
                                            InvertImg, RandomCrop)
from albumentations import CoarseDropout  # для создания пятен (альтернатива)

# --------------------- 1. Геометрические аугментации ---------------------
def apply_affine(img):
    """Сдвиг, поворот, масштабирование."""
    aug = A.Compose([ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=1)])
    return aug(image=img)['image']

def apply_perspective(img):
    """Перспективное искажение."""
    aug = A.Compose([Perspective(scale=(0.05, 0.1), p=1)])
    return aug(image=img)['image']

def apply_elastic(img):
    """Эластичная деформация (имитация неровного почерка)."""
    aug = A.Compose([ElasticTransform(alpha=150, sigma=40, p=1)])
    return aug(image=img)['image']

def apply_tps(img):
    """Искривление по ТПС (PiecewiseAffine)."""
    aug = A.Compose([PiecewiseAffine(scale=(0.01, 0.03), nb_rows=7, nb_cols=7, p=1)])
    return aug(image=img)['image']


# --------------------- 2. Фотометрические аугментации ---------------------
def apply_brightness_contrast(img):
    """Изменение яркости и контрастности."""
    aug = A.Compose([RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1)])
    return aug(image=img)['image']

def apply_gamma(img):
    """Гамма-коррекция."""
    aug = A.Compose([RandomGamma(gamma_limit=(40, 160), p=1)])
    return aug(image=img)['image']

def apply_noise(img):
    """Добавление гауссовского шума и соли-перца."""
    aug = A.Compose([
        GaussNoise(p=0.7),
        A.SaltAndPepper(p=0.3)  # иногда только шум, иногда соль-перец
    ], p=1)
    return aug(image=img)['image']

def apply_blur(img):
    """Размытие (движение или гаусс)."""
    aug = A.Compose([MotionBlur(blur_limit=30, p=1)])
    return aug(image=img)['image']

def apply_color_jitter(img):
    """Изменение цветового тона, насыщенности, сдвиг RGB."""
    aug = A.Compose([
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
        RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5)
    ], p=1)
    return aug(image=img)['image']


# --------------------- 3. Специфические искажения текста ---------------------
def apply_fading(img):
    aug = A.Compose([
        RandomSunFlare(
            flare_roi=(0, 0, 1, 1),
            src_radius=400,
            src_color=(255, 255, 255),
            p=0.5
        ),
        CoarseDropout(
            num_holes_range=(2, 5),          # больше дырок
            hole_height_range=(20, 80),      # крупнее
            hole_width_range=(20, 80),
            fill=0,
            p=0.7                            # чаще применять
        ),
        # Дополнительный эффект выцветания:
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3)
    ], p=1)
    return aug(image=img)['image']

def apply_random_crop(img):
    """Обрезка краёв (возвращается с изменённым размером)."""
    h, w = img.shape[:2]
    # Обрежем случайную область, но сохраним пропорции, чтобы не потерять текст
    aug = A.Compose([RandomCrop(height=int(h*0.8), width=int(w*0.8), p=1)])
    return aug(image=img)['image']

def apply_line_breaks(img):
    img = img.copy()
    h, w = img.shape[:2]
    num_lines = np.random.randint(2, 4)   # 2 или 3 полосы
    for _ in range(num_lines):
        y = np.random.randint(int(h*0.1), int(h*0.9))
        thickness = np.random.randint(3, 12)
        cv2.rectangle(img, (0, y), (w, y+thickness), (255,255,255), -1)
    return img

def apply_thickness(img):
    """Изменение толщины шрифта (эрозия/дилатация)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Пороговая обработка для получения бинарного изображения
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2,2), np.uint8)
    if np.random.rand() > 0.5:
        # Утолщение (дилатация)
        thick = cv2.dilate(binary, kernel, iterations=1)
    else:
        # Утоньшение (эрозия)
        thick = cv2.erode(binary, kernel, iterations=1)
    # Инвертируем обратно
    thick = cv2.bitwise_not(thick)
    # Преобразуем обратно в RGB (3 канала)
    thick_rgb = cv2.cvtColor(thick, cv2.COLOR_GRAY2RGB)
    return thick_rgb

def apply_invert(img):
    """Инверсия цветов."""
    aug = A.Compose([InvertImg(p=1)])
    return aug(image=img)['image']