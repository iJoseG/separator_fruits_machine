import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import cv2
import numpy as np
import pandas as pd
import os
from math import log
from skimage.feature import local_binary_pattern

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

DATASET_PATH = "dataset"
OUTPUT_CSV = "fruit_features.csv"

# ==========================================================
# SEGMENTACIÓN
# ==========================================================

def segment_fruit(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur para reducir ruido
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Otsu
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Operaciones morfológicas
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    return mask, contour


# ==========================================================
# EXTRACCIÓN DE FEATURES
# ==========================================================

def extract_features(img):
    mask, contour = segment_fruit(img)

    if mask is None:
        return None

    area = cv2.countNonZero(mask)
    perimeter = cv2.arcLength(contour, True)

    x,y,w,h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h if h > 0 else 0

    circularity = (4*np.pi*area)/(perimeter**2) if perimeter>0 else 0

    hull = cv2.convexHull(contour)
    solidity = area / cv2.contourArea(hull)

    equi_diameter = np.sqrt(4*area/np.pi)

    # Hu Moments
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = [-np.sign(h)*log(abs(h)+1e-12) for h in hu]

    # Color (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_vals = hsv[:,:,0][mask==255]
    s_vals = hsv[:,:,1][mask==255]
    v_vals = hsv[:,:,2][mask==255]

    mean_h = np.mean(h_vals)
    mean_s = np.mean(s_vals)
    mean_v = np.mean(v_vals)

    # Textura - LBP
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_vals = lbp[mask==255]
    lbp_mean = np.mean(lbp_vals)

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges[mask==255]) / area

    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "solidity": solidity,
        "equivalent_diameter": equi_diameter,
        "mean_hue": mean_h,
        "mean_saturation": mean_s,
        "mean_value": mean_v,
        "lbp_mean": lbp_mean,
        "edge_density": edge_density,
        "hu1": hu_log[0],
        "hu2": hu_log[1],
        "hu3": hu_log[2]
    }


# ==========================================================
# PROCESAR DATASET COMPLETO
# ==========================================================

def process_dataset():
    rows = []

    for label in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            features = extract_features(img)

            if features:
                features["label"] = label
                rows.append(features)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV generado correctamente: {OUTPUT_CSV}")
    print(df.head())


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    process_dataset()
