import cv2
import numpy as np
import joblib
import logging
import time

import io
import pandas as pd

from telegram import InputFile
from math import log
from skimage.feature import local_binary_pattern
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# ==============================
# CONFIGURACIÓN
# ==============================

TOKEN = ""
MODEL_PATH = "fruit_classifier.pkl"

# ==============================
# LOGGING
# ==============================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ==============================
# CARGAR MODELO
# ==============================

model = joblib.load(MODEL_PATH)
print("✅ Modelo cargado correctamente")

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

    segmented = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("segmented.jpg", segmented)

    return mask, contour

def segment_fruit_grabcut(img):

    mask = np.zeros(img.shape[:2], np.uint8)

    bg_model = np.zeros((1,65), np.float64)
    fg_model = np.zeros((1,65), np.float64)

    height, width = img.shape[:2]
    rect = (10, 10, width-20, height-20)

    cv2.grabCut(
        img,
        mask,
        rect,
        bg_model,
        fg_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    mask2 = np.where(
        (mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    contours, _ = cv2.findContours(
        mask2,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)

    final_mask = np.zeros_like(mask2)
    cv2.drawContours(final_mask, [contour], -1, 255, -1)

    segmented = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("segmented.jpg", segmented)

    return final_mask, contour


# ==============================
# FUNCIÓN DE EXTRACCIÓN
# ==============================

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

    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = [-np.sign(h)*log(abs(h)+1e-12) for h in hu]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_vals = hsv[:,:,0][mask==255]
    s_vals = hsv[:,:,1][mask==255]
    v_vals = hsv[:,:,2][mask==255]

    mean_h = np.mean(h_vals)
    mean_s = np.mean(s_vals)
    mean_v = np.mean(v_vals)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_vals = lbp[mask==255]
    lbp_mean = np.mean(lbp_vals)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges[mask==255]) / area

    features = [
        area,
        perimeter,
        aspect_ratio,
        circularity,
        solidity,
        equi_diameter,
        mean_h,
        mean_s,
        mean_v,
        lbp_mean,
        edge_density,
        hu_log[0],
        hu_log[1],
        hu_log[2]
    ]

    return np.array(features).reshape(1, -1)



# ==============================
# COMANDO START
# ==============================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🍎 Envíame una imagen de una fruta y te diré cuál es."
    )

# ==============================
# MANEJADOR DE IMÁGENES
# ==============================

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):

    # Descargar imagen
    photo = update.message.photo[-1]
    file = await photo.get_file()
    await file.download_to_drive("input.jpg")

    # Leer imagen
    img = cv2.imread("input.jpg")

    # Extraer características
    features = extract_features(img)

    # Predicción
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)

    # Responder
    await update.message.reply_text(f"🍏 Predicción: {prediction[0]}")

    print("\nPredicción:", prediction[0])

    print("\nProbabilidades:")
    for fruit, prob in zip(model.classes_, probabilities[0]):
        print(f"{fruit}: {prob:.4f}")





# ==============================
# MAIN
# ==============================

def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("🤖 Bot corriendo...")
    app.run_polling()

if __name__ == "__main__":
    main()
