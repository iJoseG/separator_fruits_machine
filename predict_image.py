import cv2
import numpy as np
import joblib
from math import log
from skimage.feature import local_binary_pattern

MODEL_PATH = "fruit_classifier.pkl"

# ==========================================================
# CARGAR MODELO
# ==========================================================

model = joblib.load(MODEL_PATH)
print("Modelo cargado correctamente.")

# ==========================================================
# SEGMENTACIÓN (usar la robusta)
# ==========================================================

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

    return final_mask, contour


# ==========================================================
# EXTRAER FEATURES (igual que entrenamiento)
# ==========================================================

def extract_features(img):

    mask, contour = segment_fruit_grabcut(img)

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


# ==========================================================
# FUNCIÓN DE PREDICCIÓN
# ==========================================================

def predict(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Error cargando imagen.")
        return

    features = extract_features(img)

    if features is None:
        print("No se pudo segmentar la fruta.")
        return

    prediction = model.predict(features)
    probabilities = model.predict_proba(features)

    print("\nPredicción:", prediction[0])

    print("\nProbabilidades:")
    for fruit, prob in zip(model.classes_, probabilities[0]):
        print(f"{fruit}: {prob:.4f}")

    #cv2.imshow("Imagen", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    image_path = input("Ruta de la imagen a predecir: ")
    predict(image_path)
