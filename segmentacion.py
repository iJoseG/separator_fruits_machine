import cv2
import numpy as np

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


def predict(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Error cargando imagen.")
        return

    mask, contour = segment_fruit(img)

    if mask is None:
        return None
    
    segmented = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("segmented.jpg", segmented)


if __name__ == "__main__":

    image_path = input("Ruta de la imagen a predecir: ")
    predict(image_path)

        

