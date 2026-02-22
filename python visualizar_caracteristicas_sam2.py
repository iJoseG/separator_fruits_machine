import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo (guarda figuras, no muestra ventanas)
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.fftpack import fft, ifft
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
CHECKPOINT = "checkpoints/sam2_hiera_large.pt"   # Ruta al modelo
CONFIG = "sam2_hiera_l.yaml"                      # Archivo de configuración
DEVICE = "cpu"                                     # o "cuda" si tienes GPU
IMAGE_PATH = "tu_imagen.jpg"                       # Imagen a analizar
ENERGY_THRESHOLD = 0.99                            # 99% de energía para Fourier
OUTPUT_DIR = "resultados_fourier"                  # Carpeta donde se guardan las imágenes

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# CARGAR MODELO SAM2
# ==========================================================
print("Cargando modelo SAM2 large...")
sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE)
mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
print("Modelo listo.")

# ==========================================================
# FUNCIONES PARA DESCRIPTORES DE FOURIER
# ==========================================================
def fourier_descriptors_with_energy(contour, threshold=0.99):
    """
    Calcula descriptores de Fourier a partir del contorno.
    Retorna:
        - magnitudes: array de magnitudes normalizadas (para usar como features)
        - num_opt: número de coeficientes necesarios para threshold% de energía
        - fourier_full: todos los coeficientes complejos (sin normalizar)
    """
    pts = contour[:, 0, :].astype(np.float32)
    if len(pts) < 10:
        return None, 0, None

    # Representación compleja
    z = pts[:, 0] + 1j * pts[:, 1]

    # DFT
    fourier = fft(z)
    ac = fourier[1:]  # Coeficientes AC (frecuencias positivas)

    # Normalización para descriptores (magnitudes)
    if np.abs(ac[0]) > 0:
        magnitudes = np.abs(ac) / np.abs(ac[0])
    else:
        magnitudes = np.abs(ac)

    # Energía acumulada (con coeficientes originales)
    energia_total = np.sum(np.abs(ac)**2)
    if energia_total == 0:
        return magnitudes, 0, fourier

    energia_acum = np.cumsum(np.abs(ac)**2) / energia_total
    num_opt = np.searchsorted(energia_acum, threshold) + 1
    num_opt = min(num_opt, len(ac))

    return magnitudes, num_opt, fourier

def reconstruct_from_fourier(fourier_coeffs, num_harmonics, num_points=200):
    """
    Reconstruye el contorno usando los primeros 'num_harmonics' armónicos.
    """
    N = len(fourier_coeffs)
    K = min(num_harmonics, N//2)  # número de armónicos a usar

    # Crear array de coeficientes con ceros para las frecuencias no usadas
    coeffs_rec = np.zeros(N, dtype=complex)
    coeffs_rec[0] = fourier_coeffs[0]  # DC
    for k in range(1, K+1):
        if k < N:
            coeffs_rec[k] = fourier_coeffs[k]
        if N - k < N:
            coeffs_rec[N - k] = fourier_coeffs[N - k]  # coeficiente negativo

    # Reconstruir con IDFT
    z_rec = ifft(coeffs_rec) * N
    x_rec = np.real(z_rec)
    y_rec = np.imag(z_rec)
    return np.vstack([x_rec, y_rec]).T

# ==========================================================
# FUNCIÓN DE VISUALIZACIÓN Y GUARDADO
# ==========================================================
def visualize_and_save(img, mask, contour, idx):
    # Extraer descriptores Fourier
    magnitudes, num_opt, fourier_full = fourier_descriptors_with_energy(contour, ENERGY_THRESHOLD)

    # Crear figura
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Objeto {idx+1} - {num_opt} armónicos explican {ENERGY_THRESHOLD*100:.0f}% energía', fontsize=14)

    # 1. Imagen original + contorno
    img_contour = img.copy()
    cv2.drawContours(img_contour, [contour], -1, (0, 255, 0), 2)
    axs[0, 0].imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Contorno original')
    axs[0, 0].axis('off')

    # 2. Máscara
    axs[0, 1].imshow(mask, cmap='gray')
    axs[0, 1].set_title('Máscara')
    axs[0, 1].axis('off')

    # 3. Región segmentada
    masked = cv2.bitwise_and(img, img, mask=mask)
    axs[0, 2].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    axs[0, 2].set_title('Región segmentada')
    axs[0, 2].axis('off')

    # 4. Textura LBP
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_masked = np.zeros_like(lbp)
    lbp_masked[mask == 255] = lbp[mask == 255]
    axs[1, 0].imshow(lbp_masked, cmap='jet')
    axs[1, 0].set_title('Textura LBP')
    axs[1, 0].axis('off')

    # 5. Canal Hue (color)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_masked = np.zeros_like(hsv[:,:,0])
    h_masked[mask == 255] = hsv[:,:,0][mask == 255]
    axs[1, 1].imshow(h_masked, cmap='hsv')
    axs[1, 1].set_title('Canal Hue')
    axs[1, 1].axis('off')

    # 6. Reconstrucción Fourier (superpuesta al contorno original)
    if fourier_full is not None and num_opt > 0:
        recon = reconstruct_from_fourier(fourier_full, num_opt, num_points=200)
        contour_orig = contour[:, 0, :]
        axs[1, 2].plot(contour_orig[:, 0], contour_orig[:, 1], 'gray', linewidth=1, label='Original')
        axs[1, 2].plot(recon[:, 0], recon[:, 1], 'r-', linewidth=2, label=f'Reconstrucción ({num_opt} armónicos)')
        axs[1, 2].set_title('Comparación de contornos')
        axs[1, 2].legend()
        axs[1, 2].set_aspect('equal')
        axs[1, 2].invert_yaxis()
    else:
        axs[1, 2].text(0.5, 0.5, 'No disponible', ha='center')
    axs[1, 2].axis('off')

    plt.tight_layout()
    # Guardar figura
    output_path = os.path.join(OUTPUT_DIR, f"objeto_{idx+1}_visualizacion.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Visualización guardada en {output_path}")

    # --- Características numéricas (incluyendo desviaciones estándar de HSV) ---
    area = cv2.countNonZero(mask)
    perim = cv2.arcLength(contour, True)
    rect = cv2.boundingRect(contour)
    aspect_ratio = rect[2] / rect[3] if rect[3] != 0 else 0
    circularity = (4 * np.pi * area) / (perim**2) if perim > 0 else 0

    # Estadísticas de color HSV
    h_vals = hsv[:,:,0][mask==255]
    s_vals = hsv[:,:,1][mask==255]
    v_vals = hsv[:,:,2][mask==255]

    if len(h_vals) > 0:
        mean_h = np.mean(h_vals)
        mean_s = np.mean(s_vals)
        mean_v = np.mean(v_vals)
        std_h = np.std(h_vals)
        std_s = np.std(s_vals)
        std_v = np.std(v_vals)
    else:
        mean_h = mean_s = mean_v = 0
        std_h = std_s = std_v = 0

    # Imprimir en consola
    print(f"\n--- Objeto {idx+1} ---")
    print(f"Área: {area} px")
    print(f"Perímetro: {perim:.2f}")
    print(f"Relación aspecto: {aspect_ratio:.3f}")
    print(f"Circularidad: {circularity:.3f}")
    print(f"Hue medio: {mean_h:.1f}  |  Desv. Hue: {std_h:.2f}")
    print(f"Saturación media: {mean_s:.1f}  |  Desv. Sat.: {std_s:.2f}")
    print(f"Value medio: {mean_v:.1f}  |  Desv. Value: {std_v:.2f}")
    if magnitudes is not None:
        print(f"Primeros 5 descriptores Fourier (normalizados): {np.round(magnitudes[:5], 4)}")
        print(f"Nº óptimo para {ENERGY_THRESHOLD*100:.0f}% energía: {num_opt} de {len(magnitudes)} totales")

# ==========================================================
# PROCESAR IMAGEN
# ==========================================================
def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: No se pudo cargar {IMAGE_PATH}")
        return

    print(f"Imagen cargada: {img.shape}")

    # Redimensionar si es muy grande para acelerar
    h, w = img.shape[:2]
    max_size = 1024
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        print(f"Redimensionada a {new_w}x{new_h}")

    # Generar máscaras con SAM2
    print("Generando máscaras con SAM2 (puede tardar)...")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks_data = mask_generator.generate(img_rgb)
    print(f"Se encontraron {len(masks_data)} objetos.")

    for i, mask_data in enumerate(masks_data):
        mask = mask_data['segmentation'].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        visualize_and_save(img, mask, contour, i)

    print(f"\nProceso completado. Las imágenes se guardaron en '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()