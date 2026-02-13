import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# --- FASE 1: EXTRACCIÓN DE CARACTERÍSTICAS ---

def extraer_caracteristicas(imagen_path):
    # 1. Cargar la imagen
    img = cv2.imread(imagen_path)
    if img is None:
        return None
    
    # Redimensionar para procesar más rápido (opcional)
    img = cv2.resize(img, (300, 300))
    
    # 2. Convertir a espacio de color HSV (Mejor para detectar colores que RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 3. Eliminar el fondo (Asumimos fondo claro/blanco o distinto a la fruta)
    # Creamos una máscara para ignorar el fondo. Aquí usamos un umbral simple.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # 4. Encontrar el contorno (la silueta de la fruta)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        return None
    
    # Tomamos el contorno más grande (la fruta principal)
    c = max(contornos, key=cv2.contourArea)
    
    # --- CÁLCULO DE DATOS (Matemáticas) ---
    
    # A. Área
    area = cv2.contourArea(c)
    
    # B. Perímetro
    perimetro = cv2.arcLength(c, True)
    
    # C. Circularidad (Forma)
    # Fórmula: 4 * pi * Area / (Perímetro^2)
    # Cerca de 1.0 = Círculo (Naranja), Menos de 0.6 = Alargado (Pera/Piña)
    if perimetro == 0:
        circularidad = 0
    else:
        circularidad = (4 * np.pi * area) / (perimetro ** 2)
        
    # D. Color Promedio (Solo de la parte de la fruta, usando la máscara)
    color_medio = cv2.mean(hsv, mask=mask)
    hue_promedio = color_medio[0]  # El matiz (El color en sí)
    
    # Devolvemos un vector de características: [Color(H), Forma(Circ), Tamaño(Area)]
    features = [hue_promedio, circularidad, area]
    return features

# --- FASE 2: ENTRENAMIENTO DE LA IA (Simulación) ---

def entrenar_modelo():
    # Aquí simulamos datos que ya habríamos extraído de muchas fotos.
    # Formato: [Hue (0-180), Circularidad (0-1), Área (pixeles)]
    
    X_train = [
        [5, 0.85, 20000],   # Manzana (Roja, muy redonda)
        [175, 0.88, 19500], # Manzana (Roja extremo, redonda)
        [30, 0.60, 18000],  # Pera (Amarilla/Verde, alargada)
        [35, 0.55, 18500],  # Pera
        [15, 0.95, 15000],  # Naranja (Naranja, círculo casi perfecto)
        [20, 0.25, 25000],  # Piña (Amarilla/Café, textura rugosa hace perímetro largo -> circularidad baja)
        [110, 0.80, 8000],  # Ciruela (Azul/Morada, pequeña)
    ]
    
    y_train = ["Manzana", "Manzana", "Pera", "Pera", "Naranja", "Piña", "Ciruela"]
    
    # Usamos un Árbol de Decisión
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

# --- PROGRAMA PRINCIPAL ---

# 1. Entrenamos la IA
mi_ia = entrenar_modelo()
print("IA Entrenada y lista para separar frutas 🍎🍐🍊")

# 2. Simular el análisis de una imagen nueva (Pon aquí la ruta de tu foto)
imagen_prueba = "oran.jpg" 
datos_leidos_camara = extraer_caracteristicas(imagen_prueba)

# Como no tengo tu foto, voy a inventar unos datos que leería la cámara:
# datos_leidos_camara = [16, 0.92, 15500] # Color naranja, muy redonda, tamaño medio

print(f"\nCaracterísticas extraídas de la cámara:")
print(f"Color (Hue): {datos_leidos_camara[0]}")
print(f"Circularidad: {datos_leidos_camara[1]:.2f}")
print(f"Tamaño: {datos_leidos_camara[2]}")

# 3. Predicción
fruta_predicha = mi_ia.predict([datos_leidos_camara])
print(f"\n>> Resultado: La máquina clasifica esto como: {fruta_predicha[0].upper()}")