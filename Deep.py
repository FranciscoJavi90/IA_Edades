from deepface import DeepFace
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

# Declaramos la detección de rostros
detros = mp.solutions.face_detection
rostros = detros.FaceDetection(min_detection_confidence=0.8, model_selection=0)
# Dibujo
dibujorostro = mp.solutions.drawing_utils

# Realizamos VideoCaptura

cap = cv2.VideoCapture(0)

# Cargar las imágenes estáticas usando Pillow
ruta_feliz = "C:\\Users\\Legion 5\\Documents\\Detector de Emociones\\feliz.png"
ruta_disgustado = "C:\\Users\\Legion 5\\Documents\\Detector de Emociones\\disgustado.png"
ruta_enojo = "C:\\Users\\Legion 5\\Documents\\Detector de Emociones\\enojo.png"
ruta_seria = "C:\\Users\\Legion 5\\Documents\\Detector de Emociones\\seria.png"
ruta_sorprendido = "C:\\Users\\Legion 5\\Documents\\Detector de Emociones\\sorprendido.png"
ruta_triste = "C:\\Users\\Legion 5\\Documents\\Detector de Emociones\\triste.png"
ruta_miedoso= "C:\\Users\\Legion 5\\Documents\\Detector de Emociones\\miedoso.png"

# Convertir las imágenes a arrays de numpy con transparencia
imagenes = {
    'feliz': np.array(Image.open(ruta_feliz).convert('RGBA')),
    'disgustado': np.array(Image.open(ruta_disgustado).convert('RGBA')),
    'enojado': np.array(Image.open(ruta_enojo).convert('RGBA')),
    'serio': np.array(Image.open(ruta_seria).convert('RGBA')),
    'sorprendido': np.array(Image.open(ruta_sorprendido).convert('RGBA')),
    'triste': np.array(Image.open(ruta_triste).convert('RGBA')),
    'miedoso': np.array(Image.open(ruta_miedoso).convert('RGBA'))
}

# Empezamos
while True:
    # Leemos los fotogramas
    ret, frame = cap.read()

    # Ajustar tamaño de la ventana de la cámara
    frame = cv2.resize(frame, (1280, 720))

    # Corrección de color
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesamos
    resrostros = rostros.process(rgb)

    # Detección
    if resrostros.detections is not None:
        for rostro in resrostros.detections:
            # Extraemos información de ubicación
            al, an, c = frame.shape
            box = rostro.location_data.relative_bounding_box
            xi, yi, w, h = int(box.xmin * an), int(box.ymin * al), int(box.width * an), int(box.height * al)
            xf, yf = xi + w, yi + h

            # Dibujamos cuadrado para el rostro lo reconozca
            # cv2.rectangle(frame, (xi, yi), (xf, yf), (255, 255, 0), 1)

            # Información
            info = DeepFace.analyze(rgb, actions=['age', 'emotion','race'], enforce_detection=False)

            if info:
                info = info[0]  # Accede al primer elemento de la lista
                edad = info['age']
                emociones = info['dominant_emotion']
                race = info['race']

                # Emociones
                if emociones == 'angry':
                    emociones = 'enojado'
                elif emociones == 'disgust':
                    emociones = 'disgustado'
                elif emociones == 'fear':
                    emociones = 'miedoso'
                elif emociones == 'happy':
                    emociones = 'feliz'
                elif emociones == 'sad':
                    emociones = 'triste'
                elif emociones == 'surprise':
                    emociones = 'sorprendido'
                elif emociones == 'neutral':
                    emociones = 'serio'
                    
                     # Race
                if race == 'asian':
                    race = 'asiatico'
                if race == 'indian':
                    race = 'indio'
                if race == 'black':
                    race = 'negro'
                if race == 'white':
                    race = 'blanco'
                if race == 'middle eastern':
                    race = 'oriente medio'
                if race == 'latino hispanic':
                    race = 'latino'

                # Establecemos los estilos
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (0, 0, 0)  # Negro
                thickness = 2
                font_scale = 1

                # Mostramos info con estilos
                cv2.putText(frame, "Edad: " + str(edad), (75, 90), font, font_scale, color, thickness)
                #cv2.putText(frame, "Emociones: " + str(emociones), (75, 135), font, font_scale, color, thickness)
                #cv2.putText(frame, "Raza: " +str(race), (75, 180), font, font_scale, color, thickness)

                # Mostrar imagen en la esquina superior derecha dependiendo de la emoción
                if emociones in imagenes:
                    imagen_emocion = imagenes[emociones]

                    # Redimensionar la imagen a un tamaño más pequeño
                    alto_imagen, ancho_imagen = imagen_emocion.shape[:2]
                    escala = 0.2  # Escalar la imagen al 20% de su tamaño original
                    nuevo_ancho = int(ancho_imagen * escala)
                    nuevo_alto = int(alto_imagen * escala)
                    imagen_emocion_resized = cv2.resize(imagen_emocion, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)

                    # Posición de la esquina superior derecha
                    x_offset = frame.shape[1] - nuevo_ancho - 10  # 10 píxeles desde el borde derecho
                    y_offset = 10  # 10 píxeles desde el borde superior

                    # Superponer la imagen en la esquina superior derecha con transparencia
                    for i in range(nuevo_alto):
                        for j in range(nuevo_ancho):
                            if imagen_emocion_resized[i, j][3] != 0:  # Verificar canal alfa
                                frame[y_offset + i, x_offset + j] = (
                                    imagen_emocion_resized[i, j][:3] * (imagen_emocion_resized[i, j][3] / 255.0) +
                                    frame[y_offset + i, x_offset + j] * (1.0 - imagen_emocion_resized[i, j][3] / 255.0)
                                ).astype(np.uint8)

    # Mostramos los fotogramas
    cv2.imshow("RED NEURONAL", frame)

    # Leemos el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

cv2.destroyAllWindows()
cap.release()
