import cv2
import numpy as np

# Carga el diccionario de marcadores ArUco predefinido
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Inicia la cámara
cap = cv2.VideoCapture(0)

# Supón una cámara con matriz de calibración aproximada
# (para un ejemplo simple, esto sirve)
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((5, 1))  # Sin distorsión

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convierte a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta los marcadores ArUco en la imagen
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # Dibuja los bordes del marcador detectado
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estima la posición del marcador en 3D
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.05, camera_matrix, dist_coeffs
        )

        for rvec, tvec in zip(rvecs, tvecs):
            # Dibuja el eje 3D sobre el marcador
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            # Dibuja un cubo virtual
            axis = np.float32([
                [0, 0, 0], [0, 0.05, 0], [0.05, 0.05, 0],
                [0.05, 0, 0], [0, 0, -0.05], [0, 0.05, -0.05],
                [0.05, 0.05, -0.05], [0.05, 0, -0.05]
            ])

            # Proyecta los puntos 3D sobre la imagen
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

            imgpts = np.int32(imgpts).reshape(-1, 2)

            # Dibuja el cubo
            # Base del cubo
            frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)
            # Lados
            for i, j in zip(range(4), range(4, 8)):
                frame = cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 2)
            # Tapa del cubo
            frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2)

    cv2.imshow('AR Demo', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
