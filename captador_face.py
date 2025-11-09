import cv2
import numpy as np
import face_recognition
from scipy.spatial import distance
import pandas as pd

VIDEO_INDEX = 0
LIMIAR_EAR_ABERTO = 0.28  
PROPORCAO_PISCADA = 0.7   
DIST_MAX_RASTREIO = 50    

video = cv2.VideoCapture(VIDEO_INDEX)

if not video.isOpened():
    print("Câmera não encontrada.")

frame_rate = video.get(cv2.CAP_PROP_FPS) or 30
frame_time = 1 / frame_rate

rostos_tempo_olhando = {}       
rostos_centros = {}            
next_id = 0                   

cv2.namedWindow("Detecção de Olhar", cv2.WINDOW_NORMAL)

def calcular_ear(olho):
    A = np.linalg.norm(olho[1] - olho[5])
    B = np.linalg.norm(olho[2] - olho[4])
    C = np.linalg.norm(olho[0] - olho[3])
    return (A + B) / (2.0 * C)

def associar_id(novo_centro, centros_existentes):
    menor_dist = DIST_MAX_RASTREIO
    id_associado = None
    for id_r, centro in centros_existentes.items():
        dist = np.linalg.norm(np.array(novo_centro) - np.array(centro))
        if dist < menor_dist:
            menor_dist = dist
            id_associado = id_r
    return id_associado

try:
    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_display = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

        face_locations = face_recognition.face_locations(small_frame)
        face_landmarks_list = face_recognition.face_landmarks(small_frame, face_locations)

        novos_centros = {}

        for i, (top, right, bottom, left) in enumerate(face_locations):
            top, right, bottom, left = top*2, right*2, bottom*2, left*2
            centro_rosto = ((left + right)//2, (top + bottom)//2)

            id_rosto = associar_id(centro_rosto, rostos_centros)
            if id_rosto is None:
                id_rosto = f"Pessoa_{next_id}"
                next_id += 1

            novos_centros[id_rosto] = centro_rosto

            if i < len(face_landmarks_list):
                landmarks = face_landmarks_list[i]
                olhos_abertos = 0
                total_olhos = 0
                ears = []

                for olho_nome in ['left_eye', 'right_eye']:
                    if olho_nome in landmarks:
                        total_olhos += 1
                        olho_points = np.array(landmarks[olho_nome])*2
                        ear = calcular_ear(olho_points)
                        ears.append(ear)

                        if ear > LIMIAR_EAR_ABERTO:
                            olhos_abertos += 1
                            cor_olho = (0, 255, 0)
                        else:
                            cor_olho = (0, 0, 255)

                        cv2.polylines(frame_display, [olho_points.astype(int)], True, cor_olho, 1)
                        for p in olho_points:
                            cv2.circle(frame_display, tuple(p), 2, cor_olho, -1)

                proporcao_olhos_abertos = olhos_abertos / total_olhos if total_olhos > 0 else 0
                olhando_tela = proporcao_olhos_abertos >= PROPORCAO_PISCADA

                if olhando_tela:
                    rostos_tempo_olhando[id_rosto] = rostos_tempo_olhando.get(id_rosto, 0) + frame_time
                    cor_rosto = (0, 255, 0)
                    estado_rosto = "OLHANDO P/TELA"
                else:
                    cor_rosto = (0, 0, 255)
                    estado_rosto = "DISTRAIDO"

                cv2.rectangle(frame_display, (left, top), (right, bottom), cor_rosto, 2)
                info_texto = f"{id_rosto}: {estado_rosto} | Olhos {olhos_abertos}/{total_olhos} | EAR {[f'{e:.2f}' for e in ears]}"
                cv2.putText(frame_display, info_texto, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_rosto, 1)

        rostos_centros = novos_centros

        info = [
            f"EAR Limiar: {LIMIAR_EAR_ABERTO:.2f}",
            f"Pessoas detectadas: {len(face_locations)}",
            "Controles: +/- EAR, Q sai"
        ]
        for j, txt in enumerate(info):
            cv2.putText(frame_display, txt, (10, 30 + j*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Detecção de Olhar", frame_display)

        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        #elif key == ord('+'):
        #    LIMIAR_EAR_ABERTO += 0.01
        #elif key == ord('-'):
        #    LIMIAR_EAR_ABERTO = max(0.15, LIMIAR_EAR_ABERTO - 0.01)

except KeyboardInterrupt:
    print("Programa interrompido pelo usuário")

finally:
    video.release()
    cv2.destroyAllWindows()

    if rostos_tempo_olhando:
        df = pd.DataFrame([
            {"Pessoa": k, "Tempo olhando (s)": round(v, 2)}
            for k, v in rostos_tempo_olhando.items()
        ])

        csv_path = "resultados_faces.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print("\n=== Resultados Salvos ===")
        print(df)

    else:
        print("Nenhum rosto detectado — nada a salvar.")