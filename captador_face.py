import cv2
import face_recognition
import time

# Caminho do vídeo
video_path = "video.mp4"  # substitua pelo seu vídeo
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Erro ao abrir o vídeo. Verifique o caminho!")
    exit()

frame_rate = video.get(cv2.CAP_PROP_FPS)
frame_time = 1 / frame_rate if frame_rate > 0 else 0.03

# Dicionário para armazenar tempo de atenção por rosto
rostos_tempo_olhando = {}

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]  # Converte BGR -> RGB

    # Detecta rostos e landmarks
    faces = face_recognition.face_locations(rgb_frame)
    landmarks_list = face_recognition.face_landmarks(rgb_frame, faces)

    for i, (top, right, bottom, left) in enumerate(faces):
        # Desenha retângulos nos rostos
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Cria um ID simples para o rosto
        id_rosto = f"{i}"

        # Assume que está olhando se ambos olhos forem detectados
        olhos = landmarks_list[i].get("left_eye") and landmarks_list[i].get("right_eye")
        if olhos:
            if id_rosto not in rostos_tempo_olhando:
                rostos_tempo_olhando[id_rosto] = 0
            rostos_tempo_olhando[id_rosto] += frame_time

    cv2.imshow("Rostos no Vídeo", frame)

    # Sai com 'q'
    if cv2.waitKey(int(frame_time * 1000)) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Mostra o tempo de atenção de cada rosto
print("\nTempo de atenção olhando para a tela (segundos):")
for rosto, tempo in rostos_tempo_olhando.items():
    print(f"Rosto {rosto}: {tempo:.2f}s")
