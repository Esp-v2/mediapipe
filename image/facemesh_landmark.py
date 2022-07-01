import glob
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def mediapipe_static(dir_input, dir_output_image, dir_output_csv, name):
    # 例）dir_input : "/content/drive/MyDrive/Lesson/Facial Experience/HARUKI/mediapipe/data/input/*"
    files = glob.glob(dir_input)
    drawing_spec = mp_drawing.DrawingSpec(
        thickness=1, circle_radius=1, color=(0, 255, 0))

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

        for idx, file in tqdm(enumerate(files, start=0)):
            # 画像の読み込み
            image = cv2.imread(file)
            # 画像の色の変換
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # mediapipeで推定不可の画像に対して、3次元カラムとNaN要素をlistに追加
            if not results.multi_face_landmarks:
                facemesh_csv = []
                col_label = []
                for xyz in range(468):
                    col_label.append(str(xyz) + "_x")
                    col_label.append(str(xyz) + "_y")
                    col_label.append(str(xyz) + "_z")
                    for _ in range(3):
                        facemesh_csv.append(np.nan)

            # mediapipeで推定可能な画像に対して、3次元カラムと推定3次元座標をlistに追加
            else:
                annotated_image = image.copy()

                for face_landmarks in results.multi_face_landmarks:
                    facemesh_csv = []
                    col_label = []

                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)

                    for xyz, landmark in enumerate(face_landmarks.landmark):
                        col_label.append(str(xyz) + "_x")
                        col_label.append(str(xyz) + "_y")
                        col_label.append(str(xyz) + "_z")
                        facemesh_csv.append(landmark.x)
                        facemesh_csv.append(landmark.y)
                        facemesh_csv.append(landmark.z)

            # 1枚目の画像をDataFrame構造で保存
            if idx == 0:
                data = pd.DataFrame([facemesh_csv], columns=col_label)
            # 1枚目と2枚目以降のDataFrameを縦に結合
            else:
                data1 = pd.DataFrame([facemesh_csv], columns=col_label)
                data = pd.concat([data, data1], ignore_index=True)

            try:
                cv2.imwrite(dir_output_image + str(idx) +
                            '.png', annotated_image)
            except UnboundLocalError:
                pass
            time.sleep(1)

    data.to_csv(dir_output_csv + name + '.csv')
    return data


mediapipe_static(r"C:\FaceExperience\valid\neutral\*",
                 r"C:\mediapipe\image/",
                 r"C:\mediapipe\csv\valid/",
                 "neutral")
