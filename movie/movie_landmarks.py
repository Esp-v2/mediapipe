import mediapipe as mp
import os
import shutil
import cv2
#from google.colab.patches import cv2_imshow
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
import time

# 初期設定
mp_holistic = mp.solutions.holistic
# Initialize MediaPipe Holistic.
holistic = mp_holistic.Holistic(
    static_image_mode=True, min_detection_confidence=0.5)
# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mp_face_mesh = mp.solutions.face_mesh

# サンプル動画を読み込み静止画に分解し、imagesフォルダーに保管
base_path = r"C:\Users\proje\Desktop\Sign_Language\DataSets"
video_path = base_path+r"\train\sign_1\001_001_002.mp4"

# base_pathに既にimagesフォルダーがあれば削除
if os.path.isdir(base_path+'/images'):
    shutil.rmtree(base_path+'/images')
os.makedirs(base_path+'/images', exist_ok=True)


def video_2_images(video_file=video_path,
                   image_dir=base_path+'/images/',
                   image_file='%s.png'):
    # Initial setting
    i = 0
    interval = 3
    length = 300

    cap = cv2.VideoCapture(video_file)
    while(cap.isOpened()):
        flag, frame = cap.read()
        if flag == False:
            break
        if i == length*interval:
            break
        if i % interval == 0:
            cv2.imwrite(image_dir+image_file %
                        str(int(i/interval)).zfill(6), frame)
        i += 1
    cap.release()


video_2_images()


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
                #cv2.imwrite(dir_output_image + str(idx) + file +
                            '.png', annotated_image)
            except UnboundLocalError:
                pass
            time.sleep(1)

    data.to_csv(dir_output_csv + name + '.csv')
    return data


# mediapipe_static(input, output_image, output_csv, name):
mediapipe_static(base_path+r"\images\*",
                 base_path+r"\mediapipeResult\image/",
                 base_path+r"\mediapipeResult\csv/",
                 "test")
