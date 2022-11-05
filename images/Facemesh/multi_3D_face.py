import os
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import mediapipe as mp


# 初期設定
mp_face_mesh = mp.solutions.face_mesh
facemesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,  # 468 or 478
    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=1,  color=(0, 255, 0))
mark_drawing_spec = mp_drawing.DrawingSpec(
    thickness=1,  circle_radius=1, color=(0, 0, 255))


def main():
    if os.name == 'nt':  # windows
        files = glob.glob(
            "C:\\Users\\proje\\Desktop\\DataSets\\CelebA\\test\\*")
    else:  # mac
        files = glob.glob(
            "/Users/shu/Desktop/DataSets/test/*")

    multi_xyz_rgb = pd.DataFrame(index=[], columns=[])
    for fname in files:
        image = cv2.imread(fname)
        results = facemesh.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            basename = os.path.basename(fname)
            # ランドマークの座標dataframeとarray_imageを取得
            df_xyz, annotated_image = landFace(basename, image, results)

            # """
            # ランドマーク記載画像を整形
            annotated_image = cv2.cvtColor(
                annotated_image, cv2.COLOR_BGR2RGB)  # BGRtoRGB
            annotated_image = Image.fromarray(annotated_image.astype(np.uint8))
            # annotated_image.show()
            annotated_image.save("./annotated_image.jpg")
            # """

            height, width, channels = image.shape[:3]
            # ランドマークの色情報を取得
            df_rgb = color(basename, image, df_xyz, height, width)

            # xyzとrgb結合
            xyz_rgb = pd.concat([df_xyz, df_rgb], axis=1)
            # 複数枚のxyz-rgb
            multi_xyz_rgb = pd.concat([multi_xyz_rgb, xyz_rgb], axis=0)
            print(multi_xyz_rgb)

    multi_xyz_rgb.to_csv('./xyzrgb.csv', header=True, index=False)


# 顔のランドマークの色情報を抽出する
"""
image:cv2.imreadで読み込んだ画像
xyz:ランドマークの座標
height:画像の高さサイズ
width:画像の幅サイズ
"""


def color(basename, image, xyz, height, width):
    label = []
    data = []
    length = int(len(xyz.columns)/3)
    for _ in range(length):
        index = _*3
        x = xyz.iloc[0, index]
        y = xyz.iloc[0, index+1]
        z = xyz.iloc[0, index+2]

        if pd.isna(x) or pd.isna(y) or pd.isna(z):
            b = np.nan
            g = np.nan
            r = np.nan
        else:
            if x > 1:
                x = 1
            x = int(x*(width-1))

            if y > 1:
                y = 1
            y = int(y*(height-1))

            if x > width-1:
                x = width-1
            if y > height-1:
                y = height-1

            b = int(image[y, x, 0])
            g = int(image[y, x, 1])
            r = int(image[y, x, 2])
        data.extend([r, g, b])
        label.extend([str(_)+"_r", str(_)+"_g", str(_)+"_b"])

    df = pd.DataFrame([data], columns=label, index=[basename])
    return df


# 顔のランドマーク
def landFace(basename, image, results):
    label = []
    data = []

    for face_landmarks in results.multi_face_landmarks:
        if face_landmarks:
            annotated_image = image.copy()
            # ランドマークを描画する
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mark_drawing_spec,
                connection_drawing_spec=mesh_drawing_spec)

            for index, landmark in enumerate(face_landmarks.landmark):
                data.extend([landmark.x, landmark.y, landmark.z])
                label.extend(
                    [str(index)+"_x", str(index)+"_y", str(index)+"_z"])

        else:  # 検出されなかったら欠損値nanを登録する
            print("検出なし！")
            for i in range(478*3):
                data.append(np.nan)
            [label.extend([str(i)+"_x", str(i)+"_y", str(i)+"_z"])
                for i in range(478)]
        df = pd.DataFrame([data], columns=label, index=[basename])
    return df, annotated_image


if __name__ == "__main__":
    main()
