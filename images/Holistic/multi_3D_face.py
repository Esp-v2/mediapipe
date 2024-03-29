import os
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import mediapipe as mp

# 初期設定
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def main():
    if os.name == 'nt':  # windows
        files = glob.glob(
            "C:\\Users\\proje\\Desktop\\DataSets\\GENKI-R2009a\\datasets\\smile\\*")
    else:  # mac
        files = glob.glob(
            "/Users/shu/Desktop/DataSets/GENKI-R2009a/datasets/smile/*")

    multi_xyz_rgb = pd.DataFrame(index=[], columns=[])
    for fname in files:
        print(fname)
        image = cv2.imread(fname)
        results = holistic.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # ランドマークの座標dataframeとarray_imageを取得
        df_xyz, landmark_image = landmark(image)

        """
        # ランドマーク記載画像を整形
        landmark_image = cv2.cvtColor(
            landmark_image, cv2.COLOR_BGR2RGB)  # BGRtoRGB
        landmark_image = Image.fromarray(landmark_image.astype(np.uint8))
        landmark_image.save("landmark.jpg")
        """

        height, width, channels = image.shape[:3]
        # ランドマークの色情報を取得
        df_rgb = color(image, df_xyz, height, width)

        # xyzとrgb結合
        xyz_rgb = pd.concat([df_xyz, df_rgb], axis=1)
        # 複数枚のxyz-rgb
        multi_xyz_rgb = pd.concat([multi_xyz_rgb, xyz_rgb], axis=0)
        print(multi_xyz_rgb)

    multi_xyz_rgb.to_csv('./xyzrgb.csv', header=True, index=True)


# 顔のランドマークの色情報を抽出する
"""
image:cv2.imreadで読み込んだ画像
xyz:ランドマークの座標
height:画像の高さサイズ
width:画像の幅サイズ
"""


def color(image, xyz, height, width):
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

    df = pd.DataFrame([data], columns=label)
    return df


# 顔のランドマーク
def face(results, annotated_image):
    label = []
    data = []
    if results.face_landmarks:
        # ランドマークを描画する
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

        for index, landmark in enumerate(results.face_landmarks.landmark):
            data.extend([landmark.x, landmark.y, landmark.z])
            label.extend([str(index)+"_x", str(index)+"_y", str(index)+"_z"])

    else:  # 検出されなかったら欠損値nanを登録する
        print("検出なし！")
        for i in range(468*3):
            data.append(np.nan)
        [label.extend([str(i)+"_x", str(i)+"_y", str(i)+"_z"])
         for i in range(468)]
    df = pd.DataFrame([data], columns=label)
    return df


# imageに対してmediapipeでランドマークを表示、出力する
def landmark(image):
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()
    # ランドマーク取得
    df_xyz = face(results, annotated_image)
    return df_xyz, annotated_image


if __name__ == "__main__":
    main()
