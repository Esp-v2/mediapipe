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
    image = cv2.imread('/Users/shu/Desktop/kanae.jpg')
    results = holistic.process(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # ランドマークの座標dataframeとarray_imageを取得
    df_xyz, landmark_image = landmark(image)

    # ランドマーク記載画像を整形
    landmark_image = cv2.cvtColor(
        landmark_image, cv2.COLOR_BGR2RGB)  # BGRtoRGB
    landmark_image = Image.fromarray(landmark_image.astype(np.uint8))
    landmark_image.show()

    height, width, channels = image.shape[:3]
    # ランドマークの色情報を取得
    df_rgb = color(image, df_xyz, height, width)

    # xyzとrgb結合
    df_xyz_rgb = pd.concat([df_xyz, df_rgb], axis=1)
    df_xyz_rgb.to_csv('./xyzrgb.csv', header=False, index=False)


# 顔のランドマークの色情報を抽出する
def color(image, xyz, height, width):
    label = ['r', 'g', 'b']
    data = []
    for index in range(len(xyz)):
        b = image[int(xyz.iloc[index, 0]*height),
                  int(xyz.iloc[index, 1]*width), 0]
        g = image[int(xyz.iloc[index, 0]*height),
                  int(xyz.iloc[index, 1]*width), 1]
        r = image[int(xyz.iloc[index, 0]*height),
                  int(xyz.iloc[index, 1]*width), 2]
        data.append([r, g, b])

    df = pd.DataFrame(data, columns=label)
    return df


# 顔のランドマーク
def face(results, annotated_image):
    label = ["x", "y", "z"]
    data = []
    if results.face_landmarks:
        # ランドマークを描画する
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

        for landmark in results.face_landmarks.landmark:
            data.append([landmark.x, landmark.y, landmark.z])

    else:  # 検出されなかったら欠損値nanを登録する
        data.append([np.nan, np.nan, np.nan])

    df = pd.DataFrame(data, columns=label)
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
