import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# 初期設定
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def main():
    image = cv2.imread('test.jpg')
    results = holistic.process(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # ランドマークのdataframeとarray_imageを取得
    landmark_df, landmark_bgr = landmark(image)

    # 画像を整形
    landmark_rgb = cv2.cvtColor(landmark_bgr, cv2.COLOR_BGR2RGB)  # BGRtoRGB
    landmark_image = Image.fromarray(landmark_rgb.astype(np.uint8))

    # 結果を出力
    print(landmark_df)
    landmark_image.save("holisticLandmark.jpg")


# 顔のランドマーク
def face(results, annotated_image, label, csv):
    if results.face_landmarks:
        # ランドマークを描画する
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

        for index, landmark in enumerate(results.face_landmarks.landmark):
            label.append("face_"+str(index) + "_x")
            label.append("face_"+str(index) + "_y")
            label.append("face_"+str(index) + "_z")
            csv.append(landmark.x)
            csv.append(landmark.y)
            csv.append(landmark.z)

    else:  # 検出されなかったら欠損値nanを登録する
        for index in range(468):
            label.append("face_"+str(index) + "_x")
            label.append("face_"+str(index) + "_y")
            label.append("face_"+str(index) + "_z")
            for _ in range(3):
                csv.append(np.nan)
    return label, csv


# 右手のランドマーク
def r_hand(results, annotated_image, label, csv):
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS)

        for index, landmark in enumerate(results.right_hand_landmarks.landmark):
            label.append("r_hand_"+str(index) + "_x")
            label.append("r_hand_"+str(index) + "_y")
            label.append("r_hand_"+str(index) + "_z")
            csv.append(landmark.x)
            csv.append(landmark.y)
            csv.append(landmark.z)

    else:
        for index in range(21):
            label.append("r_hand_"+str(index) + "_x")
            label.append("r_hand_"+str(index) + "_y")
            label.append("r_hand_"+str(index) + "_z")
            for _ in range(3):
                csv.append(np.nan)
    return label, csv


# 左手のランドマーク
def l_hand(results, annotated_image, label, csv):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS)

        for index, landmark in enumerate(results.left_hand_landmarks.landmark):
            label.append("l_hand_"+str(index) + "_x")
            label.append("l_hand_"+str(index) + "_y")
            label.append("l_hand_"+str(index) + "_z")
            csv.append(landmark.x)
            csv.append(landmark.y)
            csv.append(landmark.z)

    else:
        for index in range(21):
            label.append("l_hand_"+str(index) + "_x")
            label.append("l_hand_"+str(index) + "_y")
            label.append("l_hand_"+str(index) + "_z")
            for _ in range(3):
                csv.append(np.nan)

    return label, csv


# 姿勢のランドマーク
def pose(results, annotated_image, label, csv):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS)

        for index, landmark in enumerate(results.pose_landmarks.landmark):
            label.append("pose_"+str(index) + "_x")
            label.append("pose_"+str(index) + "_y")
            label.append("pose_"+str(index) + "_z")
            csv.append(landmark.x)
            csv.append(landmark.y)
            csv.append(landmark.z)

    else:
        for index in range(33):
            label.append("pose_"+str(index) + "_x")
            label.append("pose_"+str(index) + "_y")
            label.append("pose_"+str(index) + "_z")
            for _ in range(3):
                csv.append(np.nan)

    return label, csv


# imageに対してmediapipeでランドマークを表示、出力する
def landmark(image):
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()

    label = []
    csv = []

    # 姿勢→顔→右手→左手の順番でランドマーク取得
    label, csv = pose(results, annotated_image, label, csv)
    label, csv = face(results, annotated_image, label, csv)
    label, csv = r_hand(results, annotated_image, label, csv)
    label, csv = l_hand(results, annotated_image, label, csv)

    df = pd.DataFrame([csv], columns=label)

    return df, annotated_image


if __name__ == "__main__":
    main()