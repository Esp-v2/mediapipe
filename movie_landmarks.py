# image file names to files in list format
files=[]
for name in sorted(glob.glob(base_path+'/images/*.png')):
    files.append(name)

# Read images with OpenCV.
images = {name: cv2.imread(name) for name in files}

for name, image in images.items():
    # Convert the BGR image to RGB and process it with MediaPipe Pose.
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 顔→右手→左手→姿勢の順にcsv保存
    # mediapipeで推定不可の画像に対して、3次元カラムとNaN要素をlistに追加
    landmark_csv = []
    col_label = []
    # 顔
    if not results.multi_face_landmarks:
        for xyz in range(468):
            col_label.append(str(xyz) + "_x")
            col_label.append(str(xyz) + "_y")
            col_label.append(str(xyz) + "_z")
            for _ in range(3):
                landmark_csv.append(np.nan)
    # 右手
    if not results.right_hand_landmarks:
        for xyz in range(21):
            col_label.append(str(xyz) + "_x")
            col_label.append(str(xyz) + "_y")
            col_label.append(str(xyz) + "_z")
            for _ in range(3):
                facemesh_csv.append(np.nan)
    # 左手
    if not results.left_hand_landmarks:
        for xyz in range(21):
            col_label.append(str(xyz) + "_x")
            col_label.append(str(xyz) + "_y")
            col_label.append(str(xyz) + "_z")
            for _ in range(3):
                facemesh_csv.append(np.nan)
    # 姿勢
    if not results.pose_landmarks:
        for xyz in range(33):
            col_label.append(str(xyz) + "_x")
            col_label.append(str(xyz) + "_y")
            col_label.append(str(xyz) + "_z")
            for _ in range(3):
                facemesh_csv.append(np.nan)
    
                
    # Draw pose landmarks.
    annotated_image = image.copy()
    #　左手
    mp_drawing.draw_landmarks(annotated_image, 
                            results.left_hand_landmarks, 
                            mp_holistic.HAND_CONNECTIONS)
    # 右手
    mp_drawing.draw_landmarks(annotated_image,
                            results.right_hand_landmarks, 
                            mp_holistic.HAND_CONNECTIONS)
    # 顔
    mp_drawing.draw_landmarks(image=annotated_image, 
                            landmark_list=results.face_landmarks, 
                            connections=mp_holistic.FACEMESH_TESSELATION,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)
    # 姿勢
    mp_drawing.draw_landmarks(image=annotated_image, 
                            landmark_list=results.pose_landmarks, 
                            connections=mp_holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)

    cv2.imwrite(name, annotated_image)