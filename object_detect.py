from load_onnx import YOLOv8
from utils import class_names
import mediapipe as mp
import cv2

model_path1 = "yolov8n.onnx"
model1 = YOLOv8(model_path1, conf_thres=0.2, iou_thres=0.1)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.3, min_tracking_confidence=0.2)


def make_3_landmark_img(color_mode, image, cimage, landmark1, landmark2, landmark3, cx, cy, sp):
    if int(landmark1.x) == 0 and int(landmark1.y) == 0 and int(landmark2.x) == 0 and int(landmark2.y) == 0 and int(landmark3.x) == 0 and int(landmark3.y) == 0:
        cir_radius = 4
        cir_color = (255, 0, 0)
        cir_thickness = 2
        if color_mode == 1:
            line_color = (0, 0, 255)
        elif color_mode == 2:
            line_color = (0, 255, 0)
        line_thickness = 2

        image_width = cimage.shape[1]
        image_height = cimage.shape[0]

        coo1 = (int((landmark1.x / sp * image_width) + cx), int((landmark1.y / sp * image_height) + cy))
        image = cv2.circle(image, coo1, cir_radius, cir_color, cir_thickness)

        coo2 = (int((landmark2.x / sp * image_width) + cx), int((landmark2.y / sp * image_height) + cy))
        image = cv2.circle(image, coo2, cir_radius, cir_color, cir_thickness)

        coo3 = (int((landmark3.x / sp * image_width) + cx), int((landmark3.y / sp * image_height) + cy))
        image = cv2.circle(image, coo3, cir_radius, cir_color, cir_thickness)

        image = cv2.line(image, coo1, coo2, line_color, line_thickness)
        image = cv2.line(image, coo2, coo3, line_color, line_thickness)

    return image


def detectImage(frame):
    display_image = frame
    boxes, scores, class_ids = model1(frame)
    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        label = class_names[class_id]
        if label == "person":
            buffer_size = 50

            x1 = int(x1 - buffer_size)
            x1 = 0 if x1 < 0 else int(x1)
            y1 = int(y1 - buffer_size)
            y1 = 0 if y1 < 0 else int(y1)
            x2 = int(x2 + buffer_size)
            x2 = int(width) if x2 > width else int(x2)
            y2 = int(y2 + buffer_size)
            y2 = int(height) if y2 > height else int(y2)
            
            cropped_image = frame[y1:y2, x1:x2]
            if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                pass
            elif Exception:
                scale_percent = 5
                img_width = int(cropped_image.shape[1] * scale_percent)
                img_height = int(cropped_image.shape[0] * scale_percent)
                dim = (img_width, img_height)
                cropped_image = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)
                # display_image = cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                results = pose.process(cropped_image)
                if results.pose_landmarks is not None:
                    right_shoulder_landmark = results.pose_landmarks.landmark[12]
                    right_elbow_landmark = results.pose_landmarks.landmark[14]
                    right_wrist_landmark = results.pose_landmarks.landmark[16]

                    right_hip_landmark = results.pose_landmarks.landmark[24]
                    right_knee_landmark = results.pose_landmarks.landmark[26]
                    right_ankle_landmark = results.pose_landmarks.landmark[28]

                    left_shoulder_landmark = results.pose_landmarks.landmark[11]
                    left_elbow_landmark = results.pose_landmarks.landmark[13]
                    left_wrist_landmark = results.pose_landmarks.landmark[15]

                    left_hip_landmark = results.pose_landmarks.landmark[23]
                    left_knee_landmark = results.pose_landmarks.landmark[25]
                    left_ankle_landmark = results.pose_landmarks.landmark[27]

                    display_image = make_3_landmark_img(1, display_image, cropped_image, right_shoulder_landmark,
                                                        right_elbow_landmark, right_wrist_landmark, x1, y1,
                                                        scale_percent)
                    display_image = make_3_landmark_img(1, display_image, cropped_image, left_shoulder_landmark,
                                                        left_elbow_landmark, left_wrist_landmark, x1, y1,
                                                        scale_percent)
                    display_image = make_3_landmark_img(2, display_image, cropped_image, right_hip_landmark,
                                                        right_knee_landmark, right_ankle_landmark, x1, y1,
                                                        scale_percent)
                    display_image = make_3_landmark_img(2, display_image, cropped_image, left_hip_landmark,
                                                        left_knee_landmark, left_ankle_landmark, x1, y1,
                                                        scale_percent)
    return display_image
