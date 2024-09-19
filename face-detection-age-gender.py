import cv2
import numpy as np

def detect_faces(frame, face_net):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    locs = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append(frame[startY:endY, startX:endX])
            locs.append((startX, startY, endX, endY))
    
    return faces, locs

def predict_age(face, age_net):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    preds = age_net.forward()
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    return age_list[preds[0].argmax()]

def predict_gender(face, gender_net):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    gender_net.setInput(blob)
    preds = gender_net.forward()
    return "Male" if preds[0][0] > preds[0][1] else "Female"  # Reversed the condition

# Load pre-trained models
face_net = cv2.dnn.readNet("res10_300x300_ssd_iter_140000.caffemodel", "deploy.prototxt.txt")
age_net = cv2.dnn.readNet("age_net.caffemodel", "deploy_age2.prototxt")
gender_net = cv2.dnn.readNet("gender_net.caffemodel", "deploy_gender2.prototxt")

# Open video file
video = cv2.VideoCapture("input_video.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    faces, locs = detect_faces(frame, face_net)
    
    for face, (startX, startY, endX, endY) in zip(faces, locs):
        age = predict_age(face, age_net)
        gender = predict_gender(face, gender_net)
        
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        label = f"{gender}, {age}"
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()