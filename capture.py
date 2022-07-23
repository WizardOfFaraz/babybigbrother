#!/usr/local/bin/python
import cv2
import face_recognition
import os
import numpy as np
import pychromecast


vid = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
chromecasts, browser = pychromecast.get_listed_chromecasts(friendly_names=["SHIELD"])
cast = chromecasts[0]
cast.wait()
mc = cast.media_controller

image1 = face_recognition.load_image_file(os.path.abspath("face.jpg"))
image1_face_encoding = face_recognition.face_encodings(image1)[0]


known_face_encodings = [
    image1_face_encoding,
]
known_face_names = [
    "Isaac"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
found_issac = False

while(True):
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    print(rgb_small_frame)
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
               
            face_names.append(name)
    if 'Isaac' in face_names and mc.status.player_state != 'PAUSED':
        mc.pause()
    elif 'Isaac' not in face_names and mc.status.player_state == 'PAUSED':
        mc.play()
                   
    process_this_frame = not process_this_frame
    print ("Face detected -- {}".format(face_names))
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  #  for (x, y, w, h) in faces:
  #    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
  #
  #  cv2.imshow('frame', frame)
  #
  #  if cv2.waitKey(1) & 0xFF == ord('q'):
  #    break
  #
vid.release()
cv2.destroyAllWindows()
