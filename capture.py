#!/usr/local/bin/python
from absl import app 
from absl import flags

from tabulate import tabulate
import cv2
import face_recognition
import os
import numpy as np
import pychromecast
import time

FLAGS = flags.FLAGS

flags.DEFINE_string('chromecast_name', None, 'The Chromecast friendly name.')
flags.DEFINE_integer('video_capture_device', 0, 'The video capture device to use.  The default device is 0, +1 for each additional device to use.')
flags.DEFINE_integer('cooldown_seconds', 0, 'The number of seconds to cooldown once the subject is no longer recognized before resuming playback.')
flags.DEFINE_boolean('list_chromecast_devices', False, 'If True, will list all available Chromecast devices that may be used.')
flags.DEFINE_boolean('display_preview_window', False, 'If True, will display a preview of the camera.')
flags.DEFINE_string('facial_image_location', None, 'The facial image file location.')
flags.DEFINE_integer('number_of_times_to_upsample', 2,
                     'The number of times to upsample the frame for processing, the higher the number, the further back the image can be.')

def main(argv):
    if FLAGS.list_chromecast_devices:
        services, browser = pychromecast.discovery.discover_chromecasts()
        devices = [] 
        for count, service in enumerate(services):
            devices.append([count, service.model_name, service.friendly_name, service.host, service.manufacturer])
        print(tabulate(devices, headers=["model_name", "friendly_name", "host", "manufacturer"]))
        pychromecast.discovery.stop_discovery(browser)
        exit(0)

    # Connect to devices
    vid = cv2.VideoCapture(FLAGS.video_capture_device)
    chromecasts, browser = pychromecast.get_listed_chromecasts(friendly_names=[FLAGS.chromecast_name])
    cast = chromecasts[0] # Only expect a single device to use
    cast.wait()
    mc = cast.media_controller

    # Facial recognition setup
    image1 = face_recognition.load_image_file(os.path.abspath(FLAGS.facial_image_location))
    image1_face_encoding = face_recognition.face_encodings(image1)[0]

    known_face_encodings = [
        image1_face_encoding,
    ]
    known_face_names = [
        "Recognized"
    ]
    face_locations = []
    face_encodings = []
    face_names = []

    start_time = 0
    set_pause = False
    found_face = False
    # Process 1 frame every 30fps for better performance
    process_this_frame = 29

    # Main loop
    while(True):
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        process_this_frame = process_this_frame + 1
        if process_this_frame % 30 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
        
            face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=FLAGS.number_of_times_to_upsample)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    start_time = time.time()
                    found_face = True
                    if mc.status.player_state != 'PAUSED':
                        set_pause = True
                        mc.pause()          
                face_names.append(name)

        if set_pause:
            if time.time() - start_time > FLAGS.cooldown_seconds:
                mc.play()
                set_pause = False
                start_time = 0
         
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

        if FLAGS.display_preview_window:
            cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(main)
