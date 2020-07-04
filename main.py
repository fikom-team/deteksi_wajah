# Load Library Face Rec
import face_recognition
import cv2
import numpy as np


# Take Webcam nya

video_capture = cv2.VideoCapture(0)

# Load Gambar yang akan di kenali
hanan_image = face_recognition.load_image_file("1.jpg")
hanan_face_encode = face_recognition.face_encodings(hanan_image)[0]
# Load Gambar Lagi Untuk Perbandingan Ke Dua
awi_image = face_recognition.load_image_file("2.jpg")
awi_face_encode = face_recognition.face_encodings(awi_image)[0]

# Buat Array untuk Wajah yang di kenali
wajah_yang_dikenali = [
    hanan_face_encode,
    awi_face_encode
]

nama_wajah_yang_di_kenali = [
    "Hanan",
    "Awi"
]

while True:

    # Ambil Tiap Frame di wabcam
    ret, frame = video_capture.read()

    # Convert Warna nya
    rgb_frame = frame[:, :, ::-1]

    # Cari Wajah yang telah di encode
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Looping tiap wajah yang tampil di frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Liat Apakah ada gambar yang sama
        matches = face_recognition.compare_faces(
            wajah_yang_dikenali, face_encoding)
        name = "Tidk di ketahui"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(
            wajah_yang_dikenali, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = nama_wajah_yang_di_kenali[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
