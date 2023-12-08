import face_recognition as fr
import os
import cv2
import numpy as np

def get_encoded_faces():
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.lower().endswith((".jpg", ".png")):
                face = fr.load_image_file(os.path.join("faces", f))
                encoding = fr.face_encodings(face)[0]
                encoded[os.path.splitext(f)[0]] = encoding

    return encoded

def read_face_data(name):
    file_path = os.path.join("faces", f"{name}.txt")
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
        return contents
    except FileNotFoundError:
        return f"No additional information found for {name}."

def classify_face(frame, faces_encoded, known_face_names):
    face_locations = fr.face_locations(frame)
    unknown_face_encodings = fr.face_encodings(frame, face_locations)

    matching_profiles_found = False

    for (top, right, bottom, left), face_encoding in zip(face_locations, unknown_face_encodings):
        matches = fr.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        if any(matches):
            matching_profiles_found = True
            face_distances = fr.face_distance(faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_data = read_face_data(name)
                print(f"Additional information for {name}:\n{face_data}")
                cv2.putText(frame, f"{name}\n{face_data}", (left - 20, bottom + 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.rectangle(frame, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)
        cv2.putText(frame, name, (left - 20, bottom + 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    if not matching_profiles_found:
        print("No matching profiles found in the database")

    return frame

def main():
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        frame = classify_face(frame, faces_encoded, known_face_names)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()