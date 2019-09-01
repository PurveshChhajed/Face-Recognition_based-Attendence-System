import cv2
import numpy as numpy
import os, time
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=256)

FACE_DIR = "data_new/"


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def main():
    create_folder(FACE_DIR)
    while True:
        face_id = input("Enter id for face: ")
        try:
            face_folder = FACE_DIR + str(face_id)
            break
        except:
            print("Invalid input. id must be int")
            continue

    img_no = 1
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 8000)   #7680, 4320
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 5000)
    total_imgs = 150
    while True:
        ret, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray)
        if len(faces) == 1:
            face = faces[0]
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_img = img_gray[y-50:y + h+100, x-50:x + w+100]
            face_aligned = face_aligner.align(img, img_gray, face)

            face_img = face_aligned
            img_path = face_folder+"."+ str(img_no) + ".jpg"
            cv2.imwrite(img_path, face_img)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255,255, 0), 3)
            cv2.imshow('frame', img)
            #cv2.imshow("aligned", face_img)
            img_no += 1
        #cv2.imshow("Saving", img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
        elif img_no>=total_imgs+1:
            print("Successfully Captured")
            break
    cap.release()
    cv2.destroyAllWindows()

main()
