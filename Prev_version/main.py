import cv2
import numpy as np
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(r"c:\Advait\VS_Code\VS code 2.0\Face recognition\haarcascade_frontalface_default.xml")

#variable declaration
global count #couning the images 
global name
global temp_name

def input_name():
    name = input("Enter name ")
    return name

temp_name = input_name()

cap = cv2.VideoCapture(0)
save_dir = r"C:\Advait\VS_Code\VS code 2.0\Face recognition\Images"
known_path = os.path.join(save_dir, "known")
runtime_path = os.path.join(save_dir, "runtime_images")
if not os.path.exists(known_path):
    os.makedirs(known_path)
if not os.path.exists(runtime_path):
    os.makedirs(runtime_path)
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
def take_images():
    print("Taking Images")
    count = 0
    start_time = time.time()
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            #int arguments are the img then starting point of the rectange then the ending point of the rectange
            #then the color of the rectange then the width of the rectangle

            #extract face data
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100,100))
            file_name = os.path.join(known_path, f"{temp_name}_{count}.jpeg")
            cv2.imwrite(file_name, face)
            count += 1

        cv2.imshow('img',img) #Displays the current frame (img) in a window titled 'img'
        k = cv2.waitKey(30) & 0xff
        if(k == 27) or (time.time() - start_time >= 5):#Checks if the pressed key is ESC (ASCII value 27).
            break

    cap.release()#Releases the webcam resource.
    cv2.destroyAllWindows()
    return True
def analyse_img():
    print("Analysing Image")
    cap = cv2.VideoCapture(0)   # re-open camera
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, img = cap.read()
        if not ret or img is None:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # save runtime face in COLOR
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))
            file_name = os.path.join(runtime_path, f"{temp_name}_runtime.jpg")
            cv2.imwrite(file_name, face)

            # update known images list
            known_images = [
                os.path.join(known_path, f)
                for f in os.listdir(known_path)
                if f.endswith(".jpeg") or f.endswith(".jpg")
            ]

            # compare with known images
            for k_img in known_images:
                try:
                    result = DeepFace.verify(
                        img1_path=file_name, img2_path=k_img, model_name="VGG-Face"
                    )
                    if result["verified"]:
                        print(f"Match found: {file_name} matches {os.path.basename(k_img)}")
                        break
                except Exception as e:
                    print(f"Error comparing {file_name} and {k_img}: {e}")

        cv2.imshow("Analyse", img)
        if cv2.waitKey(30) & 0xff == 27:  # ESC to stop
            break

    cap.release()
    cv2.destroyAllWindows()

def raise_error():
    raise Exception("Error")

def main():
    temp_button = input("Enter 1 to run ")
    if(temp_button == '1'):
        print("Hi Temp this will beocome a seperate winow to woth buttons")
        return_value_flag = take_images()
        if(return_value_flag):
            time.sleep(1)
            analyse_img()
    elif(temp_button == '2'):
        analyse_img()

main()
