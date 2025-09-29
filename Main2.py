import cv2
import os
import time
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(
    r"c:\Advait\VS_Code\VS code 2.0\Face recognition\haarcascade_frontalface_default.xml"
)

def input_name():
    return input("Enter name: ").strip()

name = input_name()

cap = cv2.VideoCapture(0)
base_dir = r"C:\Advait\VS_Code\VS code 2.0\Face recognition\Images"
known_dir = os.path.join(base_dir, "known", name)
runtime_dir = os.path.join(base_dir, "runtime_images")
  
os.makedirs(known_dir, exist_ok=True)
os.makedirs(runtime_dir, exist_ok=True)

def take_images():
    print("Taking Images...")
    count = 0
    start_time = time.time()
    while True:
        ret, img = cap.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))

            file_name = os.path.join(known_dir, f"{name}_{count}.jpg")
            cv2.imwrite(file_name, face)
            count += 1

        cv2.imshow("Capture", img)
        if (cv2.waitKey(30) & 0xff == 27) or (time.time() - start_time > 5):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True

def analyse_img():
    print("Analysing Image...")
    cap = cv2.VideoCapture(0)

    analysis_complete = False
    while True:
        ret, img = cap.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))

            runtime_file = os.path.join(runtime_dir, f"{name}_runtime.jpg")
            cv2.imwrite(runtime_file, face)

            matched = False
            for k_img in os.listdir(known_dir):
                k_img_path = os.path.join(known_dir, k_img)
                if not k_img_path.lower().endswith((".jpg", ".jpeg")):
                    continue

                try:
                    result = DeepFace.verify(
                        img1_path=runtime_file,
                        img2_path=k_img_path,
                        model_name="VGG-Face",
                        distance_metric="cosine",
                        enforce_detection=False
                    )
                    if result["verified"] and result["distance"] < 0.3:
                        print(f"✅ Match found: {k_img}")
                        matched = True
                        break
                except Exception as e:
                    print(f"Error comparing with {k_img}: {e}")

            if not matched:
                print("❌ Unknown face detected")
            else:
                analysis_complete = True
                break # Exit the loop over detected faces

        cv2.imshow("Analyse", img)
        # Exit if 'ESC' is pressed or if a match was found
        if (cv2.waitKey(30) & 0xff == 27) or analysis_complete:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Analysis finished.")

def main():
    choice = input("Enter 1 to capture new images, 2 to analyse: ").strip()
    if choice == "1":
        if take_images():
            time.sleep(1)
            analyse_img()
    elif choice == "2":
        analyse_img()

main()