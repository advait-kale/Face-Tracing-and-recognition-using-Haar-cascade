import cv2
import os
import time
import numpy as np
from deepface import DeepFace

# -------------------- config & paths --------------------
CASCADE_PATH = r"c:\Advait\VS_Code\VS code 2.0\Face recognition\haarcascade_frontalface_default.xml"

KNOWN_DIR    = os.path.join(BASE_DIR, "known")
RUNTIME_DIR  = os.path.join(BASE_DIR, "runtime_images")
TEMP_FACE    = os.path.join(RUNTIME_DIR, "temp_face.jpg")

os.makedirs(KNOWN_DIR,   exist_ok=True)
os.makedirs(RUNTIME_DIR, exist_ok=True)

# load detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

MODEL_NAME = "VGG-Face"            #1 keep your model
DISTANCE_METRIC = "cosine"         # cosine distance
COSINE_THRESH = 0.30               # stricter threshold (tune 0.25~0.35)
SKIP_FRAMES = 10                   # verify every N frames to reduce lag
DETECT_SIZE = (640, 480)           # downscale for faster detection
CROP_SIZE   = (224, 224)           # VGG-Face friendly crop

# -------------------- helpers --------------------
def cosine_distance(a, b):
    a = np.array(a); b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return 1.0 - float(np.dot(a, b) / denom)

def list_known_images():
    paths = []
    for root, _, files in os.walk(KNOWN_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(root, f))
    return paths

def label_from_path(path):
    # KNOWN_DIR/<person_name>/image.jpg  -> person_name
    parts = os.path.normpath(path).split(os.sep)
    # find index of "known" and take the next part as person folder
    for i, p in enumerate(parts):
        if p == os.path.basename(KNOWN_DIR):
            if i + 1 < len(parts):
                return parts[i + 1]
    return "unknown"

# -------------------- capture (saves per person) --------------------
def take_images(person_name="user"):
    print("📸 Taking Images...")
    person_dir = os.path.join(KNOWN_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count = 0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        # downscale for detection
        small = cv2.resize(frame, DETECT_SIZE)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)  # detection only
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(small, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # map back to original scale (frame may be larger than DETECT_SIZE)
            sx = frame.shape[1] / DETECT_SIZE[0]
            sy = frame.shape[0] / DETECT_SIZE[1]
            X, Y, W, H = int(x*sx), int(y*sy), int(w*sx), int(h*sy)

            crop = frame[Y:Y+H, X:X+W]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, CROP_SIZE)
            out_path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
            cv2.imwrite(out_path, crop)
            count += 1

        cv2.imshow("Capture", small)
        if (cv2.waitKey(1) & 0xFF) == 27 or (time.time() - start) > 5:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Saved {count} images in {person_dir}")
    return count > 0

# -------------------- cache embeddings (once) --------------------
def build_known_cache():
    print("⚡ Building cache of known embeddings...")
    paths = list_known_images()
    if not paths:
        print(f"⚠️ No images found in: {KNOWN_DIR}")
        return [], [], []

    embeddings, labels, paths_out = [], [], []
    for p in paths:
        try:
            # DeepFace.represent for your version: use img_path only (no 'model' kwarg)
            rep = DeepFace.represent(
                img_path=p,
                model_name=MODEL_NAME,
                enforce_detection=False
            )
            if isinstance(rep, list) and len(rep) > 0 and "embedding" in rep[0]:
                emb = rep[0]["embedding"]
            else:
                emb = rep["embedding"] if isinstance(rep, dict) else None

            if emb is not None:
                embeddings.append(emb)
                labels.append(label_from_path(p))
                paths_out.append(p)
                print(f"  ✓ cached: {os.path.basename(p)}")
        except Exception as e:
            print(f"  ✗ skipping {p}: {e}")

    print(f"✅ Cached {len(embeddings)} face embeddings.")
    return embeddings, labels, paths_out

# -------------------- analyse (fast) --------------------
def analyse_img():
    # Build cache once (fast to compare later)
    known_embs, known_labels, known_paths = build_known_cache()
    if not known_embs:
        print("❌ No known embeddings available. Capture faces first.")
        return

    print("🔍 Analysing...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        small = cv2.resize(frame, DETECT_SIZE)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)  # detection only
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(small, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # map back to original coords and crop from full-res frame
            sx = frame.shape[1] / DETECT_SIZE[0]
            sy = frame.shape[0] / DETECT_SIZE[1]
            X, Y, W, H = int(x*sx), int(y*sy), int(w*sx), int(h*sy)

            crop = frame[Y:Y+H, X:X+W]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, CROP_SIZE)

            # only compute every N frames to reduce lag
            label_text = "..."
            if frame_idx % SKIP_FRAMES == 0:
                # save temp crop to disk and get embedding
                cv2.imwrite(TEMP_FACE, crop)
                try:
                    rep = DeepFace.represent(
                        img_path=TEMP_FACE,
                        model_name=MODEL_NAME,
                        enforce_detection=False
                    )
                    if isinstance(rep, list) and len(rep) > 0 and "embedding" in rep[0]:
                        run_emb = rep[0]["embedding"]
                    else:
                        run_emb = rep["embedding"] if isinstance(rep, dict) else None

                    if run_emb is not None:
                        # compare to cached embeddings
                        best_label = "Unknown"
                        best_dist  = 999.0
                        for k_emb, k_lab in zip(known_embs, known_labels):
                            d = cosine_distance(run_emb, k_emb)
                            if d < best_dist:
                                best_dist, best_label = d, k_lab

                        if best_dist < COSINE_THRESH:
                            label_text = f"{best_label} ({best_dist:.2f})"
                        else:
                            label_text = f"Unknown ({best_dist:.2f})"
                    else:
                        label_text = "No embedding"
                except Exception as e:
                    label_text = f"Err: {str(e)[:18]}"

            # draw label
            cv2.putText(small, label_text, (x, max(0, y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if "Unknown" not in label_text else (0,0,255), 2)

        cv2.imshow("Analyse", small)
        frame_idx += 1

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------- main --------------------
def main():
    choice = input("Enter 1 to capture new images, 2 to analyse: ").strip()
    if choice == "1":
        person = input("Enter your name: ").strip()
        if take_images(person):
            # after new images, immediately analyse
            analyse_img()
    elif choice == "2":
        analyse_img()
    else:
        print("Choose 1 or 2.")

if __name__ == "__main__":
    main()
