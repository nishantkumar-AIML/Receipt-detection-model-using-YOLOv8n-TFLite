import os
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_FILE = "receipt_model.tflite"
ACCEPTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

def main():
    print("🚀 Starting Receipt Parser...")

    # 1. Model Check
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: '{MODEL_FILE}' can't find !")
        print("👉 Make sure this file in this folder .")
        return

    # 2. Model Load
    try:
        print(f"📦 Loading {MODEL_FILE}...")
        #  task='detect' 
        model = YOLO(MODEL_FILE, task='detect') 
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return

    # 3. Find Images
    files = os.listdir(".")
    images = [f for f in files if f.lower().endswith(ACCEPTED_EXTENSIONS)]

    if not images:
        print("⚠️ not able to find image !")
        print(f"👉 Please add photo ({ACCEPTED_EXTENSIONS}) in this folder .")
        return

    print(f"📸 Found {len(images)} image(s). Processing...\n")

    # 4. Process Images
    for img_file in images:
        print(f"🔎 Scanning: {img_file} ...")
        # task='detect' 
        results = model.predict(img_file, save=True, conf=0.5, task='detect')
        
        for result in results:
            result.show() # Popup
            
            print(f"\n--- Result for {img_file} ---")
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = result.names[cls_id]
                conf = float(box.conf[0])
                print(f"   ✅ Found: {name} (Confidence: {conf:.2f})")

    print("\n🎉 Done! Results 'runs/detect' save in folder .")

if __name__ == "__main__":
    main()
