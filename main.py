import os
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_FILE = "receipt_model.tflite"
ACCEPTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

def main():
    print("🚀 Starting Receipt Parser...")

    # 1. Model Check
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: '{MODEL_FILE}' nahi mila!")
        print("👉 Make sure ye file isi folder mein ho.")
        return

    # 2. Model Load
    try:
        print(f"📦 Loading {MODEL_FILE}...")
        # Yahan task='detect' add kiya hai warning hatane ke liye
        model = YOLO(MODEL_FILE, task='detect') 
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return

    # 3. Find Images
    files = os.listdir(".")
    images = [f for f in files if f.lower().endswith(ACCEPTED_EXTENSIONS)]

    if not images:
        print("⚠️ Koi image nahi mili!")
        print(f"👉 Please ek photo ({ACCEPTED_EXTENSIONS}) folder mein daalo.")
        return

    print(f"📸 Found {len(images)} image(s). Processing...\n")

    # 4. Process Images
    for img_file in images:
        print(f"🔎 Scanning: {img_file} ...")
        # task='detect' yahan bhi explicitly bata diya
        results = model.predict(img_file, save=True, conf=0.5, task='detect')
        
        for result in results:
            result.show() # Popup
            
            print(f"\n--- Result for {img_file} ---")
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = result.names[cls_id]
                conf = float(box.conf[0])
                print(f"   ✅ Found: {name} (Confidence: {conf:.2f})")

    print("\n🎉 Done! Results 'runs/detect' folder mein save ho gaye hain.")

if __name__ == "__main__":
    main()