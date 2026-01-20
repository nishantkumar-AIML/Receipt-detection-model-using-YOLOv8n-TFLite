# Receipt-detection-model-using-YOLOv8n-TFLite
receipt_model.tflite for SmartBill Android app.  Model: YOLOv8 Nano (Object Detection)  Format: TensorFlow Lite (Float32)  Resolution: 640x640  Training: 50 Epochs (~5 hours )  Detects: Merchant, Total, Date, Invoice details.

This is a custom Object Detection model I trained to automatically read receipts and invoices. It’s based on "YOLOv8" but I’ve converted it to 'TFLite' so it’s super lightweight and ready to run on "Android apps" or "Edge devices".

I trained this specifically to handle supermarket receipts (like from Billa, Spar, etc.) and extract details like prices, dates,shop name and taxes.

## 🤔 What does it do?
It takes an image of a receipt and detects 12 different types of information. It’s pretty good at telling apart the store name from the actual items!

Here is exactly what it can find:
* 🛒  Stores: `Billa`, `Hofer`, `Spar`, `Unimarkt`
* 📅  Details: `Date`, `Invoice ID`, `Price`, `Sum (Total)`, `Tax Info`
* 📝  Content: `Address`, `Article (Items)`, `Payment Info`

## 🚀 Performance (It works really well!)
I spent some time tuning this model. Even though I had way more examples of "Articles" than "Store Names" in my dataset, the model learned to recognize the stores almost perfectly.

* **Accuracy:** It has an mAP@0.5 of **0.987**. (Basically, it's ~99% accurate).
* **Speed:** Since it's TFLite, it's optimized for speed on mobile devices.

### Check out the training charts:

<img width="2250" height="1500" alt="image" src="https://github.com/user-attachments/assets/6ab4d5f0-2fda-4420-89be-e527c291f96d" />

<img width="3000" height="2250" alt="image" src="https://github.com/user-attachments/assets/53dadbf2-5d20-459c-b537-2703d7b89098" />

<img width="2400" height="1200" alt="image" src="https://github.com/user-attachments/assets/429c0ff3-ea26-46e3-a7c2-a18488ca7201" />


> **My observation:** The "Date" field is the hardest part because every receipt writes dates differently, but the model still gets it right most of the time.

## 📱 How to use it
Since this is a `.tflite` file, you can easily drop it into your Python script or a mobile app.

### Running with Python (Ultralytics)
If you just want to test it out on your PC:

```python
from ultralytics import YOLO

# Load the custom TFLite model
model = YOLO('./best.tflite')

# Test on an image
results = model.predict('my_receipt.jpg')

# Show results
results[0].show()'''

steps to run the program in vs code
python3.11 -m venv .venv
source YOUR FILE LOCATION /bin/activate
python main.py
