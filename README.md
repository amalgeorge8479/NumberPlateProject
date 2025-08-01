# Number Plate Detection System

This project detects UK number plates from images or videos using YOLOv8 and EasyOCR, and displays vehicle details using a public API in a simple Streamlit web interface.

## Technologies Used

- YOLOv8 (Ultralytics)
- EasyOCR
- OpenCV
- Streamlit (for UI)
- SQLite & CSV/Excel Export

## How to Run

1. Install dependencies:
- pip install streamlit opencv-python ultralytics easyocr pandas numpy requests

2. Run the Streamlit app:
- streamlit run app.py

3. Upload an image or video file and see the result!

##  Output Includes:
- Cropped number plate images
- Annotated frames
- CSV & Excel summary
- Vehicle info: Make, Model, Year, Region