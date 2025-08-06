# 🚗 Number Plate Detection System

This project detects UK number plates from images or videos using YOLOv8 and EasyOCR, and displays vehicle details using a public API in a simple Streamlit web interface.

## Technologies Used

- YOLOv8 (Ultralytics)
- EasyOCR
- OpenCV
- Streamlit (for UI)
- SQLite & CSV/Excel Export

##  Features

- 📸 **Image and Video Input** – Upload from browser UI  
- 🅿️ **Automatic Plate Detection** – Detects and crops plates using YOLOv8  
- 🔡 **OCR for Plate Text** – EasyOCR reads plate numbers  
- 🌍 **Vehicle Info Lookup** – Integration with a public vehicle data API  
- 📂 **Export Results** – Outputs CSV, Excel, cropped images, and annotated frames  
- 💡 **Simple Web Interface** – Built with Streamlit for ease of use  
- 🧹 **Validates UK Format** – Filters out invalid or foreign plates  
- 🗂️ **Database Storage** – Records results in SQLite

### How to Run

## 1.  Install Dependencies

Make sure you have Python installed, then run:

```bash
pip install streamlit opencv-python ultralytics easyocr pandas numpy requests
```

## 2.  Run the Streamlit App

Start the application with:

```bash
streamlit run app.py
```

Once started, it will open in your browser at:
```arduino
http://localhost:
```


##  Output Includes:

- Cropped number plate images

- Annotated image frames

- Extracted plate numbers

- Vehicle information:

  Make, Model, Year, Region

- Data export to:

- CSV file

- Excel file

- SQLite database
