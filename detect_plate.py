#Importing Required Libraries
import os, cv2, re, csv, sys, sqlite3, requests
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import Counter
import easyocr

# Configuration Settings
MODEL_PATH = "best.pt"
OUTPUT_CSV, OUTPUT_EXCEL = "detected_plates.csv", "detected_plates.xlsx"
CROPPED_PLATES_DIR, FRAMES_DIR = "cropped_plates", "frames"
FRAME_SKIP = 2
UK_PLATE_REGEX = r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$'
OCR_ALLOWLIST = 'ABCDEFGHJKLMNPRSTUVWXYZ0123456789'
os.makedirs(CROPPED_PLATES_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
model, reader = YOLO(MODEL_PATH), easyocr.Reader(['en'])

#APi to fetch vehicle info
def fetch_vehicle_info(plate):
    url = "https://uk.api.vehicledataglobal.com/r2/lookup"
    API_KEY = "3FF4A98B-F1D7-4B79-B7E6-F506EF9B2CC6"
    PACKAGE_NAME = "VehicleDetails"
    params = {
        "packagename": PACKAGE_NAME,
        "apikey": API_KEY,
        "vrm": plate
    }
    try:
        response = requests.get(url, params=params)
        print("API response:", response.text)  # Debug print
        if response.status_code == 200:
            data = response.json()
            vehicle_data = data.get("Results", {}).get("VehicleDetails", {})
            model_data = data.get("Results", {}).get("ModelDetails", {}).get("ModelIdentification", {})
            color_data = data.get("Results", {}).get("VehicleHistory", {}).get("ColourDetails", {})
            return {
                "make": model_data.get("Make"),
                "model": model_data.get("Model"),
                "year": vehicle_data.get("VehicleIdentification", {}).get("YearOfManufacture"),
                "region": plate[:2].upper()
            }
        print(f"❌ API error or no data found: {response.status_code}")
    except Exception as e:
        print(f"❌ Exception during API call: {e}")
    return {"make": None, "model": None, "year": None, "color": None}

#Database Setup Function
def setup_db():
    conn = sqlite3.connect("plates.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS plate_detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_number TEXT,
        timestamp TEXT,
        frame_number INTEGER,
        image_path TEXT,
        region_code TEXT,
        make TEXT,
        model TEXT,
        year INTEGER,
        color TEXT
    )''')
    conn.commit()
    conn.close()

# Plate Correction Function
def correct_uk_plate(plate):
    cands = [plate]
    if len(plate) == 7:
        for i in [2, 3]:
            if plate[i] == "L": cands.append(plate[:i]+"1"+plate[i+1:])
            if plate[i] == "1": cands.append(plate[:i]+"L"+plate[i+1:])
        cands += [plate.replace("8","B"), plate.replace("B","8"), plate.replace("0","O"), plate.replace("O","0")]
    for c in cands:
        if re.match(UK_PLATE_REGEX, c): return c
    return plate

# OCR Processing Function
def get_best_ocr_result(crop, reader):
    results = []
    results.append(reader.readtext(crop, detail=0, paragraph=False, allowlist=OCR_ALLOWLIST))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    results.append(reader.readtext(gray, detail=0, paragraph=False, allowlist=OCR_ALLOWLIST))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.medianBlur(thresh, 3)
    results.append(reader.readtext(thresh, detail=0, paragraph=False, allowlist=OCR_ALLOWLIST))
    best_plate, max_score = "", -1
    for r in results:
        if not r: continue
        text = "".join(r).upper().replace(" ", "").strip()
        norm = re.sub(r'[^A-Z0-9]', '', text)
        norm = correct_uk_plate(norm)
        score = 20 if re.match(UK_PLATE_REGEX, norm) else 5 if re.match(r'^[A-Z0-9]{5,8}$', norm) else 0
        score += len(norm)
        if score > max_score: best_plate, max_score = norm, score
    return best_plate, max_score

# Consensus Plate Function
def get_consensus_plate(cands, best_plate):
    if not cands: return ""
    l = [len(c) for c in cands]
    if not l: return ""
    mlen = Counter(l).most_common(1)[0][0]
    fc = [c for c in cands if len(c) == mlen]
    if not fc: return ""
    cp = ""
    for i in range(mlen):
        chars = [c[i] for c in fc]
        vote = Counter(chars).most_common()
        cp += best_plate[i] if len(vote)>1 and vote[0][1]==vote[1][1] and i<len(best_plate) else vote[0][0]
    return cp

def levenshtein_distance(s1, s2):
    if len(s1)<len(s2): return levenshtein_distance(s2,s1)
    if len(s2)==0: return len(s1)
    prev = range(len(s2)+1)
    for i,c1 in enumerate(s1):
        curr = [i+1]
        for j,c2 in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1!=c2)))
        prev = curr
    return prev[-1]

# Video Processing Function
def detect_video(video_path):
    frame_count, tracked_plates, CONFIRM = 0, [], 3
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print("Error: Cannot open video."); return "Not Detected", pd.DataFrame()
    with open(OUTPUT_CSV, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Frame", "Plate Number", "Filename", "Region", "Make", "Model", "Year", "Color"])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            if frame_count % FRAME_SKIP: continue
            results = model(frame, conf=0.5)
            for box in results[0].boxes:
                conf, cls = float(box.conf[0]), int(box.cls[0])
                if conf<0.5 or cls!=0: continue
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cropped = frame[y1:y2, x1:x2]
                h,w = cropped.shape[:2]
                if w==0 or h==0: continue
                ar = w/h
                if ar<1.2 or ar>8.0 or w<50 or h<15: continue
                plate, score = get_best_ocr_result(cropped, reader)
                if len(plate)<5: continue
                if not re.match(UK_PLATE_REGEX, plate): continue
                region_code = plate[:2].upper()
                match_found = False
                for tracker in tracked_plates:
                    ptc = tracker.get('consensus_plate', tracker['plate'])
                    if levenshtein_distance(plate, ptc) <= 2:
                        match_found = True
                        tracker['last_frame'] = frame_count
                        tracker['hits'] += 1
                        tracker['candidates'].append(plate)
                        if score > tracker['score']: tracker['plate'], tracker['score'] = plate, score
                        if tracker['hits'] >= CONFIRM and not tracker['saved']:
                            final_plate = get_consensus_plate(tracker['candidates'], tracker['plate'])
                            plate_filename = f"{CROPPED_PLATES_DIR}/{final_plate}_frame{frame_count}.jpg"
                            cv2.imwrite(plate_filename, cropped)
                            vehicle_info = fetch_vehicle_info(final_plate) or {}
                            print("Fetched vehicle info:", vehicle_info)  # Debug print
                            csv_writer.writerow([frame_count, final_plate, os.path.basename(plate_filename), region_code,
                                vehicle_info.get('make'), vehicle_info.get('model'), vehicle_info.get('year'), vehicle_info.get('color')])
                            csv_file.flush()
                            try:
                                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                conn = sqlite3.connect("plates.db")
                                conn.execute('''INSERT INTO plate_detections (
                                    plate_number, timestamp, frame_number, image_path, region_code, make, model, year, color
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                    (final_plate, timestamp, frame_count, plate_filename, region_code,
                                    vehicle_info.get('make'), vehicle_info.get('model'), vehicle_info.get('year'), vehicle_info.get('color')))
                                conn.commit()
                            except Exception as e:
                                print(f"❌ Failed to insert into DB for {final_plate}: {e}")
                            finally: conn.close()
                            print(f"✅ [Frame {frame_count}] Plate Confirmed: {final_plate} (from {tracker['candidates']})")
                            tracker['saved'], tracker['consensus_plate'] = True, final_plate
                        break
                if not match_found:
                    tracked_plates.append({'plate': plate, 'candidates': [plate], 'score': score,
                        'last_frame': frame_count, 'hits': 1, 'saved': False})
            annotated = results[0].plot()
            resized = cv2.resize(annotated, (640, 360))
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()
    try:
        df = pd.read_csv(OUTPUT_CSV)
        df.to_excel(OUTPUT_EXCEL, index=False)
        print("✅ Excel saved:", OUTPUT_EXCEL)
        last_plate = df.iloc[-1]["Plate Number"] if not df.empty else "Not Detected"
        print("\n✅ Done! Cropped plates saved in 'cropped_plates/', annotated frames in 'frames/', results in CSV & Excel.")
        return last_plate, df
    except Exception as e:
        print("Excel export failed:", e)
        return "Not Detected", pd.DataFrame()

# Image Processing Function
def detect_plate_in_image(input_path):
    frame = cv2.imread(input_path)
    if frame is None: print(f"Error: Cannot open image {input_path}."); sys.exit(1)
    results = model(frame, conf=0.5)
    best_plate, best_score, cropped_plate_img = "", -1, None
    for box in results[0].boxes:
        conf, cls = float(box.conf[0]), int(box.cls[0])
        if conf<0.5 or cls!=0: continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cropped = frame[y1:y2, x1:x2]
        h,w = cropped.shape[:2]
        cv2.imwrite(f"debug_crop_img_{x1}_{y1}.jpg", cropped)
        ocr_results = reader.readtext(cropped, detail=0, paragraph=False, allowlist=OCR_ALLOWLIST)
        print("OCR raw results:", ocr_results)
        if w==0 or h==0: continue
        if w<120 or h<40:
            cropped = cv2.resize(cropped, (max(120,w*2), max(40,h*2)), interpolation=cv2.INTER_CUBIC)
        plate, score = get_best_ocr_result(cropped, reader)
        if score > best_score and len(plate) >= 5:
            best_plate, best_score, cropped_plate_img = plate, score, cropped
    if best_plate:
        region_code = best_plate[:2].upper()
        plate_filename = f"{CROPPED_PLATES_DIR}/{best_plate}_img.jpg"
        cv2.imwrite(plate_filename, cropped_plate_img)
        vehicle_info = fetch_vehicle_info(best_plate) or {}
        print("Fetched vehicle info:", vehicle_info)  # Debug print
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn = sqlite3.connect("plates.db")
            conn.execute('''INSERT INTO plate_detections (
                plate_number, timestamp, frame_number, image_path, region_code, make, model, year, color
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (best_plate, timestamp, 1, plate_filename, region_code,
                vehicle_info.get('make'), vehicle_info.get('model'), vehicle_info.get('year'), vehicle_info.get('color')))
            conn.commit()
        except Exception as e:
            print(f" Failed to insert into DB for {best_plate}: {e}")
        finally: conn.close()
        # Write to CSV
        with open(OUTPUT_CSV, "a", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            if os.path.getsize(OUTPUT_CSV) == 0:
                csv_writer.writerow(["Frame", "Plate Number", "Filename", "Region", "Make", "Model", "Year", "Color"])
            csv_writer.writerow([1, best_plate, os.path.basename(plate_filename), region_code,
                                 vehicle_info.get('make'), vehicle_info.get('model'), vehicle_info.get('year'), vehicle_info.get('color')])
        # Write to Excel
        try:
            df = pd.read_csv(OUTPUT_CSV)
            df.to_excel(OUTPUT_EXCEL, index=False)
            print("✅ Excel saved:", OUTPUT_EXCEL)
        except Exception as e:
            print("Excel export failed:", e)
        print(f"✅ [Image] Plate Detected: {best_plate}")
        return best_plate, plate_filename
    print("❌ No valid plate detected.")
    return "Not Detected", None

# Main Execution Block
if __name__ == "__main__":
    setup_db()
    if len(sys.argv) < 3:
        print("Usage: python detect_plate.py <image|video> <path>")
        sys.exit(1)
    mode, input_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(input_path):
        print(f"Input file {input_path} does not exist."); sys.exit(1)
    if mode == "image": detect_plate_in_image(input_path)
    elif mode == "video": detect_video(input_path)
    else: print("Invalid mode. Use 'image' or 'video'.")