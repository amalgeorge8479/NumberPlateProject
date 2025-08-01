import streamlit as st
import os
import cv2
import tempfile
import pandas as pd
from detect_plate import detect_plate_in_image, detect_video, setup_db

# ---------- Streamlit Page Config ---------- #
st.set_page_config(
    page_title="Number Plate Detector",
    page_icon="\U0001F697",
    layout="centered",
    initial_sidebar_state="auto"
)

# ---------- Custom CSS for White Theme ---------- #
white_css = """
<style>
    body, .stApp {
        background-color: white;
        color: #111;
        font-family: 'Segoe UI', sans-serif;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #111;
    }
    .detected-plate-box {
        background-color: #111;
        color: white;
        padding: 1rem 2rem;
        border-radius: 0.5rem;
        font-size: 1.8rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 1rem;
        margin-bottom: 1rem;
        letter-spacing: 3px;
    }
    .vehicle-info {
        font-size: 1.2rem;
        color: #222;
    }
    .plate-container {
        margin-bottom: 2rem;
        border-bottom: 1px solid #eee;
        padding-bottom: 1rem;
    }
    .st-emotion-cache-1gulkj5 {
        display: none !important;
    }
    [role="alert"], [data-testid="stFileUploader"] > div {
        display: none !important;
    }
</style>
"""
st.markdown(white_css, unsafe_allow_html=True)

# ---------- Title ---------- #
st.markdown("""
# 🚗 Number Plate Detection App
Upload an image or video to detect UK number plates and fetch vehicle details.
""")

# ---------- Setup DB ---------- #
setup_db()

# ---------- File Upload ---------- #
uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    is_video = file_ext == "mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
        if file_ext in ["jpg", "jpeg", "png"]:
        # Show uploaded image
            st.image(temp_path, caption="Uploaded Image", use_column_width=True)

        elif file_ext == "mp4":
        # Show uploaded video
             st.video(temp_path)  

    if is_video:
        st.markdown(
            '<div style="color: black; background-color: #e6f0ff; padding: 10px; border-radius: 5px;">Processing video... this may take a while.</div>',
            unsafe_allow_html=True,
        )
        _, df = detect_video(temp_path)
        
        if not df.empty:
            # Get unique plates (remove duplicates)
            unique_plates = df.drop_duplicates(subset=['Plate Number'])
            
            # Display each plate
            for _, row in unique_plates.iterrows():
                plate = row['Plate Number']
                st.markdown(f"""
                <div class='plate-container'>
                    <h3 style='text-align: center; color: black;'>Detected Plate</h3>
                    <div style='
                        display: flex;
                        justify-content: center;
                        margin: 20px 0;
                    '>
                        <div style='
                            font-size: 42px;
                            font-weight: bold;
                            background-color: black;
                            color: white;
                            padding: 16px 50px;
                            border-radius: 10px;
                            text-align: center;
                        '>
                            {plate}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Vehicle Info
                with st.expander(f"📄 Vehicle Details - {plate}", expanded=False):
                    st.markdown(f"""
                    <div class='vehicle-info'>
                        <strong>Make:</strong> {row['Make']}<br>
                        <strong>Model:</strong> {row['Model']}<br>
                        <strong>Year:</strong> {row['Year']}<br>
                        <strong>Region:</strong> {row['Region']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("No plates detected in the video.")

    else:
        st.markdown(
            '<div style="color: black; background-color: #e6f0ff; padding: 10px; border-radius: 5px;">Processing image...</div>',
            unsafe_allow_html=True,
        )
        plate, _ = detect_plate_in_image(temp_path)
        try:
            df = pd.read_csv("detected_plates.csv")
        except:
            df = pd.DataFrame()

        if plate:
            st.markdown(f"""
            <h3 style='text-align: center; color: black;'>Detected Plate</h3>
            <div style='
                display: flex;
                justify-content: center;
                margin: 20px 0;
            '>
                <div style='
                    font-size: 42px;
                    font-weight: bold;
                    background-color: black;
                    color: white;
                    padding: 16px 50px;
                    border-radius: 10px;
                    text-align: center;
                '>
                    {plate}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ---------- Vehicle Info ---------- #
            from detect_plate import fetch_vehicle_info  # Add this import at the top if not present
            vehicle_info = fetch_vehicle_info(plate)
            with st.expander("📄 Vehicle Details", expanded=False):
                if vehicle_info:
                  st.markdown(f"""
                  <div class='vehicle-info'>
                      <strong>Make:</strong> {vehicle_info.get('make', 'N/A')}<br>
                      <strong>Model:</strong> {vehicle_info.get('model', 'N/A')}<br>
                      <strong>Year:</strong> {vehicle_info.get('year', 'N/A')}<br>
                      <strong>Region:</strong> {vehicle_info.get('region', 'N/A')}
                  </div>
                  """, unsafe_allow_html=True)
                else:
                    st.markdown("<span class='vehicle-info'>No vehicle information found.</span>", unsafe_allow_html=True)

        else:
            st.error("❌ No valid plate detected.")