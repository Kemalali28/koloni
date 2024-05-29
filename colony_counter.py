import streamlit as st  
from roboflow import Roboflow # type: ignore
from PIL import Image, ImageDraw # type: ignore
import io
import matplotlib.pyplot as plt # type: ignore
from streamlit_drawable_canvas import st_canvas # type: ignore

# API Key untuk Roboflow
api_key = "r8UPKz9aDrKQ43gHmGUx"

# Inisialisasi Roboflow dengan API Key
rf = Roboflow(api_key=api_key)

# Nama workspace dan project
workspace_name = "tes-r5ymt"
project_name = "colony_detection-z4my2"

# Inisialisasi project dan model
project = rf.workspace(workspace_name).project(project_name)
model = project.version(1).model

# Fungsi untuk melakukan prediksi dan menampilkan gambar
def detect_objects(image_path, confidence, overlap):
    result = model.predict(image_path, confidence=confidence, overlap=overlap).json()
    
    # Buka gambar asli
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Gambar bounding box pada gambar asli
    for prediction in result['predictions']:
        x = prediction['x']
        y = prediction['y']
        width = prediction['width']
        height = prediction['height']
        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2
        draw.rectangle([left, top, right, bottom], outline="blue", width=2)
    

    # Tampilkan gambar dengan matplotlib
    st.image(image, caption='Gambar dengan Deteksi Objek', use_column_width=True)

    # Tampilkan jumlah objek yang terdeteksi
    total_objects = len(result['predictions'])
    st.write(f"Total objek yang terdeteksi: {total_objects}")
    
    return image, total_objects

# Definisikan halaman
def beranda():
    st.title("Beranda")
    st.write("Selamat datang di aplikasi deteksi koloni bakteri. Gunakan halaman deteksi untuk mengunggah gambar dan mendeteksi objek.")

def deteksi():
    st.title("Deteksi Koloni Bakteri")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["png", "jpg", "jpeg"])
    confidence = st.slider("Confidence Threshold", 0, 100, 40)
    overlap = st.slider("Overlap Threshold", 0, 100, 30)
    
    if uploaded_file is not None:
        # Simpan file yang diunggah ke disk
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption='Gambar yang diunggah.', use_column_width=True)
        st.write("")
        st.write("Mendeteksi objek...")
        
        # Lakukan deteksi objek
        image, total_detected_objects = detect_objects("temp_image.png", confidence, overlap)
        
        # Konversi gambar ke format yang bisa dianotasi
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()

        # Anotasi manual menggunakan streamlit-drawable-canvas
        st.write("Tambahkan koloni secara manual jika perlu:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="red",
            background_image=Image.open(io.BytesIO(image_bytes)),
            update_streamlit=True,
            height=500,
            width=500,
            drawing_mode="rect",
            key="canvas"
        )
        
        # Hitung total objek yang terdeteksi secara manual
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            total_manual_objects = len(objects)
            
            # Buka kembali gambar asli untuk menambahkan anotasi manual
            image = Image.open("temp_image.png")
            draw = ImageDraw.Draw(image)
            for obj in objects:
                left = obj["left"]
                top = obj["top"]
                width = obj["width"]
                height = obj["height"]
                draw.rectangle([left, top, left + width, top + height], outline="blue", width=2)
                draw.text((left, top), "Manual", fill="blue")
                

            
            # Tampilkan total objek yang terdeteksi termasuk yang manual
            st.write(f"Total objek yang terdeteksi: {total_detected_objects + total_manual_objects}")

# Navigasi halaman
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman:", ("Beranda", "Deteksi"))

if page == "Beranda":
    beranda()
else:
    deteksi()
