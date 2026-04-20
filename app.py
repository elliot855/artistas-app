import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

from arquitectura import ModeloArtistas


# Clases (ajústalas a tus artistas reales)
CLASES = ['camille-pissarro', 'claude-monet', 'edgar-degas', 'pierre-auguste-renoir']

@st.cache_resource
def cargar_modelo():
    model = ModeloArtistas(num_classes=4)

    # 1. Cargar el archivo correcto (modelo_artistas.pth)
    state_dict = torch.load("modelo_artistas.pth", map_location="cpu")

    # 2. Arreglar las llaves del diccionario: Añadir "model."
    new_state_dict = {}
    for k, v in state_dict.items():
        # Como CNN.py guardó las capas sin "model.", se lo agregamos
        new_state_dict[f"model.{k}"] = v

    # 3. Cargar pesos usando strict=True para asegurarnos de que todo encaje
    model.load_state_dict(new_state_dict, strict=True)

    model.eval()
    return model
model = cargar_modelo()

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- INTERFAZ ORIGINAL ---
st.title("🎨 Clasificador de Obras de Arte")

# Mantenemos el cargador de archivos original
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])

# Añadimos la cámara justo debajo (esto activará la cámara en tu Android)
foto_camara = st.camera_input("O toma una foto directamente")

# Lógica para decidir qué imagen procesar (archivo o cámara)
image_file = None
if uploaded_file is not None:
    image_file = uploaded_file
elif foto_camara is not None:
    image_file = foto_camara

# Si hay alguna imagen seleccionada, hacemos la predicción
if image_file:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Imagen para analizar", use_container_width=True)

    # Preprocesamiento
    img_tensor = transform(image).unsqueeze(0)

    # Inferencia (Predicción)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    # Resultados (Interfaz original)
    st.subheader(f"🎯 Predicción: {CLASES[pred]}")
    st.write(f"**Confianza:** {probs[0][pred]:.2%}")
    
    st.write("Probabilidades detalladas:")
    for i, clase in enumerate(CLASES):
        st.write(f"{clase.replace('-', ' ').title()}: {probs[0][i]:.4f}")

