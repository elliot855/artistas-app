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

# 4. Interfaz de Usuario
st.set_page_config(page_title="IA Clasificador de Arte", page_icon="🎨")
st.title("🎨 Clasificador de Obras de Arte")
st.write("Identifica al autor de una pintura usando Inteligencia Artificial.")

# Selector de entrada
opcion = st.radio("Selecciona una opción:", ("Subir imagen de la galería", "Tomar una foto con la cámara"))

imagen_pil = None

if opcion == "Subir imagen de la galería":
    archivo = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
    if archivo is not None:
        imagen_pil = Image.open(archivo).convert("RGB")

else:
    # Esta opción activa la cámara en Android/iOS cuando abres el link
    foto = st.camera_input("Enfoca la obra de arte")
    if foto is not None:
        imagen_pil = Image.open(foto).convert("RGB")

# 5. Ejecutar Predicción si hay una imagen
if imagen_pil is not None:
    st.image(imagen_pil, caption="Imagen seleccionada", use_container_width=True)
    
    # Preprocesamiento
    img_tensor = transform(imagen_pil).unsqueeze(0)

    # Inferencia
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    # Mostrar Resultados
    artista_predicho = CLASES[pred_idx].replace('-', ' ').title()
    confianza = probs[0][pred_idx].item()

    st.success(f"### 🎯 Predicción: **{artista_predicho}**")
    st.write(f"Nivel de confianza: **{confianza:.2%}**")

    st.write("---")
    st.write("### 📊 Probabilidades por artista:")
    
    # Mostrar barras de probabilidad
    for i, nombre_clase in enumerate(CLASES):
        p = probs[0][i].item()
        nombre_bonito = nombre_clase.replace('-', ' ').title()
        st.write(f"**{nombre_bonito}**")
        st.progress(p)
        st.write("Probabilidades:")
        for i, clase in enumerate(CLASES):
            st.write(f"{clase}: {probs[0][i]:.4f}")

