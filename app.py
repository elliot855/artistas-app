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

st.title("🎨 Clasificador de Obras de Arte")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    st.subheader(f"🎯 Predicción: {CLASES[pred]}")
    st.write("Probabilidades:")
    for i, clase in enumerate(CLASES):
        st.write(f"{clase}: {probs[0][i]:.4f}")

# 🔥 Grad-CAM AQUÍ dentro
    from utils import generar_gradcam

    cam = generar_gradcam(model, img_tensor)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_np = np.array(image.resize((224, 224)))
    superpuesta = heatmap * 0.4 + img_np

    st.image(superpuesta.astype(np.uint8), caption="Grad-CAM")

    img_np = np.array(image.resize((224, 224)))
    superpuesta = heatmap * 0.4 + img_np

    st.image(superpuesta.astype(np.uint8), caption="Grad-CAM")
