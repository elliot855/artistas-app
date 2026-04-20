import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

from arquitectura import ModeloArtistas

# Clases (ajústalas a tus artistas reales)
CLASES = ["Van Gogh", "Picasso", "Monet", "Da Vinci"]

@st.cache_resource
def cargar_modelo():
    model = ModeloArtistas(num_classes=4)

    state_dict = torch.load("modelo_artistas.pth", map_location="cpu")

    # limpiar prefijos problemáticos
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("model.", "")
        k = k.replace("module.", "")
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    return model

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
        
from utils import generar_gradcam

cam = generar_gradcam(model, img_tensor)

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

img_np = np.array(image.resize((224, 224)))
superpuesta = heatmap * 0.4 + img_np

st.image(superpuesta.astype(np.uint8), caption="Grad-CAM")
