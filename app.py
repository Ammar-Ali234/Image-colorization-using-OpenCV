import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -------------------------------
# Load Models (cached for speed)
# -------------------------------
@st.cache_resource
def load_colorization_model():
    proto_file = "model/colorization_deploy_v2.prototxt"
    model_file = "model/colorization_release_v2.caffemodel"
    pts_file = "model/pts_in_hull.npy"

    pts_in_hull = np.load(pts_file)

    net = cv2.dnn.readNetFromCaffe(proto_file, model_file)

    # Add cluster centers
    pts = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
        np.full([1, 313], 2.606, np.float32)
    ]

    return net


# -------------------------------
# Colorization Function
# -------------------------------
def colorize_image(bw_image, net):
    frame = np.array(bw_image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame_float = frame.astype("float32") / 255.0
    lab = cv2.cvtColor(frame_float, cv2.COLOR_BGR2LAB)

    L = lab[:, :, 0]
    L_resized = cv2.resize(L, (224, 224))
    L_resized -= 50

    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab_decoded = net.forward()[0].transpose((1, 2, 0))

    ab_resized = cv2.resize(ab_decoded, (frame.shape[1], frame.shape[0]))

    lab_out = np.concatenate((L[:, :, np.newaxis], ab_resized), axis=2)

    colorized = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    output = (colorized * 255).astype("uint8")

    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Black & White Image Colorizer")

st.write("Upload a black & white photo, and the AI model will colorize it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

net = load_colorization_model()

if uploaded_file:
    bw_img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Original")
        st.image(bw_img, use_container_width=True)

    with col2:
        st.header("Colorized")
        colorized_img = colorize_image(bw_img, net)
        st.image(colorized_img, use_container_width=True)
