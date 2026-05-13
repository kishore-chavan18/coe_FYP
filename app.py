import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from fpdf import FPDF
import os
from io import BytesIO

st.set_page_config(
    page_title="Smart Document Scanner",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Smart Document Scanner")
st.write("Upload a document image and convert it into a clean scanned PDF.")

# -----------------------------
# Helper Functions
# -----------------------------

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def scan_document(image):
    # Convert PIL to OpenCV
    img = np.array(image)
    original = img.copy()

    # Resize for processing speed
    ratio = img.shape[0] / 500.0
    resized = cv2.resize(img, (int(img.shape[1] / ratio), 500))

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(
        edged.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    # If document contour found
    if screenCnt is not None:
        warped = four_point_transform(
            original,
            screenCnt.reshape(4, 2) * ratio
        )
    else:
        warped = original

    # Convert to grayscale
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold for clean B&W scan
    scanned = cv2.adaptiveThreshold(
        gray_warped,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return scanned


def image_to_pdf(image_array):
    temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_image.name, image_array)

    pdf = FPDF()
    pdf.add_page()

    pdf.image(temp_image.name, x=10, y=10, w=190)

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)

    with open(temp_pdf.name, "rb") as f:
        pdf_bytes = f.read()

    os.unlink(temp_image.name)
    os.unlink(temp_pdf.name)

    return pdf_bytes


# -----------------------------
# Upload Section
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Document Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("✨ Scan Document"):
        with st.spinner("Processing..."):

            scanned_image = scan_document(image)

            with col2:
                st.subheader("Scanned Image")
                st.image(scanned_image, clamp=True, use_container_width=True)

            # Convert scanned image to bytes
            pil_scanned = Image.fromarray(scanned_image)

            img_buffer = BytesIO()
            pil_scanned.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            # PDF bytes
            pdf_bytes = image_to_pdf(scanned_image)

            st.success("Document scanned successfully!")

            # Download buttons
            st.download_button(
                label="⬇ Download Scanned Image",
                data=img_bytes,
                file_name="scanned_document.png",
                mime="image/png"
            )

            st.download_button(
                label="⬇ Download PDF",
                data=pdf_bytes,
                file_name="scanned_document.pdf",
                mime="application/pdf"
            )

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")
st.caption("Built with Streamlit + OpenCV")
