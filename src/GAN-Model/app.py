import os
import math
import io
import cv2
import torch
import torch.nn.functional as F
import streamlit as st
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

from wgan_gp import Generator  # Your generator definition

device = "cuda" if torch.cuda.is_available() else "cpu"

LR_PATCH_SIZE = 128
UPSCALE_FACTOR = 4
HR_PATCH_SIZE = LR_PATCH_SIZE * UPSCALE_FACTOR  # 512
mean = (0.5, 0.5, 0.5)
std  = (0.5, 0.5, 0.5)
normalize = T.Normalize(mean, std)

CHECKPOINT_DIR = "./02-Upscale-Project/Image-Upscaler/src/GAN-Model/checkpoints"
os.makedirs("streamlit_outputs", exist_ok=True)

@st.cache_data(show_spinner=False)
def load_latest_checkpoint(checkpoint_dir):
    ckpts = [f for f in os.listdir(checkpoint_dir)
             if f.startswith("gan_epoch_") and f.endswith(".pth")]
    if not ckpts:
        raise FileNotFoundError("No checkpoints found in directory.")
    latest_ckpt = max(ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    ckpt_path   = os.path.join(checkpoint_dir, latest_ckpt)

    checkpoint = torch.load(ckpt_path, map_location=device)
    gen = Generator(channels_img=3, features_g=256).to(device)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    gen.eval()
    return gen, ckpt_path

def upscale_with_center_crop(
    pil_img,
    generator,
    lr_overlap=16,
):
    """
    Overlapped chunk-based inference:
      1) Convert PIL -> Tensor, normalize -> [-1,1].
      2) stride = LR_PATCH_SIZE - lr_overlap
      3) Slide window, upsample each patch
      4) Only keep the central region to avoid seams
      5) Return final PIL
    """
    to_tensor = T.ToTensor()
    lr_tensor = to_tensor(pil_img).unsqueeze(0).to(device)
    lr_tensor = normalize(lr_tensor.squeeze(0)).unsqueeze(0)  # ~[-1,1]

    _, _, H, W = lr_tensor.shape
    stride = LR_PATCH_SIZE - lr_overlap

    hr_overlap = lr_overlap * UPSCALE_FACTOR
    half_ov = hr_overlap // 2

    def multiple_of_stride(x):
        remainder = (x - LR_PATCH_SIZE) % stride
        if remainder == 0:
            return x
        return x + (stride - remainder)

    H_pad = multiple_of_stride(H)
    W_pad = multiple_of_stride(W)
    pad_h = H_pad - H
    pad_w = W_pad - W

    lr_tensor = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H_pad, W_pad = lr_tensor.shape

    out_h = H_pad * UPSCALE_FACTOR
    out_w = W_pad * UPSCALE_FACTOR
    sr_canvas = torch.zeros((1, 3, out_h, out_w), device=device)

    for i in range(0, H_pad - LR_PATCH_SIZE + 1, stride):
        for j in range(0, W_pad - LR_PATCH_SIZE + 1, stride):
            lr_patch = lr_tensor[:, :, i:i+LR_PATCH_SIZE, j:j+LR_PATCH_SIZE]
            with torch.no_grad():
                sr_patch = generator(lr_patch)  # [1,3,512,512], [-1,1]

            sr_i = i * UPSCALE_FACTOR
            sr_j = j * UPSCALE_FACTOR

            top = half_ov
            bottom = HR_PATCH_SIZE - half_ov
            left = half_ov
            right = HR_PATCH_SIZE - half_ov

            if i == 0:
                top = 0
            if j == 0:
                left = 0
            if i + LR_PATCH_SIZE >= H_pad:
                bottom = HR_PATCH_SIZE
            if j + LR_PATCH_SIZE >= W_pad:
                right = HR_PATCH_SIZE

            sr_cropped = sr_patch[:, :, top:bottom, left:right]

            sr_ii = sr_i + top
            sr_jj = sr_j + left
            sr_canvas[:, :,
                      sr_ii : sr_ii + (bottom - top),
                      sr_jj : sr_jj + (right - left)] = sr_cropped

    final_h = H * UPSCALE_FACTOR
    final_w = W * UPSCALE_FACTOR
    sr_canvas = sr_canvas[:, :, :final_h, :final_w]

    # De-normalize
    sr_canvas = (sr_canvas * 0.5) + 0.5
    sr_canvas = sr_canvas.clamp(0,1)

    sr_pil = T.ToPILImage()(sr_canvas.squeeze(0).cpu())
    return sr_pil

def remove_artifacts(sr_pil, artifact_method="none"):
    """
    Optional artifact removal:
      - none: do nothing
      - gaussian: mild Gaussian blur
      - bilateral: OpenCV bilateral filter
    """
    if artifact_method == "none":
        return sr_pil

    sr_np = np.array(sr_pil)  # shape [H,W,3], RGB
    if artifact_method == "gaussian":
        pil_blur = sr_pil.filter(ImageFilter.GaussianBlur(radius=1.0))
        return pil_blur
    elif artifact_method == "bilateral":
        sr_bgr = sr_np[..., ::-1]  # BGR
        sr_bgr_filt = cv2.bilateralFilter(sr_bgr, d=9, sigmaColor=75, sigmaSpace=75)
        sr_filt_rgb = sr_bgr_filt[..., ::-1]
        return Image.fromarray(sr_filt_rgb)
    else:
        return sr_pil

def apply_sharpen_and_color(sr_pil, sharpen=False, color_boost=False):
    """
    Additional post-processing for sharpening and color enhancement.
    """
    if sharpen:
        enhancer = ImageEnhance.Sharpness(sr_pil)
        sr_pil = enhancer.enhance(1.5)

    if color_boost:
        enhancer = ImageEnhance.Color(sr_pil)
        sr_pil = enhancer.enhance(1.2)

    return sr_pil

def apply_contrast_brightness(sr_pil, contrast_factor, brightness_factor):
    """
    Adjust contrast and brightness dynamically.
    """
    # Contrast
    contrast_enhancer = ImageEnhance.Contrast(sr_pil)
    sr_pil = contrast_enhancer.enhance(contrast_factor)

    # Brightness
    brightness_enhancer = ImageEnhance.Brightness(sr_pil)
    sr_pil = brightness_enhancer.enhance(brightness_factor)

    return sr_pil

def main():
    st.title("GAN-CNN Upscaling")
    st.write(
        "Upload a low-res image. We'll do patch-based SR with center-crop overlap, "
        "then optionally remove artifacts, sharpen, boost color, and adjust brightness/contrast."
    )

    with st.spinner("Loading generator..."):
        gen, ckpt_path = load_latest_checkpoint(CHECKPOINT_DIR)
    st.success(f"Loaded checkpoint from: {ckpt_path}")

    uploaded_file = st.file_uploader("Choose a low-resolution image", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        lr_pil = Image.open(uploaded_file).convert("RGB")
        st.image(lr_pil, caption="Original LR Image", use_container_width=True)

        user_overlap = st.slider("LR Overlap (pixels)", min_value=0, max_value=32, value=16, step=2)

        artifact_method = st.selectbox(
            "Artifact Removal Method",
            ["none", "gaussian", "bilateral"],
            index=0
        )

        sharpen = st.checkbox("Sharpen Image", value=False)
        color_boost = st.checkbox("Color Saturation Boost", value=False)

        contrast_factor = st.slider("Contrast", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
        brightness_factor = st.slider("Brightness", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

        if st.button("Upscale & Post-Process"):
            with st.spinner("Upscaling..."):
                sr_pil = upscale_with_center_crop(
                    lr_pil, gen,
                    lr_overlap=user_overlap
                )

            with st.spinner("Removing artifacts..."):
                sr_pil = remove_artifacts(sr_pil, artifact_method=artifact_method)

            with st.spinner("Sharpen & color enhancement..."):
                sr_pil = apply_sharpen_and_color(sr_pil, sharpen=sharpen, color_boost=color_boost)

            with st.spinner("Adjusting contrast & brightness..."):
                sr_pil = apply_contrast_brightness(sr_pil, contrast_factor, brightness_factor)

            st.image(sr_pil, caption="Final SR Result", use_container_width=True)

            buf = io.BytesIO()
            sr_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="Download Full-Res SR",
                data=byte_im,
                file_name="sr_center_crop.png",
                mime="image/png"
            )
    else:
        st.info("Please upload an LR image to begin.")

if __name__ == "__main__":
    main()
