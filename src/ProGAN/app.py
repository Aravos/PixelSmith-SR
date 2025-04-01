import os
import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageEnhance # Import ImageEnhance
from skimage import exposure
from skimage.color import rgb2lab, lab2rgb
from skimage import img_as_float

try:
    from models import Generator
except ImportError:
    st.error("Could not import the Generator model. Make sure 'models.py' is in the correct path or added to PYTHONPATH.")
    st.info("Common Fixes:\n1. Place `models.py` in the same directory as this Streamlit script.\n2. Ensure the directory containing `models.py` is in your system's PYTHONPATH environment variable.")
    st.stop()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPSCALE_FACTOR = 2
STEP = 1
LR_PATCH_SIZE = 128
HR_PATCH_SIZE = LR_PATCH_SIZE * UPSCALE_FACTOR

DEFAULT_CHECKPOINT_PATH = "./02-Upscale-Project/Image-Upscaler/src/ProGAN/Prod-model/training_state.pth"

mean = (0.5, 0.5, 0.5)
std  = (0.5, 0.5, 0.5)
normalize = T.Normalize(mean, std)
to_tensor = T.ToTensor()
to_pil = T.ToPILImage()


# --- Initialize Session State ---
if 'sharpen' not in st.session_state:
    st.session_state.sharpen = False
if 'sharpen_radius' not in st.session_state:
    st.session_state.sharpen_radius = 1.5
if 'sharpen_percent' not in st.session_state:
    st.session_state.sharpen_percent = 150
if 'sharpen_threshold' not in st.session_state:
    st.session_state.sharpen_threshold = 3
if 'smooth' not in st.session_state:
    st.session_state.smooth = False
if 'smooth_radius' not in st.session_state:
    st.session_state.smooth_radius = 0.5
if 'lr_overlap' not in st.session_state:
    st.session_state.lr_overlap = 16
# New state for brightness/contrast
if 'apply_brightness_contrast' not in st.session_state:
    st.session_state.apply_brightness_contrast = False
if 'brightness_factor' not in st.session_state:
    st.session_state.brightness_factor = 1.0 # 1.0 means no change
if 'contrast_factor' not in st.session_state:
    st.session_state.contrast_factor = 1.0 # 1.0 means no change


@st.cache_resource
def load_generator_model(ckpt_path):
    """Loads the generator model from the specified checkpoint."""
    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint file not found at: {ckpt_path}")
        st.stop()

    st.write(f"⏳ Loading generator checkpoint: `{os.path.basename(ckpt_path)}`")
    try:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        gen = Generator(in_channels=3, img_channels=3).to(DEVICE)

        state_dict = None
        if "gen_state" in checkpoint:
            state_dict = checkpoint["gen_state"]
        else:
             st.error("Could not find a valid generator state_dict in the checkpoint file.")
             st.stop()

        if all(key.startswith('module.') for key in state_dict.keys()):
             state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        gen.load_state_dict(state_dict)
        gen.eval()
        st.success(f"✅ Generator loaded successfully!")
        return gen
    except Exception as e:
        st.error(f"❌ Error loading checkpoint: {e}")
        st.error("Ensure the checkpoint file is valid, matches the Generator architecture, and the correct state_dict key is present.")
        st.exception(e)
        st.stop()


def upscale_with_center_crop(pil_img, generator, lr_overlap=16):
    """
    Upscales a PIL image by 2x using the generator with chunking and center cropping.
    """
    step = STEP
    upscale_factor = UPSCALE_FACTOR

    lr_tensor = to_tensor(pil_img).unsqueeze(0).to(DEVICE)
    lr_tensor = normalize(lr_tensor.squeeze(0)).unsqueeze(0)


    _, _, H, W = lr_tensor.shape
    stride = LR_PATCH_SIZE - lr_overlap
    hr_overlap = lr_overlap * upscale_factor
    half_ov = hr_overlap // 2

    if stride <= 0:
        st.warning(f"Overlap ({lr_overlap}px) is too large for the patch size ({LR_PATCH_SIZE}px). Reducing overlap.")
        lr_overlap = LR_PATCH_SIZE - 8
        stride = LR_PATCH_SIZE - lr_overlap
        hr_overlap = lr_overlap * upscale_factor
        half_ov = hr_overlap // 2

    def get_pad(dim_size, patch_size, stride_size):
        """Calculate padding needed for a dimension."""
        if dim_size <= patch_size:
            return max(0, patch_size - dim_size)
        remainder = (dim_size - patch_size) % stride_size
        if remainder == 0:
            return 0
        else:
            return stride_size - remainder

    pad_h = get_pad(H, LR_PATCH_SIZE, stride)
    pad_w = get_pad(W, LR_PATCH_SIZE, stride)

    lr_tensor_padded = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H_pad, W_pad = lr_tensor_padded.shape

    out_h = H_pad * upscale_factor
    out_w = W_pad * upscale_factor
    sr_canvas = torch.zeros((1, 3, out_h, out_w), device=DEVICE)

    total_patches = len(range(0, H_pad - LR_PATCH_SIZE + 1, stride)) * \
                    len(range(0, W_pad - LR_PATCH_SIZE + 1, stride))
    pbar = st.progress(0, text="Processing patches...")
    patch_count = 0


    for i in range(0, H_pad - LR_PATCH_SIZE + 1, stride):
        for j in range(0, W_pad - LR_PATCH_SIZE + 1, stride):
            lr_patch = lr_tensor_padded[:, :, i:i+LR_PATCH_SIZE, j:j+LR_PATCH_SIZE]
            with torch.no_grad():
                sr_patch = generator(lr_patch, alpha=1.0, steps=step)

            sr_i = i * upscale_factor
            sr_j = j * upscale_factor

            top_crop = half_ov if i > 0 else 0
            left_crop = half_ov if j > 0 else 0
            bottom_crop = HR_PATCH_SIZE - half_ov if (i + LR_PATCH_SIZE) < H_pad else HR_PATCH_SIZE
            right_crop = HR_PATCH_SIZE - half_ov if (j + LR_PATCH_SIZE) < W_pad else HR_PATCH_SIZE


            top_paste = sr_i + half_ov if i > 0 else 0
            left_paste = sr_j + half_ov if j > 0 else 0
            paste_h = bottom_crop - top_crop
            paste_w = right_crop - left_crop
            bottom_paste = top_paste + paste_h
            right_paste = left_paste + paste_w

            sr_cropped = sr_patch[:, :, top_crop:bottom_crop, left_crop:right_crop]

            try:
                 sr_canvas[:, :, top_paste:bottom_paste, left_paste:right_paste] = sr_cropped
            except RuntimeError as e:
                 st.warning(f"Shape mismatch during patch paste (runtime error): {e}. Cropped: {sr_cropped.shape}, Canvas Slice: {sr_canvas[:, :, top_paste:bottom_paste, left_paste:right_paste].shape}. Skipping patch.")

            patch_count += 1
            progress_percent = min(1.0, patch_count / total_patches)
            pbar.progress(progress_percent, text=f"Processing patches... ({patch_count}/{total_patches})")


    pbar.empty()

    final_h = H * upscale_factor
    final_w = W * upscale_factor
    final_h = min(final_h, sr_canvas.shape[2])
    final_w = min(final_w, sr_canvas.shape[3])
    sr_canvas = sr_canvas[:, :, :final_h, :final_w]

    sr_canvas = (sr_canvas * 0.5) + 0.5
    sr_canvas = sr_canvas.clamp(0, 1)

    sr_pil = to_pil(sr_canvas.squeeze(0).cpu())
    return sr_pil


# --- Post-Processing Functions ---

def transfer_color_from_lr(sr_pil, lr_pil):
    """
    Transfers color from the LR image to the SR image using LAB color space.
    Takes Luminance (details) from SR and Color (a*, b*) from LR.
    """
    lr_resized_pil = lr_pil.resize(sr_pil.size, Image.Resampling.LANCZOS)

    sr_np_rgb = np.array(sr_pil)
    lr_resized_np_rgb = np.array(lr_resized_pil)

    if len(sr_np_rgb.shape) < 3:
        sr_np_rgb = cv2.cvtColor(sr_np_rgb, cv2.COLOR_GRAY2RGB)
    if len(lr_resized_np_rgb.shape) < 3:
        lr_resized_np_rgb = cv2.cvtColor(lr_resized_np_rgb, cv2.COLOR_GRAY2RGB)

    sr_np_bgr = cv2.cvtColor(sr_np_rgb, cv2.COLOR_RGB2BGR)
    lr_resized_np_bgr = cv2.cvtColor(lr_resized_np_rgb, cv2.COLOR_RGB2BGR)

    try:
        sr_lab = cv2.cvtColor(sr_np_bgr, cv2.COLOR_BGR2LAB)
        lr_resized_lab = cv2.cvtColor(lr_resized_np_bgr, cv2.COLOR_BGR2LAB)
    except cv2.error as e:
         st.error(f"OpenCV error during color space conversion: {e}")
         st.warning("Skipping color transfer.")
         return sr_pil


    sr_L, sr_a, sr_b = cv2.split(sr_lab)
    lr_resized_L, lr_resized_a, lr_resized_b = cv2.split(lr_resized_lab)

    final_lab = cv2.merge((sr_L, lr_resized_a, lr_resized_b))

    final_bgr = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

    processed_pil = Image.fromarray(final_rgb)

    return processed_pil


def apply_sharpening(image_pil, radius, percent, threshold):
    """Applies Unsharp Mask filter to a PIL image."""
    if not isinstance(image_pil, Image.Image):
         st.error("Sharpening function received invalid input (expected PIL Image).")
         return image_pil
    try:
        return image_pil.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    except Exception as e:
        st.error(f"Error applying sharpening: {e}")
        return image_pil


def apply_smoothing(image_pil, radius):
    """Applies Gaussian Blur filter to a PIL image."""
    if not isinstance(image_pil, Image.Image):
         st.error("Smoothing function received invalid input (expected PIL Image).")
         return image_pil
    try:
        return image_pil.filter(ImageFilter.GaussianBlur(radius=radius))
    except Exception as e:
        st.error(f"Error applying smoothing: {e}")
        return image_pil

def apply_brightness_contrast(image_pil, brightness_factor, contrast_factor):
    """Adjusts brightness and contrast of a PIL image."""
    if not isinstance(image_pil, Image.Image):
        st.error("Brightness/Contrast function received invalid input (expected PIL Image).")
        return image_pil
    try:
        img = image_pil
        # Apply brightness if factor is not neutral (1.0)
        if brightness_factor != 1.0:
            enhancer_brightness = ImageEnhance.Brightness(img)
            img = enhancer_brightness.enhance(brightness_factor)
        # Apply contrast if factor is not neutral (1.0)
        if contrast_factor != 1.0:
            enhancer_contrast = ImageEnhance.Contrast(img)
            img = enhancer_contrast.enhance(contrast_factor)
        return img
    except Exception as e:
        st.error(f"Error applying brightness/contrast: {e}")
        return image_pil


# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide", page_title="PixelSmith-SR")
    st.title("✨ PixelSmith-SR (2x Upscaler)")
    st.markdown("Upscale low-resolution images by 2x using a ProGAN-based model, transfer original color, and apply optional enhancements.")
    st.info(f"Using device: **{DEVICE.upper()}** | Upscale Factor: **{UPSCALE_FACTOR}x**")

    with st.sidebar:
        st.header("⚙️ Controls")

        st.subheader("Model Checkpoint")
        checkpoint_path_input = st.text_input("Generator Checkpoint Path", DEFAULT_CHECKPOINT_PATH, key="ckpt_path", help="Path to the pre-trained generator .pth file.")
        if not checkpoint_path_input:
            st.warning("Please provide a path to the generator checkpoint file.")
            st.stop()
        elif not os.path.exists(checkpoint_path_input):
             st.error(f"Checkpoint file not found at the specified path: `{checkpoint_path_input}`")
             st.stop()
        else:
            gen = load_generator_model(checkpoint_path_input)


        st.subheader("Image Upload")
        uploaded_file = st.file_uploader("Choose a low-resolution image", type=["png", "jpg", "jpeg"], key="uploader")

        st.subheader("Upscaling Settings")
        st.session_state.lr_overlap = st.slider(
            "Patch Overlap (pixels)",
            min_value=0, max_value=min(64, LR_PATCH_SIZE - 8),
            value=st.session_state.lr_overlap,
            step=4,
            key="overlap_slider",
            help=f"Adjusts the overlap between image patches (max {min(64, LR_PATCH_SIZE - 8)}px for {LR_PATCH_SIZE}px patches). Higher values can reduce seams but increase computation."
        )

        st.subheader("Post-Processing")
        with st.expander("Color Transfer (Recommended)", expanded=True):
             apply_color_transfer = st.checkbox("Transfer color from original LR image", value=True, key="color_transfer_cb", help="Merges details from the upscaled image with colors from the original image (resized). Helps maintain original color fidelity.")

        with st.expander("🔪 Sharpening"):
             st.session_state.sharpen = st.checkbox("Apply Sharpening", value=st.session_state.sharpen, key="sharpen_cb")
             if st.session_state.sharpen:
                st.session_state.sharpen_radius = st.slider("Radius", 0.1, 5.0, st.session_state.sharpen_radius, 0.1, key="sharp_radius_s", help="Controls the size of the edge area to affect.")
                st.session_state.sharpen_percent = st.slider("Percent (Strength)", 50, 300, st.session_state.sharpen_percent, 10, key="sharp_perc_s", help="How much contrast to add at edges (strength).")
                st.session_state.sharpen_threshold = st.slider("Threshold", 0, 20, st.session_state.sharpen_threshold, 1, key="sharp_thresh_s", help="Minimum brightness change to sharpen (ignores noise).")

        with st.expander("💧 Smoothing (Anti-aliasing)"):
             st.session_state.smooth = st.checkbox("Apply Gentle Smoothing", value=st.session_state.smooth, key="smooth_cb")
             if st.session_state.smooth:
                st.session_state.smooth_radius = st.slider("Radius (Strength)", 0.1, 3.0, st.session_state.smooth_radius, 0.1, key="smooth_radius_s", help="Controls the strength of the Gaussian blur for smoothing.")

        # New Brightness/Contrast Expander
        with st.expander("☀️ Brightness & Contrast"):
            st.session_state.apply_brightness_contrast = st.checkbox("Adjust Brightness/Contrast", value=st.session_state.apply_brightness_contrast, key="bc_cb")
            if st.session_state.apply_brightness_contrast:
                st.session_state.brightness_factor = st.slider(
                    "Brightness Factor",
                    min_value=0.5, max_value=1.5,
                    value=st.session_state.brightness_factor,
                    step=0.05,
                    key="brightness_slider",
                    help="Adjust overall brightness. 1.0 is original, <1.0 is darker, >1.0 is brighter."
                )
                st.session_state.contrast_factor = st.slider(
                    "Contrast Factor",
                    min_value=0.5, max_value=2.0,
                    value=st.session_state.contrast_factor,
                    step=0.05,
                    key="contrast_slider",
                    help="Adjust overall contrast. 1.0 is original, <1.0 reduces contrast, >1.0 increases contrast."
                )


    if uploaded_file is not None:
        try:
            lr_pil = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Failed to open or process the uploaded image: {e}")
            st.stop()


        st.subheader("Image Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Orignal")
            st.image(lr_pil, caption="Original Low-Resolution Image", use_container_width=True)
            st.write(f"Dimensions: {lr_pil.width}x{lr_pil.height}")

        sr_pil = None
        with col2:
            st.write("Upscaling...")
            with st.spinner(f"Upscaling image 2x (Overlap: {st.session_state.lr_overlap}px)..."):
                 try:
                    sr_pil = upscale_with_center_crop(lr_pil, gen, lr_overlap=st.session_state.lr_overlap)
                    st.image(sr_pil, caption="Raw Upscaled Image (2x)", use_container_width=True)
                    st.write(f"Dimensions: {sr_pil.width}x{sr_pil.height}")
                 except Exception as e:
                    st.error(f"An error occurred during upscaling:")
                    st.exception(e)

        final_pil = sr_pil
        post_processing_steps = []

        if final_pil:
            with col3:
                 st.write("Post-processing...")
                 with st.spinner("Applying post-processing steps..."):

                    if apply_color_transfer:
                        try:
                            final_pil = transfer_color_from_lr(final_pil, lr_pil)
                            post_processing_steps.append("Color Transfer")
                        except Exception as e:
                            st.error("Error during Color Transfer:")
                            st.exception(e)

                    if st.session_state.sharpen:
                        try:
                            final_pil = apply_sharpening(final_pil,
                                                         st.session_state.sharpen_radius,
                                                         st.session_state.sharpen_percent,
                                                         st.session_state.sharpen_threshold)
                            post_processing_steps.append("Sharpening")
                        except Exception as e:
                            st.error("Error during Sharpening:")
                            st.exception(e)

                    if st.session_state.smooth:
                         try:
                            final_pil = apply_smoothing(final_pil, st.session_state.smooth_radius)
                            post_processing_steps.append("Smoothing")
                         except Exception as e:
                             st.error("Error during Smoothing:")
                             st.exception(e)

                    # Apply Brightness/Contrast LAST
                    if st.session_state.apply_brightness_contrast:
                        try:
                            final_pil = apply_brightness_contrast(final_pil,
                                                                 st.session_state.brightness_factor,
                                                                 st.session_state.contrast_factor)
                            # Only add if factors are not 1.0 (neutral)? Or always add if checkbox is checked?
                            # Let's always add if the box is checked, even if factors are 1.0, for clarity.
                            post_processing_steps.append("Brightness/Contrast")
                        except Exception as e:
                            st.error("Error during Brightness/Contrast adjustment:")
                            st.exception(e)


                 caption_text = "Final Output"
                 if post_processing_steps:
                     caption_text += f" ({', '.join(post_processing_steps)})"
                 elif final_pil is sr_pil: # Check if final is same object as sr_pil (means nothing applied)
                     caption_text = "Raw Upscaled Image (No Post-Processing)"

                 st.image(final_pil, caption=caption_text, use_container_width=True)
                 st.write(f"Dimensions: {final_pil.width}x{final_pil.height}")

            if final_pil:
                buf = io.BytesIO()
                final_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="⬇️ Download Final Image",
                    data=byte_im,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_pixelsmith_final.png",
                    mime="image/png",
                    key="download_final"
                )

            # Offer raw download only if post-processing was actually applied
            if sr_pil and final_pil is not sr_pil:
                buf_raw = io.BytesIO()
                sr_pil.save(buf_raw, format="PNG")
                byte_im_raw = buf_raw.getvalue()
                st.download_button(
                    label="⬇️ Download Raw Upscaled Image",
                    data=byte_im_raw,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_pixelsmith_raw_sr.png",
                    mime="image/png",
                    key="download_raw"
                )


    else:
        st.info("⬅️ Upload an image using the sidebar controls to begin.")

if __name__ == "__main__":
    if not os.path.exists(DEFAULT_CHECKPOINT_PATH) and 'Generator' in globals():
        st.sidebar.warning(f"Default checkpoint '{DEFAULT_CHECKPOINT_PATH}' not found. Creating a dummy file for testing UI structure. Replace with your actual model checkpoint.")
        try:
            os.makedirs(os.path.dirname(DEFAULT_CHECKPOINT_PATH) or '.', exist_ok=True)
            dummy_gen = Generator(in_channels=3, img_channels=3)
            dummy_state = {"gen_state": dummy_gen.state_dict()}
            torch.save(dummy_state, DEFAULT_CHECKPOINT_PATH)
            st.sidebar.info("Dummy checkpoint created.")
        except Exception as e:
             st.sidebar.error(f"Could not create dummy checkpoint: {e}")
             try:
                 with open(DEFAULT_CHECKPOINT_PATH, 'w') as f: f.write("dummy")
             except Exception as fe:
                 st.sidebar.error(f"Could not create dummy file: {fe}")

    main()