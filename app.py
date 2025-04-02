# -*- coding: utf-8 -*-
import os
import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageEnhance
import hashlib

# --- Page Config ---
st.set_page_config(layout="wide", page_title="PixelSmith-SR")

# --- Model Imports ---
_old_model_import_error = None
_new_model_import_error = None
try:
    from Prod.newmodels import Generator as NewGenerator
    NEW_MODEL_AVAILABLE = True
except ImportError as e:
    _new_model_import_error = f"Could not import 'Generator' from 'Prod/newmodels.py'. Error: {e}"
    NEW_MODEL_AVAILABLE = False
    NewGenerator = None
try:
    from Prod.models import Generator
    OLD_MODEL_AVAILABLE = True
except ImportError as e:
    _old_model_import_error = f"Could not import 'Generator' from 'Prod/models.py'. Error: {e}"
    OLD_MODEL_AVAILABLE = False
    if 'Generator' not in locals() or Generator is None:
        Generator = None

# --- Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPSCALE_FACTOR = 2
LR_PATCH_SIZE = 128
HR_PATCH_SIZE = LR_PATCH_SIZE * UPSCALE_FACTOR
DEFAULT_CHECKPOINT_PATH = "./Prod/training_state.pth"
NEW_CHECKPOINT_PATH = "./Prod/new_training_state.pth"

# --- Model Config ---
MODEL_OPTIONS = {}
if NEW_MODEL_AVAILABLE:
    MODEL_OPTIONS["New Model"] = {
        "class": NewGenerator,
        "path": NEW_CHECKPOINT_PATH,
        "params": {"in_channels": 3, "img_channels": 3}
    }
if OLD_MODEL_AVAILABLE:
    MODEL_OPTIONS["Old Model"] = {
        "class": Generator,
        "path": DEFAULT_CHECKPOINT_PATH,
        "params": {"in_channels": 3, "img_channels": 3}
    }

# --- Initial Checks ---
if _old_model_import_error:
    st.error(_old_model_import_error)
if _new_model_import_error:
    st.error(_new_model_import_error)
if not MODEL_OPTIONS:
    st.error("No generator models configured. Cannot proceed.")
    st.stop()

# --- Transforms ---
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
normalize = T.Normalize(mean, std)
to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

# --- Session State ---
default_session_state = {
    'sharpen': False,
    'sharpen_radius': 1.5,
    'sharpen_percent': 150,
    'sharpen_threshold': 3,
    'smooth': False,
    'smooth_radius': 0.5,
    'lr_overlap': 4,
    'apply_brightness_contrast': False,
    'brightness_factor': 1.0,
    'contrast_factor': 1.0,
    'model_loaded_message_shown': ''
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Model Loading Cache ---
@st.cache_resource
def load_selected_generator_model(model_name: str, model_class: type, ckpt_path: str, model_params: dict):
    if not model_class:
        st.error(f"Model class for '{model_name}' not available.")
        st.stop()
    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint for '{model_name}' not found: `{ckpt_path}`")
        st.stop()
    loading_placeholder = st.sidebar.empty()
    loading_placeholder.info(f"⏳ Loading: **{model_name}** (`{os.path.basename(ckpt_path)}`)")
    try:
        gen = model_class(**model_params).to(DEVICE)
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = None
        potential_keys = ["gen_state", "state_dict", "generator", "g_state", "model"]
        if isinstance(checkpoint, dict):
            for key in potential_keys:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            if state_dict is None and all(isinstance(k, str) for k in checkpoint.keys()):
                state_dict = checkpoint
        elif hasattr(checkpoint, 'state_dict') and callable(checkpoint.state_dict):
            state_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, (dict, torch.Tensor)):
            state_dict = checkpoint
        if state_dict is None:
            loading_placeholder.error(f"Cannot find valid state_dict in checkpoint for {model_name}.")
            st.stop()
        if all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        gen.load_state_dict(state_dict)
        gen.eval()
        loading_placeholder.empty()
        return gen
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"❌ Error loading {model_name}:")
        st.exception(e)
        st.stop()

# --- Upscaling Internals ---
def _internal_upscale(pil_img, generator, lr_overlap, us=UPSCALE_FACTOR):
    lr_tensor = to_tensor(pil_img).unsqueeze(0).to(DEVICE)
    lr_tensor = normalize(lr_tensor.squeeze(0)).unsqueeze(0)
    _, _, H, W = lr_tensor.shape
    stride = LR_PATCH_SIZE - lr_overlap
    hr_overlap = lr_overlap * us
    half_ov = hr_overlap // 2
    if stride <= 0:
        lr_overlap = LR_PATCH_SIZE - 8
        stride = LR_PATCH_SIZE - lr_overlap
        hr_overlap = lr_overlap * us
        half_ov = hr_overlap // 2
    def get_pad(ds, ps, ss):
        return max(0, ps - ds) if ds <= ps else (0 if (ds - ps) % ss == 0 else ss - (ds - ps) % ss)
    pad_h = get_pad(H, LR_PATCH_SIZE, stride)
    pad_w = get_pad(W, LR_PATCH_SIZE, stride)
    lr_tensor_padded = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H_pad, W_pad = lr_tensor_padded.shape
    out_h = H_pad * us
    out_w = W_pad * us
    sr_canvas = torch.zeros((1, 3, out_h, out_w), device=DEVICE)
    nph = (H_pad - LR_PATCH_SIZE + stride) // stride if H_pad > LR_PATCH_SIZE else 1
    npw = (W_pad - LR_PATCH_SIZE + stride) // stride if W_pad > LR_PATCH_SIZE else 1
    total_patches = max(1, nph * npw)
    pbar_placeholder = st.empty()
    patch_count = 0
    for i in range(0, H_pad - LR_PATCH_SIZE + 1, stride):
        for j in range(0, W_pad - LR_PATCH_SIZE + 1, stride):
            if total_patches > 0 and patch_count % 5 == 0:
                prog = min(1.0, patch_count / total_patches)
                try:
                    pbar_placeholder.progress(prog, text=f"Processing... ({patch_count}/{total_patches})")
                except:
                    pass
            lr_patch = lr_tensor_padded[:, :, i:i+LR_PATCH_SIZE, j:j+LR_PATCH_SIZE]
            with torch.no_grad():
                try:
                    sr_patch = generator(lr_patch, alpha=1.0, steps=1)
                except TypeError as te:
                    st.error(f"Model ({generator.__class__.__name__}) inference TypeError patch ({i},{j}): {te}. Adapt _internal_upscale.")
                    pbar_placeholder.empty()
                    st.stop()
                except Exception as e:
                    st.error(f"Inference error patch ({i},{j}): {e}")
                    pbar_placeholder.empty()
                    st.stop()
            sr_i = i * us
            sr_j = j * us
            top_crop = half_ov if i > 0 else 0
            left_crop = half_ov if j > 0 else 0
            last_row = (i + stride) > (H_pad - LR_PATCH_SIZE)
            last_col = (j + stride) > (W_pad - LR_PATCH_SIZE)
            bottom_crop = HR_PATCH_SIZE - half_ov if not last_row else HR_PATCH_SIZE
            right_crop = HR_PATCH_SIZE - half_ov if not last_col else HR_PATCH_SIZE
            top_paste = sr_i + top_crop
            left_paste = sr_j + left_crop
            paste_h = bottom_crop - top_crop
            paste_w = right_crop - left_crop
            bottom_paste = top_paste + paste_h
            right_paste = left_paste + paste_w
            bottom_paste = min(bottom_paste, out_h)
            right_paste = min(right_paste, out_w)
            bottom_crop = min(bottom_crop, HR_PATCH_SIZE)
            right_crop = min(right_crop, HR_PATCH_SIZE)
            paste_h = bottom_paste - top_paste
            paste_w = right_paste - left_paste
            bottom_crop = top_crop + paste_h
            right_crop = left_crop + paste_w
            try:
                if paste_h > 0 and paste_w > 0 and (bottom_crop - top_crop) > 0 and (right_crop - left_crop) > 0:
                    sr_cropped = sr_patch[:, :, top_crop:bottom_crop, left_crop:right_crop]
                    target_slice = sr_canvas[:, :, top_paste:bottom_paste, left_paste:right_paste]
                    if sr_cropped.shape == target_slice.shape:
                        sr_canvas[:, :, top_paste:bottom_paste, left_paste:right_paste] = sr_cropped
            except Exception as e:
                st.warning(f"Patch paste error ({i},{j}): {e}. Skip.")
            patch_count += 1
    pbar_placeholder.empty()
    final_h = H * us
    final_w = W * us
    final_h = min(final_h, sr_canvas.shape[2])
    final_w = min(final_w, sr_canvas.shape[3])
    sr_canvas = sr_canvas[:, :, :final_h, :final_w]
    sr_canvas = (sr_canvas * 0.5) + 0.5
    sr_canvas = sr_canvas.clamp(0, 1)
    return to_pil(sr_canvas.squeeze(0).cpu())

# --- Upscaling Cache ---
@st.cache_data(show_spinner=False)
def cached_upscale_image(lr_img_bytes, _model_name, _lr_overlap):
    model_info = MODEL_OPTIONS[_model_name]
    generator = load_selected_generator_model(_model_name, model_info["class"], model_info["path"], model_info["params"])
    if generator is None:
        return None
    lr_pil = Image.open(io.BytesIO(lr_img_bytes)).convert("RGB")
    sr_pil = _internal_upscale(lr_pil, generator, _lr_overlap)
    if sr_pil:
        buf = io.BytesIO()
        sr_pil.save(buf, format="PNG")
        return buf.getvalue()
    return None

# --- Post-Processing Internals ---
def _internal_transfer_color(sr_pil, lr_pil):
    try:
        if sr_pil.size[0] == 0 or sr_pil.size[1] == 0:
            return sr_pil
        lr_resized = lr_pil.resize(sr_pil.size, Image.Resampling.LANCZOS)
        sr_rgb = np.array(sr_pil)
        lr_rgb = np.array(lr_resized)
        if len(sr_rgb.shape) < 3 or sr_rgb.shape[2] == 1:
            sr_rgb = cv2.cvtColor(sr_rgb, cv2.COLOR_GRAY2RGB)
        elif sr_rgb.shape[2] == 4:
            sr_rgb = cv2.cvtColor(sr_rgb, cv2.COLOR_RGBA2RGB)
        if len(lr_rgb.shape) < 3 or lr_rgb.shape[2] == 1:
            lr_rgb = cv2.cvtColor(lr_rgb, cv2.COLOR_GRAY2RGB)
        elif lr_rgb.shape[2] == 4:
            lr_rgb = cv2.cvtColor(lr_rgb, cv2.COLOR_RGBA2RGB)
        sr_bgr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)
        lr_bgr = cv2.cvtColor(lr_rgb, cv2.COLOR_RGB2BGR)
        sr_lab = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2LAB)
        lr_lab = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2LAB)
        sr_L, _, _ = cv2.split(sr_lab)
        _, lr_a, lr_b = cv2.split(lr_lab)
        final_lab = cv2.merge((sr_L, lr_a, lr_b))
        final_bgr = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)
        return Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))
    except Exception as e:
        st.error(f"Color transfer error: {e}. Skip.")
        return sr_pil

def _internal_apply_sharpen(img, r, p, t):
    return img.filter(ImageFilter.UnsharpMask(r, p, t)) if isinstance(img, Image.Image) else img

def _internal_apply_smooth(img, r):
    return img.filter(ImageFilter.GaussianBlur(r)) if isinstance(img, Image.Image) else img

def _internal_apply_bc(img, b, c):
    if not isinstance(img, Image.Image):
        return img
    try:
        if b != 1.0:
            img = ImageEnhance.Brightness(img).enhance(b)
        if c != 1.0:
            img = ImageEnhance.Contrast(img).enhance(c)
        return img
    except Exception as e:
        st.error(f"B/C error: {e}. Skip.")
        return img

# --- Post-Processing (No Caching) ---
def post_process_image(sr_img_bytes, lr_img_bytes, _apply_transfer, _sharpen_params, _smooth_params, _bc_params):
    if sr_img_bytes is None:
        return None, []
    final_pil = Image.open(io.BytesIO(sr_img_bytes)).convert("RGB")
    lr_pil = Image.open(io.BytesIO(lr_img_bytes)).convert("RGB")
    steps = []
    original_pil = final_pil
    if _apply_transfer:
        processed = _internal_transfer_color(final_pil, lr_pil)
        final_pil, steps = (processed, steps + ["Color Transfer"]) if processed is not final_pil else (final_pil, steps)
    if _sharpen_params['apply']:
        processed = _internal_apply_sharpen(final_pil, _sharpen_params['radius'], _sharpen_params['percent'], _sharpen_params['threshold'])
        final_pil, steps = (processed, steps + ["Sharpening"]) if processed is not final_pil else (final_pil, steps)
    if _smooth_params['apply']:
        processed = _internal_apply_smooth(final_pil, _smooth_params['radius'])
        final_pil, steps = (processed, steps + ["Smoothing"]) if processed is not final_pil else (final_pil, steps)
    if _bc_params['apply']:
        processed = _internal_apply_bc(final_pil, _bc_params['brightness'], _bc_params['contrast'])
    if _bc_params['apply'] and (processed is not final_pil or _bc_params['brightness'] != 1.0 or _bc_params['contrast'] != 1.0):
        final_pil, steps = (processed, steps + ["Brightness/Contrast"])
    if final_pil is original_pil and not steps:
        return sr_img_bytes, []
    buf = io.BytesIO()
    final_pil.save(buf, format="PNG")
    return buf.getvalue(), steps

# --- Model Change Callback ---
def handle_model_change():
    st.session_state.model_loaded_message_shown = ''
    cached_upscale_image.clear()

# --- Main App ---
def main():
    st.title("✨ PixelSmith-SR (2x Upscaler)")
    st.markdown("Upscale low-resolution images by 2x using a selected GAN model, transfer original color, and apply optional enhancements.")
    st.info(f"Using device: **{DEVICE.upper()}** | Upscale Factor: **{UPSCALE_FACTOR}x** | Patch Size: {LR_PATCH_SIZE}px")
    gen = None

    with st.sidebar:
        st.header("⚙️ Controls")
        st.subheader("Model Selection")
        available_names = list(MODEL_OPTIONS.keys())
        current_sel = st.session_state.get("model_select", available_names[0])
        selected_model = st.selectbox("Select Generator Model", options=available_names, key="model_select", index=available_names.index(current_sel), on_change=handle_model_change)
        model_info = MODEL_OPTIONS[selected_model]
        gen = load_selected_generator_model(selected_model, model_info["class"], model_info["path"], model_info["params"])
        if gen is not None and st.session_state.model_loaded_message_shown != selected_model:
            st.success(f"✅ {selected_model} loaded!")
            st.session_state.model_loaded_message_shown = selected_model
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader("Choose image", type=["png", "jpg", "jpeg", "webp"], key="uploader")
        st.subheader("Upscaling Settings")
        st.session_state.lr_overlap = st.slider("Patch Overlap (px)", 0, min(64, LR_PATCH_SIZE - 8), st.session_state.lr_overlap, 4, key="overlap_slider", help=f"Max rec: {min(64, LR_PATCH_SIZE - 8)}px")
        st.subheader("Post-Processing")
        with st.expander("Color Transfer", expanded=True):
            apply_tf = st.checkbox("Transfer color", value=True, key="color_transfer_cb")
        with st.expander("🔪 Sharpening"):
            st.session_state.sharpen = st.checkbox("Apply Sharpen", value=st.session_state.sharpen, key="sharpen_cb")
            if st.session_state.sharpen:
                st.session_state.sharpen_radius = st.slider("Radius", 0.1, 5.0, st.session_state.sharpen_radius, 0.1, key="sharp_radius_s")
                st.session_state.sharpen_percent = st.slider("Percent", 50, 300, st.session_state.sharpen_percent, 10, key="sharp_perc_s")
                st.session_state.sharpen_threshold = st.slider("Threshold", 0, 20, st.session_state.sharpen_threshold, 1, key="sharp_thresh_s")
        with st.expander("💧 Smoothing"):
            st.session_state.smooth = st.checkbox("Apply Smoothing", value=st.session_state.smooth, key="smooth_cb")
            if st.session_state.smooth:
                st.session_state.smooth_radius = st.slider("Radius ", 0.1, 3.0, st.session_state.smooth_radius, 0.1, key="smooth_radius_s")
        with st.expander("☀️ Brightness & Contrast"):
            st.session_state.apply_bc = st.checkbox("Adjust B/C", value=st.session_state.apply_brightness_contrast, key="bc_cb")
            if st.session_state.apply_bc:
                st.session_state.brightness_factor = st.slider("Brightness", 0.5, 1.5, st.session_state.brightness_factor, 0.05, key="brightness_slider")
                st.session_state.contrast_factor = st.slider("Contrast", 0.5, 2.0, st.session_state.contrast_factor, 0.05, key="contrast_slider")

    if uploaded_file is not None:
        if gen is None:
            st.error("Generator model not loaded.")
            st.stop()
        try:
            lr_bytes = uploaded_file.getvalue()
            lr_pil = Image.open(io.BytesIO(lr_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"Failed to load image: {e}")
            st.stop()
        st.subheader("Image Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original Input**")
            st.image(lr_pil, caption=f"Original ({lr_pil.width}x{lr_pil.height})", use_container_width=True)
        sr_bytes = None
        sr_pil = None
        with col2:
            st.markdown(f"**Raw Upscale ({selected_model})**")
            with st.spinner(f"Upscaling ({UPSCALE_FACTOR}x)..."):
                sr_bytes = cached_upscale_image(lr_bytes, selected_model, st.session_state.lr_overlap)
            if sr_bytes:
                try:
                    sr_pil = Image.open(io.BytesIO(sr_bytes)).convert("RGB")
                    st.image(sr_pil, caption=f"Raw Upscaled ({sr_pil.width}x{sr_pil.height})", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to display raw image: {e}")
                    sr_pil = None
            else:
                st.error("Upscaling failed.")
        final_bytes = None
        final_pil = None
        steps = []
        with col3:
            st.markdown("**Final Output**")
            if sr_bytes:
                sharp_p = {
                    'apply': st.session_state.sharpen,
                    'radius': st.session_state.sharpen_radius,
                    'percent': st.session_state.sharpen_percent,
                    'threshold': st.session_state.sharpen_threshold
                }
                smooth_p = {
                    'apply': st.session_state.smooth,
                    'radius': st.session_state.smooth_radius
                }
                bc_p = {
                    'apply': st.session_state.apply_bc,
                    'brightness': st.session_state.brightness_factor,
                    'contrast': st.session_state.contrast_factor
                }
                with st.spinner("Applying post-processing..."):
                    final_bytes, steps = post_process_image(sr_bytes, lr_bytes, apply_tf, sharp_p, smooth_p, bc_p)
                if final_bytes:
                    try:
                        final_pil = Image.open(io.BytesIO(final_bytes)).convert("RGB")
                        suffix = f"({final_pil.width}x{final_pil.height})"
                        cap = f"Final (No Post-Processing) {suffix}" if not steps else f"Final ({', '.join(steps)}) {suffix}"
                        st.image(final_pil, caption=cap, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to display final image: {e}")
                        final_pil = None
                else:
                    st.warning("Post-processing failed.")
                    final_pil = None
            else:
                st.markdown("<p style='text-align: center; color: grey;'><i>Raw upscale needed.</i></p>", unsafe_allow_html=True)
        st.markdown("---")
        dl1, dl2 = st.columns(2)
        with dl1:
            if final_bytes:
                model_id = selected_model.replace(" ", "").lower()
                fname = f"{os.path.splitext(uploaded_file.name)[0]}_pixelfinal_{model_id}.png"
                st.download_button(label="⬇️ Download Final", data=final_bytes, file_name=fname, mime="image/png", key="dl_final", use_container_width=True)
            else:
                st.download_button(label="⬇️ Download Final", data=b'', disabled=True, use_container_width=True)
        with dl2:
            diff = sr_bytes and final_bytes and (sr_bytes != final_bytes)
            if sr_bytes and diff:
                model_id = selected_model.replace(" ", "").lower()
                fname_raw = f"{os.path.splitext(uploaded_file.name)[0]}_pixelraw_{model_id}.png"
                st.download_button(label="⬇️ Download Raw", data=sr_bytes, file_name=fname_raw, mime="image/png", key="dl_raw", use_container_width=True)
            elif sr_bytes and not diff:
                st.markdown("<p style='text-align:center;color:grey;'><i>Raw==Final</i></p>", unsafe_allow_html=True)
            else:
                st.download_button(label="⬇️ Download Raw", data=b'', disabled=True, use_container_width=True)
    else:
        st.info("⬅️ Select model & upload image.")

def create_dummy_checkpoint(p, cls, params):
    if not cls:
        return False
    if not os.path.exists(p):
        ctx = st.sidebar
        ctx.warning(f"Chkpt '{os.path.basename(p)}' missing. Dummy.")
        try:
            os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
            dummy = cls(**params).cpu()
            state = {"gen_state": dummy.state_dict()}
            torch.save(state, p)
            return True
        except Exception as e:
            ctx.error(f"Dummy fail {os.path.basename(p)}:{e}")
        try:
            open(p, 'w').write("dummy")
            return True
        except Exception as fe:
            ctx.error(f"Dummy file fail:{fe}")
            return False
    return False

if __name__ == "__main__":
    dummy_created = False
    with st.sidebar:
        st.markdown("---")
        st.caption("Chkpt Status:")
    for name, info in MODEL_OPTIONS.items():
        if info.get('class', None):
            if create_dummy_checkpoint(info['path'], info['class'], info['params']):
                dummy_created = True
    main()
