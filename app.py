import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import skfuzzy as fuzz
import io
import time
from scipy.ndimage import (binary_fill_holes, binary_erosion,
                            binary_dilation, binary_closing, label)

# ================================================================
#  PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="🧠 Fuzzy Tumor Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================================================
#  CSS
# ================================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        background-attachment: fixed;
    }
    .block-container { padding-top:2rem !important; padding-bottom:2rem !important; max-width:1200px !important; }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
    @keyframes fadeInUp { from{opacity:0;transform:translateY(30px);}to{opacity:1;transform:translateY(0);} }
    @keyframes fadeIn   { from{opacity:0;}to{opacity:1;} }
    @keyframes shimmer  { 0%{background-position:-200% center;}100%{background-position:200% center;} }
    @keyframes glow     { 0%,100%{opacity:0.5;}50%{opacity:1;} }
    @keyframes spin     { 0%{transform:rotate(0deg);}100%{transform:rotate(360deg);} }
    .main-header{text-align:center;padding:2.5rem 1rem 1.5rem 1rem;animation:fadeInUp .8s ease-out;}
    .main-header h1{font-size:2.8rem;font-weight:800;
        background:linear-gradient(135deg,#3B82F6,#22D3EE,#3B82F6);background-size:200% auto;
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        animation:shimmer 3s linear infinite;margin-bottom:.3rem;letter-spacing:-.02em;}
    .main-header p{color:rgba(255,255,255,.5);font-size:1.05rem;font-weight:300;letter-spacing:.05em;}
    .glass-card{background:rgba(255,255,255,.04);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
        border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:1.5rem;margin-bottom:1rem;
        animation:fadeInUp .8s ease-out;transition:all .3s cubic-bezier(.4,0,.2,1);}
    .glass-card:hover{background:rgba(255,255,255,.07);border-color:rgba(255,255,255,.15);
        transform:translateY(-2px);box-shadow:0 20px 40px rgba(0,0,0,.3);}
    .glass-card h3{color:rgba(255,255,255,.9);font-weight:600;font-size:1.05rem;
        margin-bottom:.8rem;display:flex;align-items:center;gap:.5rem;}
    .metric-card{background:rgba(255,255,255,.04);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
        border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:1.3rem 1.5rem;text-align:center;
        transition:all .3s cubic-bezier(.4,0,.2,1);animation:fadeInUp 1s ease-out;}
    .metric-card:hover{transform:translateY(-4px) scale(1.02);box-shadow:0 15px 35px rgba(0,0,0,.3);}
    .metric-card .metric-icon{font-size:1.8rem;margin-bottom:.4rem;}
    .metric-card .metric-value{font-size:1.8rem;font-weight:700;margin-bottom:.2rem;letter-spacing:-.02em;}
    .metric-card .metric-label{color:rgba(255,255,255,.5);font-size:.8rem;font-weight:400;
        text-transform:uppercase;letter-spacing:.1em;}
    .metric-tumor .metric-value{color:#EF4444;} .metric-boundary .metric-value{color:#F59E0B;}
    .metric-confidence .metric-value{color:#22D3EE;} .metric-dice .metric-value{color:#10B981;}
    .metric-tumor{border-left:3px solid #EF4444;} .metric-boundary{border-left:3px solid #F59E0B;}
    .metric-confidence{border-left:3px solid #22D3EE;} .metric-dice{border-left:3px solid #10B981;}
    .output-card{background:rgba(255,255,255,.04);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
        border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:1.2rem;
        transition:all .3s cubic-bezier(.4,0,.2,1);animation:fadeInUp 1.2s ease-out;overflow:hidden;}
    .output-card:hover{background:rgba(255,255,255,.07);transform:translateY(-3px);
        box-shadow:0 20px 40px rgba(0,0,0,.3);border-color:rgba(255,255,255,.15);}
    .output-card h4{color:rgba(255,255,255,.85);font-weight:600;font-size:.95rem;
        margin-bottom:.8rem;padding-bottom:.5rem;border-bottom:1px solid rgba(255,255,255,.06);}
    .status-bar{background:rgba(59,130,246,.08);backdrop-filter:blur(10px);
        border:1px solid rgba(59,130,246,.2);border-radius:14px;padding:1rem 1.5rem;
        text-align:center;color:rgba(255,255,255,.8);font-size:.95rem;margin:1rem 0;animation:fadeIn 1s ease-out;}
    .status-bar.warning{background:rgba(245,158,11,.08);border-color:rgba(245,158,11,.2);}
    .status-bar.success{background:rgba(16,185,129,.08);border-color:rgba(16,185,129,.2);}
    .processing{text-align:center;padding:2rem;animation:fadeIn .5s ease-out;}
    .processing .spinner{width:40px;height:40px;border:3px solid rgba(59,130,246,.2);
        border-top:3px solid #3B82F6;border-radius:50%;animation:spin 1s linear infinite;
        margin:0 auto 1rem auto;}
    .processing p{color:rgba(255,255,255,.6);font-size:.9rem;}
    .divider{height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.1),transparent);margin:1.5rem 0;}
    .section-title{color:rgba(255,255,255,.7);font-size:.85rem;font-weight:600;text-transform:uppercase;
        letter-spacing:.15em;margin-bottom:1rem;margin-top:.5rem;animation:fadeIn 1s ease-out;}
    .app-footer{text-align:center;color:rgba(255,255,255,.25);font-size:.75rem;
        padding:2rem 0 1rem 0;letter-spacing:.05em;}
    [data-testid="stFileUploader"]{background:transparent !important;}
    [data-testid="stFileUploader"] section{background:rgba(255,255,255,.03) !important;
        border:1px dashed rgba(59,130,246,.3) !important;border-radius:16px !important;padding:1.5rem !important;}
    [data-testid="stFileUploader"] section:hover{border-color:rgba(59,130,246,.5) !important;
        background:rgba(59,130,246,.05) !important;}
    .stSlider > div > div{background-color:rgba(59,130,246,.3) !important;}
    .stSlider > div > div > div > div{background-color:#3B82F6 !important;}
    [data-testid="stSidebar"]{background:rgba(15,23,42,.95) !important;backdrop-filter:blur(20px) !important;}
    .streamlit-expanderHeader{background:rgba(255,255,255,.03) !important;
        border-radius:12px !important;color:rgba(255,255,255,.8) !important;}
    .placeholder-container{text-align:center;padding:4rem 2rem;animation:fadeInUp 1s ease-out;}
    .placeholder-container .big-icon{font-size:5rem;margin-bottom:1.5rem;opacity:.3;animation:glow 3s ease-in-out infinite;}
    .placeholder-container h2{color:rgba(255,255,255,.4);font-weight:600;font-size:1.4rem;margin-bottom:.5rem;}
    .placeholder-container p{color:rgba(255,255,255,.25);font-size:.95rem;}
    </style>
    """, unsafe_allow_html=True)


# ================================================================
#  STEP 1 — PREPROCESSING
# ================================================================

def preprocess_image(uploaded_file):
    """Load, convert to grayscale, normalize to [0, 1]."""
    img = Image.open(uploaded_file).convert('L')
    img_array = np.array(img, dtype=np.float64)
    return img_array, img_array / 255.0


# ================================================================
#  STEP 2 — BRAIN MASKING
#
#  WHY this is needed:
#    Background pixels make up ~60-70% of any brain MRI image and
#    are very dark (intensity near 0). If we run FCM on the whole
#    image, one entire cluster is "wasted" modelling empty space,
#    and the dark background pulls centroids downward, making it
#    impossible to reliably separate tumor from normal brain.
#
#  HOW it works:
#    1. Otsu's threshold (pure numpy) finds the natural valley
#       between background and tissue in the intensity histogram.
#    2. We use 50% of that threshold (conservative) so we never
#       accidentally cut into dim brain tissue.
#    3. Morphological ops clean up the mask:
#         - erode ×1 → removes noise at boundaries
#         - dilate ×2 → restores brain margin
#         - fill_holes → covers ventricles (dark but inside brain)
# ================================================================

def _otsu_threshold(img):
    """Pure-numpy Otsu threshold — no skimage dependency."""
    hist, bin_edges = np.histogram(img.flatten(), bins=256, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total     = float(hist.sum())
    sum_total = float(np.dot(bin_centers, hist))
    best_var, best_thresh = 0.0, 0.0
    weight_b = sum_b = 0.0
    for h, bc in zip(hist, bin_centers):
        weight_b += h
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b  += bc * h
        mean_b  = sum_b / weight_b
        mean_f  = (sum_total - sum_b) / weight_f
        var     = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var > best_var:
            best_var, best_thresh = var, bc
    return best_thresh


def create_brain_mask(img_normalized):
    """
    Returns a boolean mask: True = brain pixel, False = background.
    Works on skull-stripped and non-skull-stripped MRI alike.
    """
    thresh = _otsu_threshold(img_normalized)
    # 50% of Otsu keeps all real brain tissue while removing dark background
    mask   = img_normalized > (thresh * 0.5)
    struct = np.ones((5, 5), dtype=bool)
    mask   = binary_erosion(mask,  structure=struct, iterations=1)
    mask   = binary_dilation(mask, structure=struct, iterations=2)
    mask   = binary_fill_holes(mask)
    return mask.astype(bool)


# ================================================================
#  STEP 3 — FCM + TUMOR CLUSTER IDENTIFICATION
#
#  Bug 1 fixed — wrong default cluster count:
#    n_clusters=2 with a whole-image input gives:
#      Cluster A: background   (centroid ≈ 0.05)
#      Cluster B: entire brain (centroid ≈ 0.45)
#    → argmax picks the entire brain as "tumor".
#    Now we run FCM on brain pixels only, so n_clusters=3 gives:
#      Cluster A: dark brain (CSF, sulci)
#      Cluster B: normal parenchyma
#      Cluster C: bright regions (tumor / edema)   ← correct
#
#  Bug 3 fixed — naive argmax cluster selection:
#    "Brightest cluster = tumor" fails on T1 MRI where white matter
#    is the brightest majority cluster (not tumor).
#    New rule: brightest cluster that is a MINORITY (<50% of brain).
#    Tumor is bright AND rare. Normal tissue dominates.
# ================================================================

def _identify_tumor_cluster(membership, centroids):
    """
    Pick the tumor cluster:
      1. Sort clusters brightest-first.
      2. Return the first (brightest) that covers < 50% of brain pixels.
      3. Fallback to absolute brightest if all are large (edge case).
    """
    centroid_vals = centroids.flatten()
    bright_first  = np.argsort(centroid_vals)[::-1]
    n_pixels      = membership.shape[1]
    cluster_props = membership.sum(axis=1) / n_pixels   # soft proportions

    for idx in bright_first:
        if cluster_props[idx] < 0.50:
            return int(idx)
    return int(bright_first[0])   # fallback


def apply_fcm(img_normalized, n_clusters=3, m=2.0, error=1e-5, maxiter=500):
    """
    Complete FCM pipeline with all fixes applied.

    Returns
    -------
    tumor_membership : ndarray (h, w)   membership in tumor cluster
    labels           : ndarray (h, w)   hard label per pixel
    centroids        : ndarray (n_clusters, 1)
    jm               : list             objective fn per iteration
    fpc              : float            fuzzy partition coefficient
    brain_mask       : ndarray (h, w)   bool, True = brain pixel
    """
    h, w = img_normalized.shape

    # ── Brain mask ────────────────────────────────────────────────────
    brain_mask   = create_brain_mask(img_normalized)
    brain_pixels = img_normalized[brain_mask]

    # Safety: if mask somehow empty, fall back to whole image
    if brain_pixels.size < n_clusters * 20:
        brain_mask   = np.ones((h, w), dtype=bool)
        brain_pixels = img_normalized.flatten()

    # ── FCM on brain pixels only ──────────────────────────────────────
    centroids, membership, _, _, jm, _, fpc = fuzz.cluster.cmeans(
        data=brain_pixels.reshape(1, -1),   # shape (1, N_brain)
        c=n_clusters,
        m=m,
        error=error,
        maxiter=maxiter,
        seed=42
    )
    # membership shape: (n_clusters, N_brain)

    # ── Identify tumor cluster ────────────────────────────────────────
    tumor_idx = _identify_tumor_cluster(membership, centroids)

    # ── Map results back to full (h×w) space ─────────────────────────
    brain_idx = np.where(brain_mask.flatten())[0]

    # Tumor membership: background pixels stay 0 (never tumor)
    full_mem = np.zeros(h * w, dtype=np.float64)
    full_mem[brain_idx] = membership[tumor_idx, :]
    tumor_membership = full_mem.reshape(h, w)

    # Label map: background gets label n_clusters (rendered as dark)
    full_labels = np.full(h * w, n_clusters, dtype=np.int32)
    full_labels[brain_idx] = np.argmax(membership, axis=0)
    labels = full_labels.reshape(h, w)

    return tumor_membership, labels, centroids, jm, fpc, brain_mask


# ================================================================
#  STEP 4 — REGION EXTRACTION + POST-PROCESSING
#
#  Bug 4 fixed — no morphological cleanup:
#    Raw FCM membership maps have salt-and-pepper noise (isolated
#    1-2 pixel "tumor" specks scattered across the whole image due to
#    MRI scanner intensity noise). These are not real tumor.
#
#    Fix:
#      1. Binary closing (3×3, 2 iters) — fills tiny internal gaps.
#      2. Connected component analysis — removes components smaller
#         than 0.1% of image area (generous; won't erase real tumors).
# ================================================================

def _postprocess_mask(mask_uint8, min_frac=0.001):
    """Remove noise speckles from a binary mask."""
    binary  = mask_uint8.astype(bool)
    struct  = np.ones((3, 3), dtype=bool)
    binary  = binary_closing(binary, structure=struct, iterations=2)
    min_px  = max(10, int(min_frac * binary.size))
    labeled_arr, n = label(binary)
    for cid in range(1, n + 1):
        if (labeled_arr == cid).sum() < min_px:
            binary[labeled_arr == cid] = False
    return binary.astype(np.uint8)


def extract_regions(membership_map, alpha=0.3, beta=0.7, brain_mask=None):
    """
    Classify each pixel into one of three clinical regions:
        µ  > β          → Tumor Core      (high-confidence tumor)
        α ≤ µ ≤ β       → Boundary Zone   (transitional / uncertain)
        µ  < α          → Normal Tissue   (high-confidence healthy)

    Post-processing removes noise from tumor and boundary masks.
    Normal-tissue mask is restricted to brain pixels only.
    """
    tumor_mask    = (membership_map > beta).astype(np.uint8)
    boundary_mask = ((membership_map >= alpha) &
                     (membership_map <= beta)).astype(np.uint8)
    normal_mask   = (membership_map < alpha).astype(np.uint8)

    tumor_mask    = _postprocess_mask(tumor_mask,    min_frac=0.001)
    boundary_mask = _postprocess_mask(boundary_mask, min_frac=0.0005)

    if brain_mask is not None:
        normal_mask[~brain_mask] = 0   # exclude background from "normal"

    return tumor_mask, boundary_mask, normal_mask


# ================================================================
#  VISUALIZATION
# ================================================================

def create_overlay(img_normalized, tumor_mask, boundary_mask):
    overlay = np.stack([img_normalized] * 3, axis=-1)
    tc = np.array([0.94, 0.27, 0.27])   # red
    bc = np.array([0.96, 0.62, 0.04])   # amber
    overlay[tumor_mask    == 1] = overlay[tumor_mask    == 1] * 0.3 + tc * 0.7
    overlay[boundary_mask == 1] = overlay[boundary_mask == 1] * 0.4 + bc * 0.6
    return np.clip(overlay, 0, 1)


def _fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#0F172A', edgecolor='none')
    buf.seek(0); img = Image.open(buf); plt.close(fig)
    return img


def render_grayscale(img_array):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_array, cmap='gray'); ax.axis('off')
    fig.patch.set_facecolor('#0F172A'); plt.tight_layout(pad=0.5)
    return _fig_to_pil(fig)


def render_cluster_map(labels, n_clusters, brain_mask):
    """Background pixels rendered dark so they don't masquerade as a tissue cluster."""
    display = labels.astype(float)
    display[~brain_mask] = 0   # background → darkest on viridis
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(display, cmap='viridis', vmin=0, vmax=max(n_clusters - 1, 1))
    ax.axis('off'); fig.patch.set_facecolor('#0F172A')
    plt.tight_layout(pad=0.5); return _fig_to_pil(fig)


def render_heatmap(membership_map):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(membership_map, cmap='jet', vmin=0, vmax=1); ax.axis('off')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors='white', labelsize=8)
    cb.set_label('Tumor Membership µ', color='white', fontsize=9)
    fig.patch.set_facecolor('#0F172A'); plt.tight_layout(pad=0.5)
    return _fig_to_pil(fig)


def render_overlay(overlay_img):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(overlay_img); ax.axis('off')
    fig.patch.set_facecolor('#0F172A'); plt.tight_layout(pad=0.5)
    return _fig_to_pil(fig)


def render_brain_mask(brain_mask):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(brain_mask.astype(float), cmap='Blues', vmin=0, vmax=1); ax.axis('off')
    fig.patch.set_facecolor('#0F172A'); plt.tight_layout(pad=0.5)
    return _fig_to_pil(fig)


# ================================================================
#  METRICS + STATUS
#
#  Bonus fix: metrics now relative to BRAIN area, not full image.
#  "Tumor = 5% of brain" is clinically meaningful.
#  "Tumor = 0.8% of image" (when 70% is background) is misleading.
# ================================================================

def compute_metrics(tumor_mask, boundary_mask, brain_mask):
    n_brain = max(int(brain_mask.sum()), 1)
    return (np.sum(tumor_mask) / n_brain * 100,
            np.sum(boundary_mask) / n_brain * 100)


def get_status_message(tumor_pct, boundary_pct):
    if tumor_pct < 1.0:
        return "✅ No significant tumor region detected.", "success"
    elif boundary_pct > tumor_pct * 1.5:
        return (f"⚠️ Diffuse tumor with large uncertain boundary. "
                f"Core: {tumor_pct:.1f}% · Boundary: {boundary_pct:.1f}% of brain — "
                f"suggests infiltrative margins."), "warning"
    elif boundary_pct > 3:
        return (f"🧠 Tumor detected with moderate boundary uncertainty. "
                f"Core: {tumor_pct:.1f}% · Boundary: {boundary_pct:.1f}% of brain."), ""
    else:
        return (f"🎯 Tumor detected with well-defined margins. "
                f"Core: {tumor_pct:.1f}% of brain tissue."), "success"


# ================================================================
#  MAIN
# ================================================================

def main():
    inject_css()

    st.markdown("""
    <div class="main-header">
        <h1>🧠 Fuzzy Tumor Detector</h1>
        <p>Brain-Masked FCM · Membership-Based Uncertainty Modeling</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<h3 style="color:rgba(255,255,255,.9);font-size:1.1rem;">⚙️ FCM Parameters</h3>',
                    unsafe_allow_html=True)

        n_clusters = st.slider(
            "Clusters within brain", 2, 6, 3,
            help="3 = CSF/dark · normal brain · tumor.  "
                 "Use 4–5 for richer separation. Background always excluded before clustering."
        )
        m_param   = st.slider("Fuzzification (m)", 1.1, 4.0, 2.0, 0.1,
                              help="Higher m = softer cluster transitions")
        alpha     = st.slider("Lower threshold α", 0.10, 0.49, 0.30, 0.05,
                              help="µ < α → Normal Tissue")
        beta      = st.slider("Upper threshold β", 0.51, 0.95, 0.70, 0.05,
                              help="µ > β → Tumor Core")
        show_mask = st.checkbox("Show brain mask panel", value=False)

        st.markdown('<div style="height:1px;background:rgba(255,255,255,.08);margin:1rem 0;"></div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="color:rgba(255,255,255,.35);font-size:.73rem;line-height:1.7;">
        <b>Fixed pipeline:</b><br>
        1. Otsu masking → brain only<br>
        2. FCM on brain pixels<br>
        3. Tumor = brightest minority cluster<br>
        4. Morphological cleanup<br>
        5. Metrics vs. brain area<br><br>
        <b>Zones:</b><br>
        µ > β → Tumor Core<br>
        α ≤ µ ≤ β → Boundary<br>
        µ < α → Normal Tissue
        </div>
        """, unsafe_allow_html=True)

    # ── Upload ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📤 Upload MRI Image</div>',
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a brain MRI image (PNG, JPG, TIF)",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        label_visibility="collapsed"
    )

    # ── Processing ───────────────────────────────────────────────────
    if uploaded_file is not None:
        with st.spinner(""):
            st.markdown("""
            <div class="processing">
                <div class="spinner"></div>
                <p>Otsu brain masking · FCM on brain pixels · Morphological cleanup…</p>
            </div>
            """, unsafe_allow_html=True)

            img_array, img_norm = preprocess_image(uploaded_file)

            (tumor_mem, labels,
             centroids, jm, fpc, brain_mask) = apply_fcm(
                img_norm, n_clusters=n_clusters, m=m_param
            )

            tumor_mask, boundary_mask, normal_mask = extract_regions(
                tumor_mem, alpha=alpha, beta=beta, brain_mask=brain_mask
            )

            overlay = create_overlay(img_norm, tumor_mask, boundary_mask)
            tumor_pct, boundary_pct = compute_metrics(tumor_mask, boundary_mask, brain_mask)
            normal_pct = max(0.0, 100.0 - tumor_pct - boundary_pct)
            time.sleep(0.4)

        # ── Status ───────────────────────────────────────────────────
        msg, msg_type = get_status_message(tumor_pct, boundary_pct)
        st.markdown(f'<div class="status-bar {msg_type}">{msg}</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Metrics ──────────────────────────────────────────────────
        st.markdown(
            '<div class="section-title">📊 Analysis Metrics (% of brain tissue)</div>',
            unsafe_allow_html=True
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card metric-tumor">
                <div class="metric-icon">🧠</div>
                <div class="metric-value">{tumor_pct:.1f}%</div>
                <div class="metric-label">Tumor Core</div></div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card metric-boundary">
                <div class="metric-icon">⚠️</div>
                <div class="metric-value">{boundary_pct:.1f}%</div>
                <div class="metric-label">Boundary Zone</div></div>""",
                unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card metric-confidence">
                <div class="metric-icon">📊</div>
                <div class="metric-value">{fpc:.3f}</div>
                <div class="metric-label">Fuzzy Partition Coeff.</div></div>""",
                unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card metric-dice">
                <div class="metric-icon">💚</div>
                <div class="metric-value">{normal_pct:.1f}%</div>
                <div class="metric-label">Normal Tissue</div></div>""",
                unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Output grid ───────────────────────────────────────────────
        st.markdown('<div class="section-title">🖼️ Segmentation Results</div>',
                    unsafe_allow_html=True)

        panels = [
            ("🔬 Original MRI Image",      render_grayscale(img_array)),
            ("🗺️ FCM Cluster Map",          render_cluster_map(labels, n_clusters, brain_mask)),
            ("🌡️ Tumor Membership Heatmap", render_heatmap(tumor_mem)),
            ("🎯 Boundary Map",             render_overlay(overlay)),
        ]
        if show_mask:
            panels.append(("🧩 Otsu Brain Mask", render_brain_mask(brain_mask)))

        for i in range(0, len(panels), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j >= len(panels):
                    break
                title, img = panels[i + j]
                with col:
                    h4 = (
                        '🎯 Boundary Map — '
                        '<span style="color:#EF4444">Tumor</span> | '
                        '<span style="color:#F59E0B">Boundary</span>'
                        if "Boundary Map" in title else title
                    )
                    st.markdown(
                        f'<div class="output-card"><h4>{h4}</h4></div>',
                        unsafe_allow_html=True
                    )
                    st.image(img, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Convergence + Scatter ─────────────────────────────────────
        st.markdown('<div class="section-title">📈 Convergence Analysis</div>',
                    unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)

        with cc1:
            st.markdown('<div class="glass-card"><h3>📉 Objective Function Jm</h3>',
                        unsafe_allow_html=True)
            fig_c, ax_c = plt.subplots(figsize=(6, 3.5))
            ax_c.plot(range(len(jm)), jm, color='#3B82F6', linewidth=2,
                      marker='o', markersize=2, markerfacecolor='#22D3EE')
            ax_c.set_xlabel('Iteration', color='white', fontsize=10)
            ax_c.set_ylabel('$J_m$', color='white', fontsize=10)
            ax_c.set_title('FCM Convergence', color='white', fontsize=12, fontweight='bold')
            ax_c.tick_params(colors='white', labelsize=8)
            ax_c.set_facecolor('#1E293B')
            ax_c.grid(True, alpha=0.15, color='white')
            for sp in ax_c.spines.values(): sp.set_color((1,1,1,0.1))
            fig_c.patch.set_facecolor('#0F172A'); plt.tight_layout()
            st.pyplot(fig_c); plt.close(fig_c)
            st.markdown('</div>', unsafe_allow_html=True)

        with cc2:
            st.markdown('<div class="glass-card"><h3>🔗 Membership vs. Intensity (brain pixels)</h3>',
                        unsafe_allow_html=True)
            fig_s, ax_s = plt.subplots(figsize=(6, 3.5))
            b_int = img_norm[brain_mask].flatten()
            b_mem = tumor_mem[brain_mask].flatten()
            sidx  = np.random.choice(len(b_int), size=min(4000, len(b_int)), replace=False)
            ax_s.scatter(b_int[sidx], b_mem[sidx], c=b_mem[sidx],
                         cmap='jet', alpha=0.4, s=3, edgecolors='none')
            ax_s.axhline(y=alpha, color='#F59E0B', linestyle='--',
                         linewidth=1.5, label=f'α = {alpha}')
            ax_s.axhline(y=beta,  color='#EF4444', linestyle='--',
                         linewidth=1.5, label=f'β = {beta}')
            ax_s.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax_s.set_xlabel('Pixel Intensity', color='white', fontsize=10)
            ax_s.set_ylabel('Tumor Membership µ', color='white', fontsize=10)
            ax_s.set_title('Brain pixels only', color='white', fontsize=11, fontweight='bold')
            ax_s.tick_params(colors='white', labelsize=8)
            ax_s.set_facecolor('#1E293B')
            ax_s.grid(True, alpha=0.15, color='white')
            ax_s.legend(fontsize=8, facecolor='#1E293B', edgecolor='none', labelcolor='white')
            for sp in ax_s.spines.values(): sp.set_color((1,1,1,0.1))
            fig_s.patch.set_facecolor('#0F172A'); plt.tight_layout()
            st.pyplot(fig_s); plt.close(fig_s)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Expander ──────────────────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        with st.expander("ℹ️  What was broken and how it was fixed"):
            st.markdown("""
            <div style="color:rgba(255,255,255,.7);font-size:.9rem;line-height:1.85;">
            <p><b>Bug 1 — n_clusters = 2 labeled entire brain as tumor.</b><br>
            With 2 clusters, FCM splits background vs. all brain tissue.
            argmax then picks the entire brain. Fixed: <b>default n_clusters = 3</b>
            (CSF/dark · normal brain · tumor) run on brain pixels only.</p>

            <p><b>Bug 2 — Background included in clustering.</b><br>
            ~60–70% of any MRI image is background. This wastes a cluster slot
            and pulls centroids toward zero. Fixed: <b>Otsu brain mask</b> computed
            first; FCM runs exclusively on extracted brain pixels.</p>

            <p><b>Bug 3 — argmax(centroids) picks the wrong cluster.</b><br>
            On T1 MRI, white matter is the brightest majority cluster — not tumor.
            Fixed: pick the <b>brightest non-dominant cluster</b> (&lt; 50% of brain pixels).
            Tumor is bright AND rare; normal tissue is bright AND dominant.</p>

            <p><b>Bug 4 — No post-processing on raw FCM output.</b><br>
            MRI scanner noise creates isolated 1–2 pixel "tumor" specks everywhere.
            Fixed: <b>morphological closing</b> fills gaps; connected component
            analysis removes components smaller than 0.1% of image area.</p>

            <p><b>Bonus — Metrics now relative to brain, not full image.</b><br>
            "Tumor = 0.8% of image" is meaningless when 70% is background.
            Now computed as <b>% of brain tissue area</b>.</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        # ── Placeholder ───────────────────────────────────────────────
        st.markdown("""
        <div class="placeholder-container">
            <div class="big-icon">🧠</div>
            <h2>Upload an MRI Image to Begin</h2>
            <p>Works on FLAIR · T1 · T2 · T1-CE · Any resolution · Skull-stripped or not</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""<div class="glass-card" style="text-align:center;">
                <h3 style="justify-content:center;">🧩 Brain Masking</h3>
                <p style="color:rgba(255,255,255,.45);font-size:.85rem;">
                Otsu threshold removes background before clustering —
                only real tissue enters FCM.</p></div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="glass-card" style="text-align:center;">
                <h3 style="justify-content:center;">🎯 Smart Cluster ID</h3>
                <p style="color:rgba(255,255,255,.45);font-size:.85rem;">
                Tumor = brightest minority cluster (&lt;50% of brain).
                Majority clusters are never labeled tumor.</p></div>""",
                unsafe_allow_html=True)
        with c3:
            st.markdown("""<div class="glass-card" style="text-align:center;">
                <h3 style="justify-content:center;">🔬 Noise Removal</h3>
                <p style="color:rgba(255,255,255,.45);font-size:.85rem;">
                Morphological closing + component filtering
                removes isolated noise speckles.</p></div>""",
                unsafe_allow_html=True)

    st.markdown("""
    <div class="app-footer">
        🧠 Fuzzy Tumor Detector · Brain-Masked FCM · Built with Streamlit & scikit-fuzzy
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()