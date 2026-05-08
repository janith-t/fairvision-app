import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import base64
import io

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="FairVision — Age Group Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background: #F4F4F0;
    color: #111;
}

[data-testid="stAppViewContainer"] { padding: 0 !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
footer { display: none !important; }


/* ── Navbar ── */
.nav-bar {
    position: sticky; top: 0; z-index: 100;
    background: rgba(244,244,240,0.92);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #E0E0D8;
    padding: 0 48px;
    height: 64px;
    display: flex; align-items: center; justify-content: space-between;
}
.nav-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 22px; font-weight: 400; color: #111;
    display: flex; align-items: center; gap: 10px;
}
.nav-logo span { color: #5B4FE8; }
.nav-badge {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px; font-weight: 600;
    background: #5B4FE8; color: white;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.nav-links { display: flex; gap: 32px; }
.nav-links a {
    font-size: 14px; font-weight: 500; color: #555;
    text-decoration: none; transition: color 0.2s;
}
.nav-links a:hover { color: #111; }

/* ── Hero section ── */
.hero-section {
    background: #F4F4F0;
    padding: 80px 48px 0;
    min-height: 580px;
    position: relative;
    overflow: hidden;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 8px;
    background: #EEEAF8; color: #5B4FE8;
    font-size: 12px; font-weight: 600;
    padding: 6px 14px; border-radius: 20px;
    letter-spacing: 0.06em; text-transform: uppercase;
    margin-bottom: 28px;
}
.hero-tag::before {
    content: ''; width: 6px; height: 6px;
    background: #5B4FE8; border-radius: 50%;
    animation: pulse-dot 1.8s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { transform: scale(1); opacity:1; }
    50% { transform: scale(1.5); opacity:0.6; }
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(42px, 6vw, 72px);
    font-weight: 400; line-height: 1.08;
    color: #111; letter-spacing: -0.02em;
    margin-bottom: 24px;
}
.hero-title em { color: #5B4FE8; font-style: italic; }
.hero-subtitle {
    font-size: 17px; font-weight: 400;
    color: #555; line-height: 1.7;
    max-width: 520px; margin-bottom: 40px;
}
.hero-cta {
    display: inline-flex; align-items: center; gap: 10px;
    background: #5B4FE8; color: white;
    font-size: 15px; font-weight: 600;
    padding: 14px 28px; border-radius: 12px;
    text-decoration: none; cursor: pointer;
    transition: all 0.2s;
    border: none;
}
.hero-cta:hover { background: #4A3FD4; transform: translateY(-1px); }
.hero-stats {
    display: flex; gap: 40px;
    margin-top: 56px; padding-top: 40px;
    border-top: 1px solid #E0E0D8;
}
.stat-item { display: flex; flex-direction: column; gap: 4px; }
.stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 32px; color: #111; letter-spacing: -0.02em;
}
.stat-label { font-size: 13px; color: #777; font-weight: 400; }

/* ── Face animation container ── */
.face-animation-wrap {
    position: absolute; right: 48px; top: 40px;
    width: 420px; height: 500px;
}

/* ── Features strip ── */
.features-strip {
    background: #EEEAF8;
    padding: 48px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
    margin-top: 0;
}
.feature-card {
    background: white;
    border: 1px solid #E8E4F4;
    border-radius: 16px;
    padding: 28px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(91,79,232,0.12);
}
.feature-icon { font-size: 28px; margin-bottom: 14px; }
.feature-title { font-size: 15px; font-weight: 600; color: #111; margin-bottom: 8px; }
.feature-desc  { font-size: 13px; color: #666; line-height: 1.6; }

/* ── Upload section ── */
.upload-section { padding: 72px 48px; background: #F4F4F0; }
.section-label {
    font-size: 12px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #5B4FE8; margin-bottom: 12px;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(28px, 4vw, 42px);
    color: #111; margin-bottom: 16px; letter-spacing: -0.02em;
}
.section-desc { font-size: 15px; color: #666; line-height: 1.7; margin-bottom: 40px; max-width: 560px; }

/* ── Result cards ── */
.result-card {
    background: white;
    border: 1px solid #E8E4F4;
    border-radius: 20px;
    padding: 32px;
    margin-top: 32px;
}
.result-header {
    display: flex; align-items: center; gap: 12px; margin-bottom: 24px;
}
.result-badge {
    background: #EEEAF8; color: #5B4FE8;
    font-size: 11px; font-weight: 700;
    padding: 4px 12px; border-radius: 20px;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.top-prediction {
    font-family: 'DM Serif Display', serif;
    font-size: 36px; color: #111;
    margin-bottom: 4px; letter-spacing: -0.02em;
}
.top-confidence { font-size: 14px; color: #5B4FE8; font-weight: 600; }
.bar-row { margin: 16px 0; }
.bar-label {
    display: flex; justify-content: space-between;
    font-size: 13px; font-weight: 500; color: #333;
    margin-bottom: 6px;
}
.bar-bg {
    background: #F0EFF8; border-radius: 8px;
    height: 10px; overflow: hidden;
}
.bar-fill {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg, #5B4FE8, #8B83F0);
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}

/* ── Info / About section ── */
.info-section {
    background: #111; color: white;
    padding: 72px 48px;
}
.info-section .section-title { color: white; }
.info-section .section-desc  { color: #aaa; }
.info-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 32px; margin-top: 40px;
}
.info-block { padding: 28px; background: #1A1A1A; border-radius: 16px; border: 1px solid #2A2A2A; }
.info-block-title { font-size: 14px; font-weight: 700; color: #5B4FE8; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
.info-block p { font-size: 14px; color: #aaa; line-height: 1.7; }

/* ── Limitations section ── */
.limit-section {
    padding: 72px 48px;
    background: #F4F4F0;
}
.limit-grid {
    display: grid; grid-template-columns: repeat(2,1fr);
    gap: 20px; margin-top: 32px;
}
.limit-card {
    background: white; border: 1px solid #E8E4F4;
    border-radius: 16px; padding: 24px;
    border-left: 4px solid #5B4FE8;
}
.limit-card-title { font-size: 14px; font-weight: 600; color: #111; margin-bottom: 8px; }
.limit-card-desc  { font-size: 13px; color: #666; line-height: 1.6; }

/* ── Footer ── */
.site-footer {
    background: #0A0A0A; color: #555;
    padding: 32px 48px;
    display: flex; justify-content: space-between; align-items: center;
    font-size: 13px;
}
.site-footer span { color: #333; }

/* ── Streamlit overrides ── */
[data-testid="stFileUploader"] {
    background: white !important;
    border: 2px dashed #C4BFEF !important;
    border-radius: 16px !important;
    padding: 32px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #5B4FE8 !important; }
[data-testid="stImage"] img { border-radius: 16px; }
div[data-testid="stButton"] > button {
    background: #5B4FE8 !important; color: white !important;
    border: none !important; border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 15px !important;
    padding: 12px 28px !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    background: #4A3FD4 !important;
    transform: translateY(-1px) !important;
}
.tag {
    display: inline-block;
    background: #F0EFF8; color: #5B4FE8;
    font-size: 12px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px;
    margin: 4px 4px 4px 0;
}
.muted { color: #777; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ── CNN Model definition (must match training) ─────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
    def forward(self, x): return self.block(x)

class FairVisionCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32), ConvBlock(32, 64),
            ConvBlock(64, 128), ConvBlock(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


# ── Load model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load(
            'fairvision_best_model.pth',
            map_location=torch.device('cpu'),
            weights_only=False
        )
        model = FairVisionCNN(num_classes=checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint
    except FileNotFoundError:
        return None, None


# ── Inference ──────────────────────────────────────────────────
def predict(image: Image.Image, model, checkpoint):
    mean = checkpoint.get('imagenet_mean', [0.485, 0.456, 0.406])
    std  = checkpoint.get('imagenet_std',  [0.229, 0.224, 0.225])
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    tensor = transform(image.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    age_labels = checkpoint.get('age_labels',
        ['0-2','3-9','10-19','20-29','30-39','40-49','50-59','60-69','more than 70'])
    top3_idx   = np.argsort(probs)[::-1][:3]
    top3       = [(age_labels[i], float(probs[i])) for i in top3_idx]
    all_probs  = [(age_labels[i], float(probs[i])) for i in range(len(age_labels))]
    return top3, all_probs


model, checkpoint = load_model()

# ── Pre-load hero image as base64 ──────────────────────────────
def _load_b64(filename):
    try:
        with open(filename, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

# ══════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="nav-bar">
  <div class="nav-logo">🧠 Fair<span>Vision</span></div>
  <div class="nav-links">
    <a href="#demo">Try Demo</a>
    <a href="#about">About</a>
    <a href="#limitations">Responsible Use</a>
  </div>
  <div class="nav-badge">CAME · IJSE</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════
_boy_b64 = _load_b64("boy.jpg")
_boy_src = f'data:image/jpeg;base64,{_boy_b64}' if _boy_b64 else ''

hero_html = (
'<div class="hero-section">'
'<div style="max-width:580px;">'
'<div class="hero-tag">AI \u00b7 Fairness \u00b7 Face Analytics</div>'
'<h1 class="hero-title">Age group<br>classification<br><em>done fairly.</em></h1>'
'<p class="hero-subtitle">FairVision is a CNN-based age group classification system built on the FairFace dataset. Designed with bias detection and mitigation at its core \u2014 for responsible face analytics.</p>'
'<div class="hero-stats">'
'<div class="stat-item"><span class="stat-num">9</span><span class="stat-label">Age groups</span></div>'
'<div class="stat-item"><span class="stat-num">97K</span><span class="stat-label">Training images</span></div>'
'<div class="stat-item"><span class="stat-num">7</span><span class="stat-label">Race groups audited</span></div>'
'<div class="stat-item"><span class="stat-num">3</span><span class="stat-label">Models compared</span></div>'
'</div>'
'</div>'
'<div class="face-animation-wrap">'
'<svg viewBox="0 0 420 500" xmlns="http://www.w3.org/2000/svg" width="420" height="500">'
'<defs>'
'<radialGradient id="glowGrad" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="#5B4FE8" stop-opacity="0.2"/><stop offset="100%" stop-color="#5B4FE8" stop-opacity="0"/></radialGradient>'
'<clipPath id="ringClip">'
'<ellipse cx="210" cy="250" rx="168" ry="200">'
'<animateTransform attributeName="transform" type="rotate" from="0 210 250" to="360 210 250" dur="16s" repeatCount="indefinite"/>'
'</ellipse>'
'</clipPath>'
'</defs>'
'<ellipse cx="210" cy="250" rx="210" ry="240" fill="url(#glowGrad)"><animate attributeName="rx" values="210;220;210" dur="4s" repeatCount="indefinite"/><animate attributeName="ry" values="240;250;240" dur="4s" repeatCount="indefinite"/></ellipse>'
f'<image href="{_boy_src}" x="0" y="20" width="420" height="470" clip-path="url(#ringClip)" preserveAspectRatio="xMidYMid slice"/>'
'<ellipse cx="210" cy="250" rx="168" ry="200" fill="none" stroke="#5B4FE8" stroke-width="1.8" stroke-dasharray="12 7" opacity="0.7"><animateTransform attributeName="transform" type="rotate" from="0 210 250" to="360 210 250" dur="16s" repeatCount="indefinite"/></ellipse>'
'<ellipse cx="210" cy="250" rx="190" ry="222" fill="none" stroke="#8B83F0" stroke-width="1" stroke-dasharray="5 10" opacity="0.4"><animateTransform attributeName="transform" type="rotate" from="360 210 250" to="0 210 250" dur="10s" repeatCount="indefinite"/></ellipse>'
'<g opacity="0.25">'
'<line x1="42" y1="178" x2="378" y2="178" stroke="#5B4FE8" stroke-width="0.6" stroke-dasharray="3 5"/>'
'<line x1="42" y1="214" x2="378" y2="214" stroke="#5B4FE8" stroke-width="0.6" stroke-dasharray="3 5"/>'
'<line x1="42" y1="250" x2="378" y2="250" stroke="#5B4FE8" stroke-width="0.6" stroke-dasharray="3 5"/>'
'<line x1="42" y1="286" x2="378" y2="286" stroke="#5B4FE8" stroke-width="0.6" stroke-dasharray="3 5"/>'
'<line x1="42" y1="322" x2="378" y2="322" stroke="#5B4FE8" stroke-width="0.6" stroke-dasharray="3 5"/>'
'</g>'
'<line x1="42" y1="50" x2="378" y2="50" stroke="#5B4FE8" stroke-width="1.8" opacity="0.5"><animate attributeName="y1" values="50;460;50" dur="3.5s" repeatCount="indefinite" calcMode="linear"/><animate attributeName="y2" values="50;460;50" dur="3.5s" repeatCount="indefinite" calcMode="linear"/><animate attributeName="opacity" values="0.5;0.15;0.5" dur="3.5s" repeatCount="indefinite"/></line>'
'<g stroke="#5B4FE8" stroke-width="2.5" fill="none" opacity="0.85">'
'<path d="M42 50 L42 84 M42 50 L76 50"><animate attributeName="opacity" values="0.85;1;0.85" dur="2s" repeatCount="indefinite"/></path>'
'<path d="M378 50 L378 84 M378 50 L344 50"><animate attributeName="opacity" values="0.85;1;0.85" dur="2s" repeatCount="indefinite" begin="0.5s"/></path>'
'<path d="M42 460 L42 426 M42 460 L76 460"><animate attributeName="opacity" values="0.85;1;0.85" dur="2s" repeatCount="indefinite" begin="1s"/></path>'
'<path d="M378 460 L378 426 M378 460 L344 460"><animate attributeName="opacity" values="0.85;1;0.85" dur="2s" repeatCount="indefinite" begin="1.5s"/></path>'
'</g>'
'<g transform="translate(280, 60)">'
'<rect x="0" y="0" width="115" height="38" rx="10" fill="#5B4FE8" opacity="0.92"><animate attributeName="y" values="0;-6;0" dur="3s" repeatCount="indefinite"/></rect>'
'<text x="12" y="14" font-family="monospace" font-size="9" fill="white" opacity="0.7">PREDICTION</text>'
'<text x="12" y="28" font-family="monospace" font-size="11" fill="white" font-weight="bold">Age: 20\u201329 \u2713</text>'
'<animate attributeName="opacity" values="1;0.85;1" dur="3s" repeatCount="indefinite"/>'
'</g>'
'<g transform="translate(16, 300)">'
'<rect x="0" y="0" width="105" height="38" rx="10" fill="#1A1040" opacity="0.88"><animate attributeName="y" values="0;5;0" dur="3.5s" repeatCount="indefinite"/></rect>'
'<text x="10" y="14" font-family="monospace" font-size="9" fill="#8B83F0">CONFIDENCE</text>'
'<text x="10" y="28" font-family="monospace" font-size="11" fill="white" font-weight="bold">87.4%</text>'
'</g>'
'</svg>'
'</div>'
'</div>'
)
st.markdown(hero_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FEATURES STRIP
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="features-strip">
  <div class="feature-card">
    <div class="feature-icon">🔍</div>
    <div class="feature-title">Bias Detection</div>
    <div class="feature-desc">Audits model performance across 7 race groups and 2 gender groups to surface hidden disparities.</div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">⚖️</div>
    <div class="feature-title">Bias Mitigation</div>
    <div class="feature-desc">Two mitigation strategies applied — class-weighted loss and oversampling via WeightedRandomSampler.</div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🧠</div>
    <div class="feature-title">CNN From Scratch</div>
    <div class="feature-desc">4-block convolutional neural network trained entirely from scratch on FairFace — no pretrained models used.</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DEMO / UPLOAD SECTION
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="upload-section" id="demo">
  <div class="section-label">Live Demo</div>
  <h2 class="section-title">Upload a face image<br>and classify the age group</h2>
  <p class="section-desc">
    Upload any front-facing portrait photo. The model will return the top 3 predicted
    age groups with confidence scores. For best results use a clear, well-lit face image.
  </p>
</div>
""", unsafe_allow_html=True)

# Streamlit upload widget
if 'use_camera' not in st.session_state:
    st.session_state.use_camera = False

with st.container():
    pad_l, main_col, pad_r = st.columns([1, 6, 1])
    with main_col:
        upload_col, preview_col = st.columns([1, 1], gap="large")
        with upload_col:
            if not st.session_state.use_camera:
                image_source = st.file_uploader(
                    "Drop your image here or click to browse",
                    type=["jpg", "jpeg", "png", "webp"],
                    label_visibility="visible"
                )
                if st.button("📷  Use Camera Instead", use_container_width=True):
                    st.session_state.use_camera = True
                    st.rerun()
            else:
                image_source = st.camera_input("Take a photo")
                if st.button("📁  Upload File Instead", use_container_width=True):
                    st.session_state.use_camera = False
                    st.rerun()
            if model is None and image_source is None:
                st.warning("⚠️ `fairvision_best_model.pth` not found. Place it in the same folder as `app.py` and restart.")

        image = Image.open(image_source).convert("RGB") if image_source is not None else None

        with preview_col:
            if image is not None:
                buf = io.BytesIO()
                image.save(buf, format='PNG')
                b64 = base64.b64encode(buf.getvalue()).decode()
                st.markdown(
                    f'<div style="background:white;border:2px dashed #C4BFEF;border-radius:16px;padding:32px;display:flex;align-items:center;justify-content:center;height:100%;box-sizing:border-box;">'
                    f'<img src="data:image/png;base64,{b64}" style="max-width:100%;max-height:180px;border-radius:10px;object-fit:contain;display:block;">'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="background:white;border:2px dashed #C4BFEF;border-radius:16px;padding:32px;display:flex;align-items:center;justify-content:center;height:100%;box-sizing:border-box;min-height:160px;">'
                    '<span style="font-size:13px;color:#aaa;font-weight:500">Image preview</span>'
                    '</div>',
                    unsafe_allow_html=True
                )

        if image is not None:
            if model is None:
                st.error("⚠️ Model file not found. Please ensure `fairvision_best_model.pth` is in the same folder as `app.py`.")
            else:
                with st.spinner("Analysing..."):
                    top3, all_probs = predict(image, model, checkpoint)

                top1_label, top1_conf = top3[0]

                bars_html = ""
                for rank, (label, conf) in enumerate(top3):
                    pct = conf * 100
                    medal = ["🥇", "🥈", "🥉"][rank]
                    bars_html += (
                        f'<div class="bar-row">'
                        f'<div class="bar-label">'
                        f'<span>{medal} &nbsp;{label}</span>'
                        f'<span style="color:#5B4FE8;font-weight:600">{pct:.1f}%</span>'
                        f'</div>'
                        f'<div class="bar-bg">'
                        f'<div class="bar-fill" style="width:{pct}%"></div>'
                        f'</div>'
                        f'</div>'
                    )

                result_html = (
                    f'<div class="result-card">'
                    f'<div class="result-header"><span class="result-badge">Prediction Result</span></div>'
                    f'<div class="top-prediction">{top1_label}</div>'
                    f'<div class="top-confidence">Top prediction \u2014 {top1_conf*100:.1f}% confidence</div>'
                    f'<div style="margin-top:28px">'
                    f'<p style="font-size:13px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#999;margin-bottom:16px">Top 3 Predictions</p>'
                    f'{bars_html}'
                    f'</div>'
                    f'<p style="font-size:12px;color:#aaa;margin-top:24px;line-height:1.6">'
                    f'\u26a0\ufe0f This prediction is generated by a CNN trained on the FairFace dataset. '
                    f'Results are probabilistic and intended for research purposes only. '
                    f'Do not use for identity verification or high-stakes decisions.'
                    f'</p>'
                    f'</div>'
                )
                st.markdown(result_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ABOUT / SYSTEM DESCRIPTION
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="info-section" id="about">
  <div class="section-label" style="color:#8B83F0">About FairVision</div>
  <h2 class="section-title" style="color:white">How the system works</h2>
  <p class="section-desc">
    FairVision is a research prototype demonstrating responsible AI development
    practices in computer vision — combining performance evaluation with
    demographic fairness analysis.
  </p>
  <div class="info-grid">
    <div class="info-block">
      <div class="info-block-title">Dataset</div>
      <p>
        Built on <strong style="color:white">FairFace</strong> — a public dataset of 97,698 face images
        annotated with age, gender, and race. Specifically designed for balanced
        demographic evaluation. Config 0.25 (224×224 tightly cropped) used.
      </p>
    </div>
    <div class="info-block">
      <div class="info-block-title">Model Architecture</div>
      <p>
        A <strong style="color:white">4-block CNN</strong> built from scratch in PyTorch.
        Each block: Conv→BatchNorm→ReLU×2→MaxPool.
        Global Average Pooling feeds into a 3-layer FC head (512→256→9).
        1,438,441 trainable parameters. No pretrained weights used.
      </p>
    </div>
    <div class="info-block">
      <div class="info-block-title">Fairness Audit</div>
      <p>
        The baseline model showed a <strong style="color:white">8.97pp race accuracy gap</strong>
        (East Asian 55.16% vs White 46.19%) and a 12.19pp intersectional gap.
        Gender gap was 3.12pp. These findings motivated mitigation strategies.
      </p>
    </div>
    <div class="info-block">
      <div class="info-block-title">Bias Mitigation</div>
      <p>
        <strong style="color:white">Strategy 1</strong> — Class-weighted loss penalises errors on rare age groups.
        <strong style="color:white">Strategy 2</strong> — WeightedRandomSampler oversamples underrepresented
        race×age combinations. Strategy 2 achieved the best fairness: race gap reduced to 7.61pp.
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MODEL PERFORMANCE SUMMARY
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="limit-section">
  <div class="section-label">Model Performance</div>
  <h2 class="section-title">Baseline vs Mitigated Models</h2>
  <p class="section-desc">
    Three models were trained and compared. The deployed model is Strategy 2
    — chosen for achieving the best fairness metrics.
  </p>
</div>
""", unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        perf_data = {
            "Model"              : ["Baseline", "Strategy 1 (Class Weights)", "Strategy 2 (Oversampling) ✅"],
            "Accuracy"           : ["48.81%", "34.81%", "42.55%"],
            "Macro F1"           : ["0.3848", "0.3309", "0.4228"],
            "Race Gap ↓"         : ["8.97 pp", "8.93 pp", "7.61 pp"],
            "Gender Gap ↓"       : ["3.12 pp", "3.12 pp", "1.38 pp"],
            "Intersect. Gap ↓"   : ["12.19 pp", "11.81 pp", "10.19 pp"],
        }
        df_perf = __import__('pandas').DataFrame(perf_data)
        st.dataframe(df_perf, use_container_width=True, hide_index=True)

        st.markdown("""
        <p class="muted" style="margin-top:12px">
          ✅ Strategy 2 (WeightedRandomSampler) is the deployed model.
          It achieves the lowest race gap (7.61pp), lowest gender gap (1.38pp),
          and highest Macro F1 (0.4228) among all three models.
        </p>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LIMITATIONS & RESPONSIBLE USE
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="limit-section" id="limitations" style="background:#F0EFF8">
  <div class="section-label">Responsible Use</div>
  <h2 class="section-title">Limitations & ethical considerations</h2>
  <p class="section-desc">
    FairVision is a research prototype. Understanding its limitations is essential
    before any use of this system.
  </p>
  <div class="limit-grid">
    <div class="limit-card">
      <div class="limit-card-title">⚠️ Not for high-stakes decisions</div>
      <div class="limit-card-desc">
        This system must not be used for identity verification, surveillance,
        access control, hiring, law enforcement, or any decision that affects individuals.
      </div>
    </div>
    <div class="limit-card">
      <div class="limit-card-title">📊 Probabilistic output only</div>
      <div class="limit-card-desc">
        Predictions are statistical estimates — not ground truth. The model
        achieves ~42% accuracy on a 9-class problem. Errors are expected and normal.
      </div>
    </div>
    <div class="limit-card">
      <div class="limit-card-title">🔵 Binary gender labels</div>
      <div class="limit-card-desc">
        The FairFace dataset uses Male/Female labels only. Non-binary, gender-fluid,
        and other gender identities are not represented in the training data.
      </div>
    </div>
    <div class="limit-card">
      <div class="limit-card-title">👴 Older age groups underperform</div>
      <div class="limit-card-desc">
        The model struggles with ages 60–69 and 70+. The 70+ class has only 842 training
        samples vs 25,598 for the 20–29 class — a 30x imbalance that persists despite mitigation.
      </div>
    </div>
    <div class="limit-card">
      <div class="limit-card-title">🌍 Remaining demographic bias</div>
      <div class="limit-card-desc">
        Despite mitigation, a 7.61pp race accuracy gap remains. The system performs
        better on East Asian faces than other groups. Continued improvement is needed.
      </div>
    </div>
    <div class="limit-card">
      <div class="limit-card-title">🔬 Research prototype only</div>
      <div class="limit-card-desc">
        This system was developed as an academic project to demonstrate bias detection
        and mitigation techniques. It is not production-ready and has not undergone
        formal ethical review for real-world deployment.
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="site-footer">
  <span>🧠 FairVision &mdash; Certified AI & ML Engineer · IJSE · 2025/2026</span>
  <span>Built with PyTorch · FairFace · Streamlit</span>
</div>
""", unsafe_allow_html=True)
