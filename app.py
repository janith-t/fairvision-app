import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import base64
import io
from streamlit_cropper import st_cropper

# ── Process logo once (used for favicon + navbar) ──────────────
def _process_logo(path="logo.png"):
    try:
        img = Image.open(path).convert("RGBA")
        arr = np.array(img, dtype=np.int32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        saturation = (np.maximum(np.maximum(r, g), b) -
                      np.minimum(np.minimum(r, g), b))
        brightness = (r + g + b) / 3
        is_bg = (saturation < 18) & (brightness > 195)
        result = np.array(img)
        result[:, :, 3] = np.where(is_bg, 0, 255).astype(np.uint8)
        out = Image.fromarray(result, "RGBA")
        bbox = out.getbbox()
        if bbox:
            out = out.crop(bbox)
        return out
    except Exception:
        return None

_logo_pil = _process_logo()

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="FairVision — Age Group Classifier",
    page_icon=_logo_pil if _logo_pil else "🧠",
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
    display: flex; align-items: center; gap: 10px;
    font-family: 'DM Serif Display', serif;
    font-size: 28px; font-weight: 400; color: #111;
}
.nav-logo img { height: 48px; width: auto; display: block; }
.nav-logo > .wm { color: #111; }
.nav-logo > .wm > span { color: #5B4FE8; }
.nav-badge {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px; font-weight: 600;
    background: #5B4FE8; color: white;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.nav-links { display: flex; gap: 32px; }
.nav-links a {
    font-size: 16px; font-weight: 500; color: #1A1040;
    text-decoration: none; transition: color 0.2s;
}
.nav-links a:hover { color: #5B4FE8; }

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
    background: #1A1040; color: #A89FD8;
    padding: 28px 48px;
    display: flex; justify-content: space-between; align-items: center;
    font-size: 13px;
}
.site-footer .footer-left {
    display: flex; align-items: center; gap: 14px;
}
.site-footer .footer-left img {
    height: 38px; width: auto; display: block; opacity: 0.92;
}
.site-footer .footer-brand {
    display: flex; flex-direction: column; gap: 3px;
}
.site-footer .footer-name {
    font-family: 'DM Serif Display', serif;
    font-size: 17px; font-weight: 400; color: #E8E3FF;
    letter-spacing: -0.01em;
}
.site-footer .footer-meta {
    font-size: 12px; color: #7B6FC0; line-height: 1.4;
}
.site-footer .footer-right {
    text-align: right; font-size: 12px; color: #6B5FA8; line-height: 1.7;
}
.site-footer .footer-right strong { color: #A89FD8; font-weight: 500; }

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
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class FairVisionCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.stage1 = nn.Sequential(ResBlock(64, 128),  ResBlock(128, 128, stride=1))
        self.stage2 = nn.Sequential(ResBlock(128, 256), ResBlock(256, 256, stride=1))
        self.stage3 = nn.Sequential(ResBlock(256, 512), ResBlock(512, 512, stride=1))
        self.gap    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(self.gap(x))


# ── Load model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load(
            'fairvision_baseline.pth',
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


def crop_to_face_guide(image: Image.Image) -> Image.Image:
    """Crop the captured camera frame to the oval guide region.

    The guide oval sits at cx=50%, cy=46%, rx=26.25%, ry=40.7% of the frame.
    We take a square bounding that region so the model sees a tight face crop.
    """
    W, H = image.size
    cx = int(0.50 * W)
    cy = int(0.46 * H)
    half = int(min(0.265 * W, 0.415 * H))  # tightest axis of the oval
    left   = max(0, cx - half)
    top    = max(0, cy - half)
    right  = min(W, cx + half)
    bottom = min(H, cy + half)
    return image.crop((left, top, right, bottom))


model, checkpoint = load_model()

# ── Pre-load hero image as base64 ──────────────────────────────
def _load_b64(filename):
    try:
        with open(filename, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

# Build base64 src for navbar from the already-processed PIL image
def _pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

_logo_src = f'data:image/png;base64,{_pil_to_b64(_logo_pil)}' if _logo_pil else ''

# ══════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════
_logo_img_html = (
    f'<img src="{_logo_src}" alt="" '
    f'style="height:48px;width:auto;object-fit:contain;display:block;">'
    if _logo_src else ''
)
st.markdown(f"""
<div class="nav-bar">
  <div class="nav-logo">{_logo_img_html}<span class="wm">Fair<span>Vision</span></span></div>
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

_group_b64 = _load_b64("group.png")
if _group_b64:
    st.markdown(f"""
    <style>
    .hero-section {{
        background-image: url('data:image/png;base64,{_group_b64}') !important;
        background-size: cover;
        background-position: center 30%;
        min-height: 620px;
    }}
    .hero-section::before {{
        content: '';
        position: absolute;
        inset: 0;
        background: rgba(244, 244, 240, 0.80);
        z-index: 0;
    }}
    /* Only lift in-flow content above the overlay, not the absolute animation */
    .hero-section > div:not(.face-animation-wrap) {{
        position: relative;
        z-index: 1;
    }}
    /* Keep animation absolutely positioned and above overlay */
    .face-animation-wrap {{
        position: absolute !important;
        right: 48px;
        top: 60px;
        z-index: 1;
    }}
    </style>
    """, unsafe_allow_html=True)

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

# Custom SVG icons — blue line-art style
_icon_bias_detect = """
<svg width="54" height="54" viewBox="0 0 54 54" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs><clipPath id="bd-lens"><circle cx="39" cy="39" r="12"/></clipPath></defs>
  <!-- Three diverse person silhouettes (demographic groups) -->
  <circle cx="8"  cy="10" r="4.5" stroke="#5B4FE8" stroke-width="2.2"/>
  <path d="M3.5 22 Q8 16 12.5 22"  stroke="#5B4FE8" stroke-width="2.2" fill="none" stroke-linecap="round"/>
  <circle cx="22" cy="8"  r="5.5" stroke="#5B4FE8" stroke-width="2.2"/>
  <path d="M16.5 22 Q22 14 27.5 22" stroke="#5B4FE8" stroke-width="2.2" fill="none" stroke-linecap="round"/>
  <circle cx="36" cy="10" r="4.5" stroke="#5B4FE8" stroke-width="2.2"/>
  <path d="M31.5 22 Q36 16 40.5 22" stroke="#5B4FE8" stroke-width="2.2" fill="none" stroke-linecap="round"/>
  <!-- AI scan beam across the faces -->
  <line x1="2" y1="15" x2="43" y2="15" stroke="#5B4FE8" stroke-width="1.6" stroke-dasharray="4 3" opacity="0.55"/>
  <!-- Magnifying glass -->
  <circle cx="39" cy="39" r="12" stroke="#5B4FE8" stroke-width="2.8"/>
  <line x1="47.5" y1="47.5" x2="52.5" y2="52.5" stroke="#5B4FE8" stroke-width="2.8" stroke-linecap="round"/>
  <!-- Disparity bar chart clipped inside the lens -->
  <g clip-path="url(#bd-lens)">
    <rect x="29" y="43" width="4.5" height="8"  rx="1" fill="#5B4FE8"/>
    <rect x="35" y="35" width="4.5" height="16" rx="1" fill="#5B4FE8" opacity="0.65"/>
    <rect x="41" y="39" width="4.5" height="12" rx="1" fill="#5B4FE8" opacity="0.38"/>
    <line x1="27" y1="51" x2="51" y2="51" stroke="#5B4FE8" stroke-width="1.6" stroke-linecap="round"/>
  </g>
</svg>"""

_icon_bias_mitig = """
<svg width="54" height="54" viewBox="0 0 54 54" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Checkmark (fairness achieved) -->
  <path d="M19 8 L24 14 L35 4" stroke="#5B4FE8" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  <!-- Scale central pillar -->
  <line x1="27" y1="14" x2="27" y2="50" stroke="#5B4FE8" stroke-width="2.8" stroke-linecap="round"/>
  <line x1="16" y1="50" x2="38" y2="50" stroke="#5B4FE8" stroke-width="2.8" stroke-linecap="round"/>
  <!-- Scale arm (level = balanced) -->
  <line x1="6"  y1="22" x2="48" y2="22" stroke="#5B4FE8" stroke-width="2.8" stroke-linecap="round"/>
  <circle cx="27" cy="22" r="3.5" fill="#5B4FE8"/>
  <!-- Left pan -->
  <line x1="8" y1="22" x2="8" y2="35" stroke="#5B4FE8" stroke-width="2.2" stroke-linecap="round"/>
  <path d="M3 35 Q8 43 13 35" stroke="#5B4FE8" stroke-width="2.2" fill="none" stroke-linecap="round"/>
  <!-- Two equal person dots on left pan -->
  <circle cx="6.5"  cy="30.5" r="2.8" stroke="#5B4FE8" stroke-width="2" fill="#EEEAF8"/>
  <circle cx="12.5" cy="30.5" r="2.8" stroke="#5B4FE8" stroke-width="2" fill="#EEEAF8"/>
  <!-- Right pan -->
  <line x1="46" y1="22" x2="46" y2="35" stroke="#5B4FE8" stroke-width="2.2" stroke-linecap="round"/>
  <path d="M41 35 Q46 43 51 35" stroke="#5B4FE8" stroke-width="2.2" fill="none" stroke-linecap="round"/>
  <!-- Two equal person dots on right pan -->
  <circle cx="44.5" cy="30.5" r="2.8" stroke="#5B4FE8" stroke-width="2" fill="#EEEAF8"/>
  <circle cx="50.5" cy="30.5" r="2.8" stroke="#5B4FE8" stroke-width="2" fill="#EEEAF8"/>
</svg>"""

_icon_cnn = """
<svg width="54" height="54" viewBox="0 0 54 54" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Input: face pixel grid (image being classified) -->
  <rect x="1" y="15" width="15" height="15" rx="2.5" stroke="#5B4FE8" stroke-width="2.2"/>
  <rect x="4.5" y="18" width="3.5" height="3.5" fill="#5B4FE8" rx="0.6"/>
  <rect x="11"  y="18" width="3.5" height="3.5" fill="#5B4FE8" rx="0.6"/>
  <rect x="6.5" y="23.5" width="6" height="2.5" fill="#5B4FE8" rx="0.6"/>
  <!-- Arrow input → layers -->
  <line x1="17" y1="22.5" x2="21" y2="22.5" stroke="#5B4FE8" stroke-width="2" stroke-linecap="round"/>
  <path d="M19 20.5 L21.5 22.5 L19 24.5" stroke="#5B4FE8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  <!-- 3 stacked convolutional feature maps -->
  <rect x="23" y="9"  width="14" height="10" rx="2" stroke="#5B4FE8" stroke-width="2.2" fill="#EEEAF8"/>
  <rect x="23" y="22" width="14" height="10" rx="2" stroke="#5B4FE8" stroke-width="2.2" fill="#EEEAF8"/>
  <rect x="23" y="35" width="14" height="10" rx="2" stroke="#5B4FE8" stroke-width="2.2" fill="#EEEAF8"/>
  <!-- Activation dots in feature maps -->
  <circle cx="28.5" cy="14" r="1.8" fill="#5B4FE8"/>
  <circle cx="33.5" cy="14" r="1.8" fill="#5B4FE8" opacity="0.42"/>
  <circle cx="28.5" cy="27" r="1.8" fill="#5B4FE8" opacity="0.65"/>
  <circle cx="33.5" cy="27" r="1.8" fill="#5B4FE8" opacity="0.32"/>
  <circle cx="28.5" cy="40" r="1.8" fill="#5B4FE8" opacity="0.45"/>
  <!-- Connection lines → output nodes -->
  <line x1="37" y1="14" x2="42" y2="21" stroke="#5B4FE8" stroke-width="1.2" opacity="0.5"/>
  <line x1="37" y1="27" x2="42" y2="27" stroke="#5B4FE8" stroke-width="1.2" opacity="0.5"/>
  <line x1="37" y1="40" x2="42" y2="33" stroke="#5B4FE8" stroke-width="1.2" opacity="0.5"/>
  <!-- Output: 3 classification nodes -->
  <circle cx="47" cy="21" r="5.5" fill="#5B4FE8"/>
  <circle cx="47" cy="33" r="4.5" stroke="#5B4FE8" stroke-width="2.2" fill="#EEEAF8"/>
  <circle cx="47" cy="44" r="4.5" stroke="#5B4FE8" stroke-width="2.2" fill="#EEEAF8"/>
  <!-- Checkmark in top (predicted) output node -->
  <path d="M43.5 21 L46.5 24.5 L51.5 17" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>"""

st.markdown(f"""
<div class="features-strip">
  <div class="feature-card">
    <div class="feature-icon">{_icon_bias_detect}</div>
    <div class="feature-title">Bias Detection</div>
    <div class="feature-desc">Audits model performance across 7 race groups and 2 gender groups to surface hidden disparities.</div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">{_icon_bias_mitig}</div>
    <div class="feature-title">Bias Mitigation</div>
    <div class="feature-desc">Two mitigation strategies applied — class-weighted loss and oversampling via WeightedRandomSampler.</div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">{_icon_cnn}</div>
    <div class="feature-title">CNN From Scratch</div>
    <div class="feature-desc">ResNet-style CNN with residual connections, trained entirely from scratch on FairFace — no pretrained models used.</div>
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

# ── Session state ──────────────────────────────────────────────
if 'use_camera'      not in st.session_state: st.session_state.use_camera      = False
if 'confirmed_crop'  not in st.session_state: st.session_state.confirmed_crop  = None
if 'confirmed_for'   not in st.session_state: st.session_state.confirmed_for   = None

# ── Crop dialog (modal popup) ───────────────────────────────────
@st.dialog("Crop Face Area", width="small")
def _crop_dialog(raw_image, file_id, W, H):
    st.markdown(
        '<p style="font-size:12px;color:#888;margin:0 0 12px;text-align:center">'
        'Drag the box to frame the face · drag corners to resize</p>',
        unsafe_allow_html=True
    )

    # Fit image to the small dialog (~460 px usable width)
    _max_px = 460
    _scale  = min(1.0, _max_px / max(W, H))
    display_img = (
        raw_image.resize((int(W * _scale), int(H * _scale)), Image.LANCZOS)
        if _scale < 1.0 else raw_image
    )

    crop_box = st_cropper(
        display_img,
        realtime_update=True,
        box_color='#5B4FE8',
        aspect_ratio=(1, 1),
        return_type='box',
        key=f"dlg_{file_id}",
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Centered small submit button
    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        if st.button("Submit", type="primary", use_container_width=True):
            inv    = 1.0 / _scale
            left   = max(0, int(crop_box['left'] * inv))
            top    = max(0, int(crop_box['top']  * inv))
            right  = min(W, int((crop_box['left'] + crop_box['width'])  * inv))
            bottom = min(H, int((crop_box['top']  + crop_box['height']) * inv))
            cropped = raw_image.crop((left, top, right, bottom))
            _buf = io.BytesIO()
            cropped.save(_buf, format='PNG')
            st.session_state.confirmed_crop = _buf.getvalue()
            st.session_state.confirmed_for  = file_id
            st.rerun()

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
                st.components.v1.html("""<script>
(function(){
    var d=window.parent.document, ID='fv-face-guide';
    function inject(){
        var cam=d.querySelector('[data-testid="stCameraInput"]');
        if(!cam||cam.querySelector('#'+ID))return;
        cam.style.position='relative';
        var el=d.createElement('div');
        el.id=ID;
        el.style.cssText='position:absolute;top:0;left:0;right:0;bottom:56px;pointer-events:none;z-index:20;';
        el.innerHTML='<svg viewBox="0 0 400 300" width="100%" height="100%" fill="none" xmlns="http://www.w3.org/2000/svg">'
            /* face oval */
            +'<ellipse cx="200" cy="138" rx="105" ry="122" stroke="white" stroke-width="2.5" opacity="0.88"/>'
            /* top-left bracket */
            +'<path d="M50 45 L50 95 M50 45 L100 45" stroke="white" stroke-width="4.5" stroke-linecap="round"/>'
            /* top-right bracket */
            +'<path d="M350 45 L350 95 M350 45 L300 45" stroke="white" stroke-width="4.5" stroke-linecap="round"/>'
            /* bottom-left bracket */
            +'<path d="M50 245 L50 195 M50 245 L100 245" stroke="white" stroke-width="4.5" stroke-linecap="round"/>'
            /* bottom-right bracket */
            +'<path d="M350 245 L350 195 M350 245 L300 245" stroke="white" stroke-width="4.5" stroke-linecap="round"/>'
            /* label */
            +'<rect x="110" y="267" width="180" height="24" rx="12" fill="black" fill-opacity="0.45"/>'
            +'<text x="200" y="283" text-anchor="middle" fill="white" font-size="12" font-family="DM Sans,sans-serif" font-weight="500" opacity="0.95">Align face within oval</text>'
            +'</svg>';
        cam.appendChild(el);
    }
    new MutationObserver(inject).observe(d.body,{childList:true,subtree:true});
    [100,400,900].forEach(function(t){setTimeout(inject,t);});
})();
</script>""", height=0, scrolling=False)
                image_source = st.camera_input("Take a photo")
                if st.button("📁  Upload File Instead", use_container_width=True):
                    st.session_state.use_camera = False
                    st.rerun()
            if model is None and image_source is None:
                st.warning("⚠️ `fairvision_baseline.pth` not found. Place it in the same folder as `app.py` and restart.")

        image = None

        # ── Helpers ───────────────────────────────────────────────
        def _placeholder(msg):
            st.markdown(
                f'<div style="background:white;border:2px dashed #C4BFEF;border-radius:16px;'
                f'padding:32px;display:flex;align-items:center;justify-content:center;'
                f'min-height:200px;box-sizing:border-box;">'
                f'<span style="font-size:13px;color:#aaa;font-weight:500">{msg}</span></div>',
                unsafe_allow_html=True
            )

        def _show_preview(pil_img):
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(
                f'<div style="background:white;border:2px solid #C4BFEF;border-radius:16px;'
                f'padding:16px;display:flex;flex-direction:column;align-items:center;gap:8px;">'
                f'<p style="font-size:11px;font-weight:700;letter-spacing:0.09em;'
                f'text-transform:uppercase;color:#5B4FE8;margin:0">Cropped Preview</p>'
                f'<img src="data:image/png;base64,{b64}" style="max-width:100%;max-height:260px;'
                f'border-radius:10px;object-fit:contain;display:block;"></div>',
                unsafe_allow_html=True
            )

        # ── Main logic ────────────────────────────────────────────
        if image_source is not None:
            raw_image = Image.open(image_source).convert("RGB")
            W, H      = raw_image.size

            if st.session_state.use_camera:
                image = crop_to_face_guide(raw_image)
                with preview_col:
                    _show_preview(image)

            else:
                _file_id = f"{image_source.name}_{image_source.size}"

                # Reset crop state when a new image is uploaded
                if st.session_state.confirmed_for != _file_id:
                    st.session_state.confirmed_crop = None
                    st.session_state.confirmed_for  = _file_id

                if st.session_state.confirmed_crop:
                    image = Image.open(io.BytesIO(st.session_state.confirmed_crop))
                    with preview_col:
                        _show_preview(image)
                    with upload_col:
                        if st.button("✎  Re-crop", use_container_width=True):
                            st.session_state.confirmed_crop = None
                            st.rerun()
                else:
                    with preview_col:
                        _placeholder("Crop & submit to see preview")
                    # Open crop dialog automatically as soon as image is uploaded
                    _crop_dialog(raw_image, _file_id, W, H)

        else:
            with preview_col:
                _placeholder("Upload an image to crop" if not st.session_state.use_camera
                             else "Take a photo to analyse")

        if image is not None:
            if model is None:
                st.error("⚠️ Model file not found. Please ensure `fairvision_baseline.pth` is in the same folder as `app.py`.")
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
        A <strong style="color:white">ResNet-style CNN</strong> built from scratch in PyTorch.
        Stem: Conv7×7→BN→ReLU→MaxPool. Then 3 stages of 2 ResBlocks each,
        with skip connections for stable gradient flow.
        Global Average Pooling feeds into a FC head (512→256→9) with Dropout.
        11,162,185 trainable parameters. No pretrained weights used.
      </p>
    </div>
    <div class="info-block">
      <div class="info-block-title">Fairness Audit</div>
      <p>
        The baseline model showed a <strong style="color:white">6.20pp race accuracy gap</strong>
        (East Asian 59.61% best vs Black 53.41% worst) and an 8.36pp intersectional gap.
        Gender gap was 0.43pp. These findings motivated mitigation strategies.
      </p>
    </div>
    <div class="info-block">
      <div class="info-block-title">Bias Mitigation</div>
      <p>
        <strong style="color:white">Strategy 1</strong> — Class-weighted loss penalises errors on rare age groups.
        <strong style="color:white">Strategy 2</strong> — WeightedRandomSampler oversamples underrepresented
        race×age combinations. Strategy 1 achieves the best fairness: race gap reduced to 5.61pp and gender gap to 0.07pp.
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
    Three models were trained and compared. The deployed model is the Baseline
    — achieving 56.45% validation accuracy on the 9-class problem.
  </p>
</div>
""", unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        perf_data = {
            "Model"              : ["Baseline ✅", "Strategy 1 (Class Weights)", "Strategy 2 (Oversampling)"],
            "Accuracy"           : ["56.45%", "51.16%", "51.09%"],
            "Macro F1"           : ["0.5300", "0.4991", "0.5121"],
            "Race Gap ↓"         : ["6.20 pp", "5.61 pp", "6.25 pp"],
            "Gender Gap ↓"       : ["0.43 pp", "0.07 pp", "0.25 pp"],
            "Intersect. Gap ↓"   : ["8.36 pp", "6.17 pp", "8.32 pp"],
        }
        df_perf = __import__('pandas').DataFrame(perf_data)
        st.dataframe(df_perf, use_container_width=True, hide_index=True)

        st.markdown("""
        <p class="muted" style="margin-top:12px">
          ✅ Baseline is the deployed model (56.45% accuracy). Strategy 1 achieves the best
          fairness — lowest race gap (5.61 pp) and gender gap (0.07 pp) — but at a small
          accuracy cost. Strategy 2 improves macro F1 over Strategy 1 but does not reduce
          the race gap further. All gaps measured as max–min accuracy across demographic groups.
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
        achieves ~56% accuracy on a 9-class problem. Errors are expected and normal.
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
        The model struggles with ages 60–69 and 70+. The 70+ class has only 686 training
        samples vs 20,432 for the 20–29 class — a 29.8× imbalance that persists despite mitigation.
      </div>
    </div>
    <div class="limit-card">
      <div class="limit-card-title">🌍 Remaining demographic bias</div>
      <div class="limit-card-desc">
        The baseline shows a 6.20pp race accuracy gap (East Asian 59.61% vs Black 53.41%).
        Strategy 1 reduces this to 5.61pp. Continued improvement is needed across all groups.
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
_footer_logo_html = (
    f'<img src="{_logo_src}" alt="FairVision">'
    if _logo_src else ''
)
st.markdown(f"""
<div class="site-footer">
  <div class="footer-left">
    {_footer_logo_html}
    <div class="footer-brand">
      <span class="footer-name">FairVision</span>
      <span class="footer-meta">Janith Thiwanka &nbsp;·&nbsp; janith.tw@gmail.com<br>Senior Software Engineer</span>
    </div>
  </div>
  <div class="footer-right">
    <strong>Built with</strong><br>PyTorch &nbsp;·&nbsp; FairFace &nbsp;·&nbsp; Streamlit
  </div>
</div>
""", unsafe_allow_html=True)
