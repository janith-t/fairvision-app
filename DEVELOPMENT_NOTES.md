# FairVision — Development Notes

**Project:** FairVision Age Group Classification Web Application  
**Developer:** Janith Thiwanka (janith.tw@gmail.com)  
**Stack:** Python · PyTorch · Streamlit · PIL · NumPy  
**Deployed at:** Streamlit Cloud  

---

## 1. Project Overview

FairVision is a research prototype web application that demonstrates **responsible AI development** in computer vision. It classifies a human face image into one of 9 age groups using a custom CNN, while surfacing demographic fairness metrics across race and gender groups.

The application was built entirely in Python using the **Streamlit** framework, with a fully custom HTML/CSS front-end rendered inside Streamlit's `st.markdown(unsafe_allow_html=True)` pipeline — no external web framework was used.

---

## 2. Machine Learning Model

### Architecture — FairVisionCNN

A **ResNet-style CNN built from scratch in PyTorch** — no pretrained weights (e.g. ImageNet) were used at any stage.

```
Input (224×224 RGB)
  └── Stem: Conv7×7 → BatchNorm → ReLU → MaxPool
  └── Stage 1: ResBlock(64→128, stride=2)  + ResBlock(128→128, stride=1)
  └── Stage 2: ResBlock(128→256, stride=2) + ResBlock(256→256, stride=1)
  └── Stage 3: ResBlock(256→512, stride=2) + ResBlock(512→512, stride=1)
  └── Global Average Pooling (1×1)
  └── FC Head: Linear(512→256) → ReLU → Dropout(0.4) → Linear(256→9)
Output: 9-class softmax (age groups)
```

Each ResBlock uses a 1×1 shortcut convolution when dimensions change, giving stable gradient flow during training.

**Total trainable parameters:** 11,162,185  
**Input resolution:** 224×224 (ImageNet normalisation: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

### Dataset — FairFace

- **Source:** FairFace public dataset (config 0.25, tightly cropped 224×224)
- **Train / Test split:** 86,744 / 10,954 images
- **Labels:** 9 age groups, 7 race groups, 2 gender groups
- **Age imbalance:** 29.8× — 70+ class has only 686 training samples vs 20,432 for the 20–29 class

### Training — Three Models Compared

| Model | Strategy | Val Accuracy | Macro F1 | Race Gap | Gender Gap | Intersect. Gap |
|---|---|---|---|---|---|---|
| **Baseline** ✅ | Standard cross-entropy | **56.45%** | 0.5300 | 6.20 pp | 0.43 pp | 8.36 pp |
| Strategy 1 | Class-weighted loss | 51.16% | 0.4991 | **5.61 pp** | **0.07 pp** | **6.17 pp** |
| Strategy 2 | WeightedRandomSampler | 51.09% | 0.5121 | 6.25 pp | 0.25 pp | 8.32 pp |

- **Deployed model:** Baseline (highest accuracy)
- **Best fairness:** Strategy 1 — lowest race gap (5.61 pp), lowest gender gap (0.07 pp), lowest intersectional gap (6.17 pp)
- Fairness gaps are measured as max–min accuracy difference across demographic subgroups

### Fairness Audit — Baseline Model

- **Race accuracy gap:** 6.20 pp (East Asian 59.61% best → Black 53.41% worst)
- **Gender accuracy gap:** 0.43 pp
- **Intersectional gap:** 8.36 pp

---

## 3. Application Structure

```
fairvision-app/
├── app.py                   # Main Streamlit application
├── fairvision_baseline.pth  # Deployed model checkpoint
├── requirements.txt         # Python dependencies
├── logo.png                 # Brand logo (processed at runtime)
├── boy.jpg                  # Hero section face image
├── group.png                # Hero background photo
└── crop_component/          # Earlier custom component attempt (unused)
    └── index.html
```

---

## 4. Application Features Built

### 4.1 Custom Navbar

A sticky navigation bar rendered in raw HTML/CSS with:
- **Brand logo** loaded from `logo.png`, background removed programmatically (see §5.1), converted to base64 and embedded as a `data:` URI — no server file serving needed
- **"FairVision" wordmark** in DM Serif Display font, with "Vision" highlighted in brand purple (`#5B4FE8`)
- Navigation links (Try Demo · About · Responsible Use) anchored to page sections
- A pill badge ("CAME · IJSE") at the right
- Sticky + frosted-glass blur effect with `backdrop-filter: blur(12px)`

### 4.2 Hero Section

- Full-bleed background using `group.png` loaded as a base64 data URI and applied via inline CSS `background-image`
- Semi-transparent overlay (`rgba(244,244,240,0.80)`) to maintain readability
- Animated SVG face graphic on the right: rotating dashed ellipse scan ring, moving scan line, floating prediction badge — all pure SVG SMIL animation (no JavaScript)
- Hero stats row: 9 age groups · 97K training images · 7 race groups audited · 3 models compared

### 4.3 Feature Cards Strip

Three cards with custom SVG icons drawn from scratch:
1. **Bias Detection** — three face silhouettes with magnifying glass showing a bar chart
2. **Bias Mitigation** — balanced scales icon with equal group dots on each pan
3. **CNN From Scratch** — input face grid → convolutional feature maps → output classification nodes

### 4.4 Image Input — Two Modes

#### Upload Mode (with crop dialog)
1. User selects a file via `st.file_uploader`
2. A **modal crop dialog** opens automatically (`@st.dialog`) powered by `streamlit-cropper`
3. The cropper renders at display scale (max 460 px wide); on Submit the crop coordinates are scaled back to original full-resolution pixel coordinates
4. The cropped image is stored in `st.session_state.confirmed_crop` (bytes) so it survives reruns
5. A preview thumbnail is shown alongside a **Re-crop** button
6. Predictions only run after the crop is submitted — no accidental inference on the uncropped image

#### Camera Mode
1. User switches to `st.camera_input` via a toggle button
2. A JavaScript `MutationObserver` injects an SVG face-guide overlay directly into the camera widget DOM (oval outline + corner brackets + "Align face within oval" label)
3. On capture, the frame is cropped to the oval guide region (`crop_to_face_guide()`) before inference

### 4.5 Inference Pipeline

```python
Transform: Resize(224,224) → ToTensor → Normalize(ImageNet stats)
Model: FairVisionCNN.eval() with torch.no_grad()
Output: softmax probabilities → top-3 age groups displayed
```

Results are shown in a styled card with:
- Top prediction in large serif type
- Confidence percentage
- Animated progress bars for top 3 predictions (CSS transition on `width`)
- Ethical disclaimer below the results

### 4.6 About / Info Section

Dark-themed section (`#111` background) with four info blocks:
- Dataset description
- Model architecture summary
- Fairness audit findings (correct figures from notebook)
- Bias mitigation strategies

### 4.7 Model Performance Table

Rendered as a `st.dataframe` comparing all three models across Accuracy, Macro F1, Race Gap, Gender Gap, and Intersectional Gap. All figures verified against the Jupyter notebook.

### 4.8 Limitations & Responsible Use

Six limitation cards covering:
- Not for high-stakes decisions
- Probabilistic output only
- Binary gender labels in FairFace
- Older age group underperformance (686 samples for 70+)
- Remaining demographic bias (6.20 pp race gap in baseline)
- Research prototype status

### 4.9 Footer

Custom dark-purple footer (`#1A1040`) with:
- Processed logo + "FairVision" wordmark
- Developer name and email
- "Senior Software Engineer" role
- "Built with PyTorch · FairFace · Streamlit" on the right

---

## 5. Key Engineering Decisions

### 5.1 Logo Background Removal (Runtime, No External Tools)

`logo.png` had a neutral gray background (RGB ≈ 227,227,227) with all alpha channels set to 255 — it appeared to have transparency but did not. Fixed at startup using NumPy:

```python
saturation = max(R,G,B) - min(R,G,B)   # per-pixel
brightness  = (R+G+B) / 3
is_bg = (saturation < 18) & (brightness > 195)
result[:,:,3] = where(is_bg, 0, 255)    # make background transparent
out = out.crop(out.getbbox())           # trim transparent padding
```

The resulting PIL Image is used both as the Streamlit favicon (`st.set_page_config(page_icon=...)`) and embedded as a base64 `<img>` in the navbar and footer.

### 5.2 Crop Dialog — Why `streamlit-cropper` Over Custom Component

An initial attempt used `st.components.v1.declare_component` with a hand-written `index.html` crop UI. This failed because:
- Streamlit iframes start at 0 px height; `componentReady` timing was unreliable
- The `img_url` could not be a local file path on Streamlit Cloud

`streamlit-cropper` with `return_type='box'` solved this cleanly. The dialog scales the image to fit the modal, then scales the returned box coordinates back to original resolution for the actual crop.

### 5.3 Python Environment

- **Local:** App runs on system Python (`/Library/Frameworks/Python.framework/Versions/3.14/`) — the `venv/` directory had a broken interpreter path pointing to a deleted old directory
- **Cloud:** Standard Streamlit Cloud environment; all dependencies declared in `requirements.txt`
- `streamlit-cropper` was initially missing from `requirements.txt`, causing `ModuleNotFoundError` on first cloud deployment

### 5.4 Background Image Delivery

`group.png` and `boy.jpg` are read as binary files, base64-encoded, and injected as `data:` URIs directly into HTML. This avoids any static file serving configuration and works identically locally and on Streamlit Cloud.

### 5.5 Model Loading

The checkpoint is loaded with `weights_only=False` to allow the custom class dictionary stored alongside the weights:

```python
checkpoint = torch.load('fairvision_baseline.pth', map_location='cpu', weights_only=False)
model = FairVisionCNN(num_classes=checkpoint['num_classes'])
model.load_state_dict(checkpoint['model_state_dict'])
```

`@st.cache_resource` ensures the model is loaded only once per server session.

---

## 6. Bias Mitigation Strategies (Detail)

### Strategy 1 — Class-Weighted Loss
Assigns higher loss weight to underrepresented age classes (especially 70+, 0–2, 3–9). Forces the model to pay more attention to rare classes during backpropagation.  
**Result:** Best fairness metrics across all three measures (race, gender, intersectional gap) but ~5 pp accuracy drop vs baseline.

### Strategy 2 — WeightedRandomSampler (Oversampling)
Computes per-sample weights based on the combined race×age group, then uses PyTorch's `WeightedRandomSampler` to oversample underrepresented intersectional subgroups.  
**Result:** Macro F1 slightly higher than Strategy 1, but race and intersectional gaps do not improve beyond Strategy 1. Gender gap is 0.25 pp vs 0.07 pp for Strategy 1.

---

## 7. Statistics Audit

During development, several statistics displayed in the application were found to be incorrect and were corrected against the Jupyter notebook (`FairVision_Complete_Notebook.ipynb`):

| Field | Was (incorrect) | Corrected |
|---|---|---|
| Strategy 1 accuracy | 34.81% | 51.16% |
| Strategy 2 accuracy | 42.55% | 51.09% |
| Baseline Macro F1 | 0.3848 | 0.5300 |
| Strategy 1 Macro F1 | 0.3309 | 0.4991 |
| Strategy 2 Macro F1 | 0.4228 | 0.5121 |
| Race gap (baseline) | 8.97 pp | 6.20 pp |
| Race gap (strategy 1) | 8.93 pp | 5.61 pp |
| Race gap (strategy 2) | 7.61 pp | 6.25 pp |
| Gender gap (baseline) | 3.12 pp | 0.43 pp |
| Gender gap (strategy 1) | 3.12 pp | 0.07 pp |
| Gender gap (strategy 2) | 1.38 pp | 0.25 pp |
| Intersectional gap (baseline) | 12.19 pp | 8.36 pp |
| Intersectional gap (strategy 1) | 11.81 pp | 6.17 pp |
| Intersectional gap (strategy 2) | 10.19 pp | 8.32 pp |
| Race audit — worst group | White 46.19% | Black 53.41% |
| Race audit — best group | East Asian 55.16% | East Asian 59.61% |
| Age imbalance (70+ samples) | 842 vs 25,598 (30×) | 686 vs 20,432 (29.8×) |

---

## 8. Dependencies (`requirements.txt`)

```
streamlit>=1.32.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
streamlit-cropper>=0.3.0
```
