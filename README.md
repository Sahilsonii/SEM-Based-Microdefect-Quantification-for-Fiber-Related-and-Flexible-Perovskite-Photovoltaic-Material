# SEM Annotation – Perovskite Defect Detection System

A complete end-to-end solution for detecting and classifying defects in perovskite solar cell materials using deep learning. The system combines manual annotation tools, automated labeling pipelines, YOLO model training infrastructure, and a production-ready web interface.

## 🎯 What This Project Does

This tool analyzes SEM (Scanning Electron Microscopy) images of perovskite materials to automatically detect and classify:
- **Defects**: PbI2 excess particles, pinholes in 3D perovskites, pinholes in 3D-2D mixed perovskites
- **Backgrounds**: Clean 3D and 3D-2D perovskite regions (non-defective areas)

## ✨ Key Features

- **🖊️ Interactive Annotation Interface**: Streamlit-based canvas for manual labeling with real-time preview
- **🤖 Auto-Annotation Pipelines**:
  - SAM (Segment Anything Model) - pixel-perfect segmentation
  - OpenCV - fast classical computer vision methods
  - YOLO inference - use trained models to label new images
- **📊 Dataset Management**: Stratified splitting, class balancing, and augmentation tools
- **⚡ Multi-Model Training**: Batch training for `yolov8s/m/l` and `yolo11s/m/l` variants
- **📈 Model Comparison Dashboard**: Interactive metrics visualization and model selection
- **🌐 Production Web App**: FastAPI backend + modern frontend for deployment
- **🔧 GPU-Accelerated**: Optimized for CUDA 11.8+ (PyTorch 2.7.1)

---

## 📁 Project Structure

```
SEM-Annotation/
├── app.py                          # Main Streamlit application (annotation + training UI)
├── requirements.txt                # Python dependencies (PyTorch CUDA 11.8)
│
├── src/                            # Core modules
│   ├── handlers/                   # Annotation tools
│   │   ├── canvas_ui.py           # Manual bounding box interface
│   │   ├── sam_handler.py         # SAM integration
│   │   ├── opencv_handler.py      # OpenCV auto-detection
│   │   └── utils.py               # File utilities
│   ├── model_handler.py            # YOLO training & inference
│   └── logger_setup.py             # Logging configuration
│
├── scripts/                        # Standalone automation tools
│   ├── train_all_gpu.py           # Batch model training (headless)
│   └── split_dataset.py           # Dataset splitting utilities
│
├── data/                           # Raw and processed data
│   ├── raw/                       # Original SEM images
│   ├── processed/                 # Pre-processed images
│   └── labels/                    # Manual annotations (excluded from git)
│
├── balanced_dataset/               # Production training dataset (git-ignored)
│   ├── images/                    # 440 balanced images (stratified)
│   ├── labels/                    # YOLO format (.txt) annotations
│   ├── data.yaml                  # YOLO dataset config
│   └── train/val/test.txt         # Split indices
│
├── runs/                           # Training outputs (git-ignored)
│   ├── detect/                    # Model checkpoints per run
│   └── metrics/                   # JSON evaluation metrics
│
├── web_app/                        # Production deployment
│   ├── backend/                   # FastAPI server
│   ├── frontend/                  # UI components
│   ├── run.bat                    # Windows launcher
│   └── requirements.txt           # Web app dependencies
│
├── experiments/                    # Research and exploration (git-ignored)
│   ├── dynamic_kernel/            # Experimental methods
│   └── logs/                      # Training logs
│
├── table_data_scripts/             # LaTeX table generation for papers
└── reports/                        # Documentation and presentations
```

### 🚫 What's Excluded from Git (.gitignore)

**Generated Data & Artifacts:**
- `balanced_dataset/` - auto-generated training sets
- `runs/detect/` - model weights and training outputs
- `runs/metrics/` - evaluation JSON files
- `labels/` - user-created annotations
- `env/` - Python virtual environment

**Large Model Files:**
- `*.pt` files (YOLO weights) - download automatically via Ultralytics
- Model checkpoints - regenerated during training

**Temporary/Experimental:**
- `experiments/` - research code and notebooks
- `__pycache__/`, `*.pyc` - Python bytecode
- `.ipynb_checkpoints` - Jupyter cache

> 💡 Only source code and configuration files are version-controlled. All datasets, models, and outputs are generated locally.

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python**: 3.10 or 3.11 (3.12+ may have compatibility issues)
- **GPU**: NVIDIA GPU with CUDA 11.8+ (for training)
- **Storage**: ~5GB for dependencies + datasets
- **OS**: Windows, Linux, or macOS

### Installation

#### 1. Clone Repository
```bash
git clone <repository-url>
cd SEM-Annotation
```

#### 2. Create Virtual Environment
```bash
python -m venv env
```

**Activate:**
```bash
# Windows
.\env\Scripts\activate

# Linux/Mac
source env/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **⚠️ GPU Users**: The `requirements.txt` is configured for **CUDA 11.8**. If you have a different CUDA version:
> 1. Install PyTorch first from [pytorch.org](https://pytorch.org/get-started/locally/) matching your CUDA version
> 2. Then install remaining requirements: `pip install streamlit pandas ultralytics streamlit-drawable-canvas opencv-python-headless pyyaml pillow`

#### 4. Verify Installation
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

---

## 🎯 How to Use

### Workflow Overview
1. **Annotate** images manually or with auto-tools
2. **Balance** dataset for optimal training
3. **Train** YOLO models (single or batch mode)
4. **Compare** model performance
5. **Deploy** best model to web app

---

### 1️⃣ Annotation Interface (Streamlit)

Launch the interactive UI:
```bash
streamlit run app.py
```

**Available Modes:**

#### 📝 Data Explorer & Labeling
- Draw bounding boxes manually on canvas
- Navigate through images with progress tracking
- Quick auto-annotation buttons (SAM, OpenCV)
- Per-folder organization (3D perovskite, 3D-2D mixed, etc.)
- Annotations saved to `labels/<class_folder>/`

#### 🤖 Auto-Annotation Inference
- Use trained YOLO models to label unlabeled images
- SAM integration for high-accuracy segmentation
- Adjustable confidence thresholds
- Batch processing across multiple folders

#### 🏋️ Train Model
- Train YOLO models directly from UI
- Choose balanced or original dataset
- Adjustable hyperparameters (epochs, batch size, image size)
- Real-time training progress in terminal

#### 📊 Model Comparison
- View metrics for all trained models
- Interactive charts (mAP, precision, recall)
- Identify best-performing model
- Load model weights for inference

---

### 2️⃣ Batch Training (Headless GPU Lab)

For automated multi-model training on dedicated GPU machines:

```bash
python scripts/train_all_gpu.py
```

**Process:**
1. Validates `dataset_split_224/` (224×224 patches, 5 classes)
2. Creates stratified 80/10/10 train/val/test split
3. Downloads missing YOLO weights automatically
4. Trains `yolo11s`, `yolo11m`, `yolo11l` sequentially (50 epochs each)
5. Saves checkpoints to `runs/detect/<model>_<timestamp>/weights/`
6. Exports metrics to `runs/metrics/<model>_<timestamp>.json`

**Output:**
- Best weights: `runs/detect/<model>/weights/best.pt`
- Training logs: `experiments/logs/training_<timestamp>.log`
- Metrics JSON: `runs/metrics/<model>.json`

---

### 3️⃣ Dataset Management

#### Create Balanced Dataset
```bash
python scripts/balance_dataset.py
```
Creates `balanced_dataset/` with:
- 80 PbI2 defect images
- 80 3D pinhole images
- 80 3D-2D pinhole images (augmented from 14)
- 100 3D background samples
- 100 3D-2D background samples
- **Total: 440 images** (54.5% defects, 45.5% backgrounds)

---

### 4️⃣ Production Deployment (Web App)

Launch the FastAPI web interface:

**Windows:**
```bash
cd web_app
run.bat
```

**Linux/Mac:**
```bash
cd web_app
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Access at: `http://localhost:8000`

---

## 📊 Understanding Model Outputs

### Training Results

After training completes, outputs are organized as:

```
runs/detect/<model>_<timestamp>/
├── weights/
│   ├── best.pt           # Best checkpoint (highest mAP)
│   └── last.pt           # Final epoch checkpoint
├── confusion_matrix.png  # Class prediction accuracy
├── results.csv           # Per-epoch metrics
└── args.yaml             # Training configuration
```

### Metrics Explained

| Metric | Meaning | Good Value |
|--------|---------|------------|
| **mAP50** | Mean Average Precision @ IoU=0.5 | >70% |
| **mAP50-95** | mAP averaged across IoU 0.5-0.95 | >50% |
| **Precision** | Correct detections / all detections | >80% |
| **Recall** | Detected objects / total objects | >75% |
| **F1 Score** | Harmonic mean of precision & recall | >75% |

### Model Comparison Dashboard

1. `streamlit run app.py` → **Model Comparison** tab
2. View side-by-side metrics for all trained models
3. Best values highlighted in green
4. Click "Load best model" to use for inference

---

## 🔧 Development & Customization

### Project Architecture

**Modular Design:**
- `src/handlers/` - annotation tools (reusable components)
- `src/model_handler.py` - YOLO training/inference wrapper
- `scripts/` - standalone automation (training pipelines)
- `web_app/` - production deployment (FastAPI + frontend)

**Import Pattern:**
```python
# From app.py or scripts/
from src.model_handler import ModelHandler
from src.handlers.canvas_ui import annotation_interface
from src.handlers.sam_handler import auto_annotate_with_sam
from src.handlers.opencv_handler import auto_annotate_with_opencv
```

### Adding New Auto-Annotation Methods

1. Create `src/handlers/my_method_handler.py`:
```python
def auto_annotate_with_mymethod(image_paths, labels_dir, **params):
    # Your detection logic
    # Save YOLO format .txt files to labels_dir
    return count  # number of images annotated
```

2. Import in `app.py` and add UI button:
```python
from src.handlers.my_method_handler import auto_annotate_with_mymethod

if st.button("My Method"):
    count = auto_annotate_with_mymethod([current_image], labels_dir)
```

### Training New YOLO Variants

Edit `scripts/train_all_gpu.py`:
```python
MODELS_TO_TRAIN = [
    {"label": "yolov9c", "weights": "yolov9c.pt"},  # Add new model
    {"label": "yolo11s", "weights": "yolo11s.pt"},
]
```

---

## 🧪 Perovskite Classes (5 Categories)

| Class ID | Name | Type | Description |
|----------|------|------|-------------|
| **0** | PbI2 Excess | Defect | Bright particles indicating lead iodide excess |
| **1** | 3D Pinholes | Defect | Dark holes/voids in 3D perovskite films |
| **2** | 3D-2D Pinholes | Defect | Dark voids in mixed-dimensional perovskites |
| **3** | 3D Background | Background | Clean 3D perovskite regions (no defects) |
| **4** | 3D-2D Background | Background | Clean 3D-2D mixed regions (no defects) |

**Dataset Composition:**
- **Defects (Classes 0-2)**: Critical for quality control
- **Backgrounds (Classes 3-4)**: Necessary to reduce false positives
- Balanced ratio prevents model bias toward any single class

---

## 🛠️ Technical Stack

| Component | Technology | Version |
|-----------|------------|----------|
| **Language** | Python | 3.10 / 3.11 |
| **Deep Learning** | PyTorch | 2.7.1+cu118 |
| **Object Detection** | Ultralytics YOLO | 8.4.7 |
| **Computer Vision** | OpenCV | Latest (headless) |
| **UI Framework** | Streamlit | 1.40.0 |
| **Web Backend** | FastAPI | Latest |
| **Annotation Tool** | streamlit-drawable-canvas | 0.9.3 |
| **GPU Acceleration** | CUDA | 11.8 |
| **Data Format** | YOLO txt | `class x_center y_center width height` |

**System Requirements:**
- **Minimum**: 8GB RAM, CPU-only (slow inference)
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **Optimal**: 32GB RAM, RTX 3060+ or better

---

## 📦 Important Notes

### First-Time Setup
1. Git tracks **only source code** - no datasets or models included
2. You must **manually add SEM images** to `data/raw/` folder
3. YOLO weights download **automatically** on first training run
4. `balanced_dataset/` is **generated** by `balance_dataset.py`

### Where Things Are Saved
- **Annotations**: `labels/<class_folder>/<image_name>.txt`
- **Model Weights**: `runs/detect/<model>_<timestamp>/weights/best.pt`
- **Training Logs**: `experiments/logs/training_<timestamp>.log`
- **Metrics**: `runs/metrics/<model>_<timestamp>.json`

### Git Tracking
✅ **Tracked**: Source code, configs, documentation  
❌ **Not Tracked**: Datasets, weights, annotations, outputs, virtual env

### Resuming Training
YOLO automatically resumes from last checkpoint if interrupted:
```bash
python scripts/train_all_gpu.py  # Skips already-trained models
```

---

## 🐛 Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce batch size in training settings (try `batch=2`)
- Lower image size (try `imgsz=640` instead of 1024)

**"No images found in folder"**
- Check that images are in `data/raw/<class_folder>/`
- Supported formats: `.jpg`, `.jpeg`, `.png`

**"Module not found" errors**
- Activate virtual environment: `.\env\Scripts\activate`
- Reinstall: `pip install -r requirements.txt`

**Training freezes on Windows**
- Set `workers=0` in training config (already default in `train_all_gpu.py`)

**SAM not working**
- SAM requires separate installation (not in default requirements)
- Follow instructions in `src/handlers/sam_handler.py`

---

## 📖 Additional Resources

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- [Ultralytics YOLOv8/v11](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [Meta's Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
