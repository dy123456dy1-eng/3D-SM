# 3D-SM Code Documentation

## Overview

This repository contains a comprehensive set of tools for 3D-SM (Three-Dimensional Settling Map) analysis, including image processing, machine learning models, and statistical analysis tools.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Modules Description](#modules-description)
- [Dependencies](#dependencies)
- [Environment Configuration](#environment-configuration)

## System Requirements

### Hardware Requirements

- **GPU**: NVIDIA GeForce RTX 5080 Laptop GPU (or compatible NVIDIA GPU with CUDA support)
- **RAM**: Minimum 16GB recommended
- **Storage**: At least 50GB free space

### Software Requirements

- **Operating System**: Windows 11 (also compatible with Linux)
- **Python Version**: 3.9
- **CUDA Version**: 12.8
- **cuDNN Version**: 91002

## Installation

### Step 1: Clone or Download the Repository

```bash
# The code is located at: 3D-SM-code
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Project Structure

```
3D-SM-code/
├── 3D-SM Database/                      # Image database management system
│   ├── image_database_manager.py       # GUI application for managing 3D-SM images
│   ├── 3D-SM/                          # Image database folder
│   └── README_IMAGE_DATABASE.md        # Database documentation
│
├── 3D-SM principle analysis code/      # Mathematical analysis and principle explanations
│   ├── Extract codes for the three time intervals of 3D-SM/
│   │   └── improved_color_quantization.py  # Color bias calculation
│   ├── Mathematical analysis of 3D-SM detection of SVI30/
│   │   ├── partial_derivatives.py      # Partial derivative calculations
│   │   ├── regression_analysis.py      # Regression analysis for SVI30
│   │   └── regression_analysis_summary.md  # Analysis summary
│   ├── Mathematical analysis of 3D-SM detection of SV30/
│   │   ├── partial_derivatives.py
│   │   ├── regression_analysis.py      # Regression analysis for SV30
│   │   └── regression_analysis_summary.md
│   └── Mathematical analysis of 3D-SM detection of MLSS/
│       ├── partial_derivatives.py
│       ├── regression_analysis.py      # Regression analysis for MLSS
│       └── regression_analysis_summary.md
│
├── 3D-SM Analysis Code/                # Main analysis and prediction code
│   ├── 3D-SM Detection AS Performance Code/
│   │   ├── Filamentous bacteria enrichment/
│   │   │   └── src/
│   │   │       ├── config.py           # Configuration file
│   │   │       ├── data_loader.py      # Data loading module
│   │   │       ├── model_image_only.py # Neural network model
│   │   │       └── predict.py          # Prediction script
│   │   ├── SVI30/
│   │   │   └── src/
│   │   │       ├── config.py
│   │   │       ├── data_loader.py
│   │   │       ├── model_image_only.py
│   │   │       └── predict.py
│   │   ├── SV30/
│   │   │   └── src/
│   │   │       ├── config.py
│   │   │       ├── data_loader.py
│   │   │       ├── model_image_only.py
│   │   │       └── predict.py
│   │   └── MLSS/
│   │       └── src/
│   │           ├── config.py
│   │           ├── data_loader.py
│   │           ├── model_image_only.py
│   │           └── predict.py
│   └── 3D-SM Preprocessing/
│       └── process_images.py           # Image preprocessing utilities
│
├── Differential analysis/              # Statistical differential analysis
│   └── Differential testing analysis.py  # ANOVA and non-parametric tests
│
├── Adaptive image processing code (Provided a case)/
│   └── box process(windows).py         # Adaptive image processing
│
├── Automatic shooting code inside the device/
│   └── capture_photo_parallel1_30(rasp).py  # Automatic image capture
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Usage Guide

### 1. Image Database Management

Use the image database manager to view, filter, and query 3D-SM images:

```bash
cd "3D-SM Database"
python image_database_manager.py
```

This launches a GUI application that allows you to:
- Load image databases from Excel files
- Filter images by any label column
- Search and query images
- Preview images with detailed information

### 2. Statistical Analysis

#### Regression Analysis

Perform regression analysis for different parameters:

```bash
# For SVI30 analysis
cd "3D-SM principle analysis code/Mathematical analysis of 3D-SM detection of SVI30"
python regression_analysis.py

# For SV30 analysis
cd "3D-SM principle analysis code/Mathematical analysis of 3D-SM detection of SV30"
python regression_analysis.py

# For MLSS analysis
cd "3D-SM principle analysis code/Mathematical analysis of 3D-SM detection of MLSS"
python regression_analysis.py
```

#### Partial Derivative Analysis

Calculate partial derivatives of regression functions:

```bash
cd "3D-SM principle analysis code/Mathematical analysis of 3D-SM detection of SVI30"
python partial_derivatives.py
```

#### Differential Testing Analysis

Perform statistical differential analysis:

```bash
cd "Differential analysis"
python Differential testing analysis.py
```

This script performs:
- Normality tests (Shapiro-Wilk)
- Homogeneity tests (Levene's test)
- Parametric tests (ANOVA)
- Non-parametric tests (Kruskal-Wallis)
- Post-hoc multiple comparisons
- Data visualization

### 3. Machine Learning Prediction

#### Prediction for SVI30

```bash
cd "3D-SM Analysis Code/3D-SM Detection AS Performance Code/SVI30/src"
python predict.py
```

#### Prediction for SV30

```bash
cd "3D-SM Analysis Code/3D-SM Detection AS Performance Code/SV30/src"
python predict.py
```

#### Prediction for MLSS

```bash
cd "3D-SM Analysis Code/3D-SM Detection AS Performance Code/MLSS/src"
python predict.py
```

#### Prediction for Filamentous Bacteria Enrichment

```bash
cd "3D-SM Analysis Code/3D-SM Detection AS Performance Code/Filamentous bacteria enrichment/src"
python predict.py
```

### 4. Image Preprocessing

Process and split images for model training:

```bash
cd "3D-SM Analysis Code/3D-SM Preprocessing"
python process_images.py
```

This script:
- Splits images horizontally into 3 parts
- Resizes each part to 224x224
- Saves processed images for training

### 5. Color Quantization Analysis

Analyze color bias in images:

```bash
cd "3D-SM principle analysis code/Extract codes for the three time intervals of 3D-SM"
python improved_color_quantization.py
```

This script calculates red bias in images and generates quantitative results.

## Modules Description

### 3D-SM Database Module

**File**: `3D-SM Database/image_database_manager.py`

**Purpose**: GUI application for managing 3D-SM image database

**Features**:
- Excel-based label database management
- Image filtering by any column
- Search and query functionality
- Image preview with detailed information
- Pagination support for large datasets

**Key Functions**:
- `load_database()`: Load Excel database and initialize system
- `apply_filter()`: Filter images based on selected criteria
- `search_images()`: Search images across all columns
- `show_images()`: Display images with pagination

### Statistical Analysis Modules

**Files**: 
- `regression_analysis.py`
- `partial_derivatives.py`
- `Differential testing analysis.py`

**Purpose**: Mathematical and statistical analysis of 3D-SM data

**Features**:
- Multiple regression models (Linear, Ridge, Lasso, Polynomial)
- Partial derivative calculations
- Normality and homogeneity tests
- ANOVA and non-parametric tests
- Post-hoc multiple comparisons
- Data visualization

### Machine Learning Modules

**Files**: 
- `model_image_only.py`
- `data_loader.py`
- `predict.py`

**Purpose**: Deep learning-based prediction using image data

**Architecture**:
- **ResNet-18/34/101** with attention mechanism
- **Input**: 3 images per sample (224x224x3)
- **Output**: Regression values or classification labels
- **Features**: Spatial attention, transfer learning, regularization

**Tasks**:
- **SVI30**: Regression (predict SVI30 values from images)
- **SV30**: Regression (predict SV30 values from images)
- **MLSS**: Regression (predict MLSS values from images)
- **Filamentous Bacteria**: Classification (classify bacteria types)

### Image Preprocessing Module

**File**: `3D-SM Analysis Code/3D-SM Preprocessing/process_images.py`

**Purpose**: Preprocess raw images for model training

**Features**:
- Horizontal splitting of images into 3 parts
- Resize to 224x224
- Format conversion and normalization

## Dependencies

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.8.0+cu128 | Deep learning framework |
| torchvision | 0.23.0+cu128 | Image processing utilities |
| pandas | 2.2.3 | Data manipulation and analysis |
| numpy | 1.26.4 | Numerical computing |
| scikit-learn | 1.5.2 | Machine learning algorithms |
| scipy | 1.14.1 | Scientific computing |
| Pillow | 11.0.0 | Image processing |
| opencv-python | 4.10.0.84 | Computer vision |
| joblib | 1.4.2 | Model persistence |
| sympy | 1.13.3 | Symbolic mathematics |
| matplotlib | 3.9.2 | Data visualization |
| seaborn | 0.13.2 | Statistical visualization |

### GPU Requirements

- **CUDA**: 12.8
- **cuDNN**: 91002
- **GPU**: NVIDIA GeForce RTX 5080 Laptop GPU (or compatible)

## Environment Configuration

### Windows 11 Setup

1. **Install Python 3.9**
   - Download from [python.org](https://www.python.org/downloads/)
   - Add Python to PATH during installation

2. **Install CUDA 12.8**
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
   - Verify installation: `nvcc --version`

3. **Install cuDNN 91002**
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Extract and copy files to CUDA directory

4. **Create Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

5. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Verify GPU Support**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Linux Setup

The setup is similar to Windows, but use `venv/bin/activate` instead of `venv\Scripts\activate`.

## Configuration Files

### config.py

Located in each task's `src/` directory, contains:

```python
# Model Configuration
NUM_CLASSES = 1  # 1 for regression, >1 for classification
BATCH_SIZE = 32
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 1000

# Data Configuration
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'data')
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'data.csv')

# Model Saving Configuration
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models', 'saved')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Data Format

### Input Data Structure

**CSV File Format** (required for each task):
```
group_id,metric1,metric2,...,target
1,0.5,0.3,...,120.5
2,0.6,0.4,...,125.3
...
```

**Image File Naming**:
```
{group_id}-{image_index}.png
# Example: 1-1.png, 1-2.png, 1-3.png
```

### Output Data Structure

**Prediction Results** (CSV format):
```
group_id,predicted_value,prediction_probability
1,120.5,0.95
2,125.3,0.92
...
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE` in config.py
   - Use smaller model (ResNet-18 instead of ResNet-101)

2. **Module Not Found**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version: `python --version` (should be 3.9)

3. **GPU Not Detected**
   - Verify CUDA installation: `nvcc --version`
   - Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
   - Reinstall PyTorch with CUDA support

4. **File Path Issues**
   - Use relative paths in config.py
   - Ensure data directory structure matches expectations

## License

This project is for academic and research purposes.

## Contact

For questions or issues, please contact the development team.

## Citation

If you use this code in your research, please cite the relevant publications.
