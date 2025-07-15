# FaceAge Project Setup

This directory contains the FaceAge project, which is a deep learning pipeline for decoding biological age from face photographs.

## Quick Setup

### Option 1: Automated Setup (Recommended)

Run the automated setup script:

```bash
python3 setup_faceage.py
```

This script will:
- Check Python version compatibility
- Create a virtual environment
- Install all required dependencies
- Check for GPU support

### Option 2: Manual Setup

1. **Create a virtual environment:**
   ```bash
   python3 -m venv faceage_env
   ```

2. **Activate the virtual environment:**
   ```bash
   # On macOS/Linux:
   source faceage_env/bin/activate
   
   # On Windows:
   faceage_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r FaceAge/requirements.txt
   ```

## Project Structure

```
FaceAge_Project/
├── FaceAge/                 # Original FaceAge repository
│   ├── src/                # Source code for training and testing
│   ├── models/             # Pre-trained model weights (to be downloaded)
│   ├── data/               # Data processing scripts and links
│   ├── notebooks/          # Demo notebooks
│   ├── outputs/            # Sample outputs
│   └── stats/              # Statistical analysis code
├── faceage_env/            # Virtual environment (created during setup)
├── setup_faceage.py        # Automated setup script
└── README.md               # This file
```

## Required Data and Models

The FaceAge project requires additional data and model weights that are not included in the repository:

### Model Weights
- **Download from:** https://drive.google.com/file/d/1KBZBMKbeqDH95KMbPKJ6vFqnf3ze2aIn/view?usp=sharing
- **Place in:** `FaceAge/models/` directory

### Training Data
- **Download from:** https://drive.google.com/drive/folders/1C2n7KekfbthDFt0t8vbVpD_lEh5lYFyj?usp=sharing
- **Place in:** `FaceAge/data/` directory

## Usage

After setup and downloading the required files:

1. **Activate the virtual environment:**
   ```bash
   source faceage_env/bin/activate
   ```

2. **Navigate to the FaceAge directory:**
   ```bash
   cd FaceAge
   ```

3. **Follow the instructions in the original README.md file**

## System Requirements

- **Python:** 3.6 or higher (3.11+ recommended)
- **Memory:** At least 8GB RAM (16GB+ recommended)
- **Storage:** At least 10GB free space
- **GPU:** Optional but recommended for faster processing

## Troubleshooting

### Common Issues

1. **TensorFlow installation fails:**
   - Try installing TensorFlow CPU version: `pip install tensorflow-cpu`
   - Or use a specific version: `pip install tensorflow==2.6.2`

2. **MTCNN installation issues:**
   - This is a known issue with newer Python versions
   - Try: `pip install mtcnn==0.1.1 --no-deps`
   - Then install dependencies manually

3. **GPU not detected:**
   - Ensure CUDA and cuDNN are properly installed
   - Check TensorFlow GPU installation: `pip install tensorflow-gpu==2.6.2`

### Getting Help

- Check the original FaceAge repository: https://github.com/AIM-Harvard/FaceAge
- Review the detailed README.md in the FaceAge directory
- Check the notebooks in `FaceAge/notebooks/` for examples

## Citation

If you use this code in your work, please cite:

```
Dennis Bontempi*, Osbert Zalay*, Danielle S. Bitterman, Nicolai Birkbak, Jack M. Qian, Hannah Roberts, Subha Perni, Andre Dekker, Tracy Balboni, Laura Warren, Monica Krishan, Benjamin H. Kann, Charles Swanton, Dirk De Ruysscher, Raymond H. Mak*, Hugo J.W.L. Aerts* - Decoding biological age from face photographs using deep learning (submitted).
```

## Disclaimer

The code and data are provided for research purposes only. They are not intended for clinical care or commercial use. See the original repository for full disclaimer. 