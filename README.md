# Deepfake Detection and Media Authentication Agent

This project implements a web-based system for detecting deepfake media and authenticating the integrity of images and videos.

## Features

- Deepfake detection using deep learning (EfficientNet)
- Media authentication through metadata analysis
- Noise pattern analysis for manipulation detection
- User-friendly web interface
- Real-time analysis results
- Support for images and videos

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YASHMANIC/Deepfake-detection.git
cd deepfake-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload an image or video file using the web interface

4. Click "Analyze Media" to process the file

5. View the detailed results, including:
   - Deepfake detection probability
   - Media authentication score
   - Noise pattern analysis
   - Metadata verification

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- Minimum 8GB RAM
- Supported operating systems: Windows, macOS, Linux

## Technical Details

### Deepfake Detection
- Uses EfficientNet-B0 architecture
- Pre-trained on large-scale deepfake datasets
- Binary classification (real/fake)

### Media Authentication
- Metadata analysis
- Error Level Analysis (ELA)
- Noise pattern inconsistency detection
- File integrity verification
