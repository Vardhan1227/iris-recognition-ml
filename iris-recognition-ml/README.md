# Iris Recognition Using Machine Learning

Final year B.Tech project for iris-based biometric recognition using Python and Machine Learning.

##  Features
- Iris segmentation using Daugman’s algorithm
- Histogram equalization
- Feature extraction using HOG
- Classification using SVM

##  Dataset Structure
Organize your images like:

```
dataset/
├── person1/
│   ├── eye1.jpg
│   └── eye2.jpg
├── person2/
```

##  How to Run

```bash
pip install -r requirements.txt
python src/iris_recognition.py
```

##  Output
- Accuracy and Classification Report printed to console
- Tested on CASIA iris image dataset

## Authors
- K. Uma Maheswari
- L. Roja Rohini
- B. Anil Varma
- M. Vardhan

## Based on
- Daugman’s Algorithm
- SVM Classification
- Histogram Equalization
