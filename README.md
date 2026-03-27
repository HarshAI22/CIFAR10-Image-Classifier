# CIFAR-10 Image Classifier

A Convolutional Neural Network for 10-class image recognition trained on the CIFAR-10 dataset.

## Tech Stack
- TensorFlow, Keras — CNN model
- NumPy — data processing
- Matplotlib — visualizations
- Streamlit — interactive web app

## Features
- CNN trained on CIFAR-10 dataset
- 10 class prediction: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Random test image prediction with confidence score
- Interactive Streamlit interface

## Project Structure

CIFAR10-Image-Classifier/
├── cifar10_cnn.ipynb
├── app.py
├── cifar10_model.keras
├── requirements.txt


## Setup
bash
pip install -r requirements.txt
streamlit run app.py
