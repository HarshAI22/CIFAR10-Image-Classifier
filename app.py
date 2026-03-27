import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10

model = load_model("cifar10_model.keras")

class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test / 255.0

st.title("CIFAR-10 Image Classifier")
st.write("Click the button to predict a random test image")

if st.button("🎲 Random Image Predict"):
    idx = np.random.randint(0, len(x_test))
    image = x_test[idx]
    true_label = class_labels[y_test[idx][0]]
    
    prediction = model.predict(image.reshape(1, 32, 32, 3))
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.image(image, caption="Test Image", width=200)
    st.success(f"Predicted: **{predicted_class}**")
    st.info(f"Actual: **{true_label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")