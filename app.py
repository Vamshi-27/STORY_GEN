import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load model
model = tf.keras.models.load_model("story_generator.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Streamlit UI
st.title("AI Story Generator")

prompt = st.text_area("Enter a prompt:", "")

if st.button("Generate Story"):
    if prompt:
        # Preprocess input
        seq = tokenizer.texts_to_sequences([prompt])
        seq = pad_sequences(seq, maxlen=500, padding='post')

        # Generate output
        prediction = model.predict(seq)
        predicted_text = " ".join([tokenizer.index_word.get(i, "") for i in np.argmax(prediction, axis=-1)])

        st.subheader("Generated Story:")
        st.write(predicted_text)
    else:
        st.warning("Please enter a prompt.")
