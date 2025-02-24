import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

print("Loading dataset...")

# Hyperparameters
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 128

# Load data
df = pd.read_csv("data/csv/train.csv")
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['prompt'].tolist() + df['story'].tolist())

print("Tokenization complete.")

# Convert text to sequences
X = tokenizer.texts_to_sequences(df['prompt'])
y = tokenizer.texts_to_sequences(df['story'])

# Padding sequences
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
y = pad_sequences(y, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

print("Sequences converted and padded.")

# Build CNN-LSTM Model
def build_model():
    print("Building the model...")
    model = tf.keras.Sequential([
        Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        Conv1D(128, 5, activation='relu', padding='same'),
        LSTM(128, return_sequences=True),  # Keep 3D shape
        LSTM(128, return_sequences=True),  # Keep 3D shape
        LSTM(128),  # Now reduces to 2D
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(MAX_VOCAB_SIZE, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Model built successfully!")

    # **Force model building**
    dummy_input = np.random.randint(0, MAX_VOCAB_SIZE, (1, MAX_SEQUENCE_LENGTH))
    model.predict(dummy_input)  # Ensure model initializes properly

    return model

model = build_model()
print("Model Summary:")
model.summary()