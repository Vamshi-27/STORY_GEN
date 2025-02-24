import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, Conv1D, LSTM, Dense, Dropout, TimeDistributed
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

print("🔍 Checking for GPU...")
physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:
    print(f"✅ GPU found: {physical_devices[0]}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    DEVICE = "/GPU:0"
else:
    print("❌ No GPU found. Training on CPU.")
    DEVICE = "/CPU:0"

# Hyperparameters
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 256
EMBEDDING_DIM = 128
BATCH_SIZE = 16
EPOCHS = 10

# Load data
print("📂 Loading dataset...")
df = pd.read_csv("data/csv/train.csv")
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['prompt'].tolist() + df['story'].tolist())

print("📝 Tokenization complete.")

# Convert text to sequences
X = tokenizer.texts_to_sequences(df['prompt'])
y = tokenizer.texts_to_sequences(df['story'])

# Padding sequences
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
y = pad_sequences(y, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Ensure all token IDs are within range
y = np.clip(y, 0, MAX_VOCAB_SIZE - 1)

print("📊 Sequences converted and padded.")

# Build CNN-LSTM Model
def build_model():
    print("🔧 Building the model...")
    model = tf.keras.Sequential([
        Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        Conv1D(128, 5, activation='relu', padding='same'),
        LSTM(128, return_sequences=True),
        LSTM(128, return_sequences=True),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(MAX_VOCAB_SIZE, activation='softmax'))  # Predict each token separately
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    print("✅ Model built successfully!")
    return model

model = build_model()

# Force model building with a dummy input
dummy_input = np.random.randint(0, MAX_VOCAB_SIZE, (1, MAX_SEQUENCE_LENGTH))
model.predict(dummy_input)  # Ensure model initializes properly

print("📊 Model Summary:")
model.summary()

# Train the Model using GPU or CPU based on availability
with tf.device(DEVICE):
    print(f"⏳ Starting training on {DEVICE}...")
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )

# Save the Model
model.save("story_generator_model.h5")
print("✅ Model saved successfully!")
