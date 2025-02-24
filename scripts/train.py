import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, Conv1D, LSTM, Dense, Dropout, TimeDistributed
)
from tensorflow.keras.utils import to_categorical

# Hyperparameters
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 256
EMBEDDING_DIM = 128
BATCH_SIZE = 16
EPOCHS = 10

# Ensure all token IDs are within range
y = np.clip(y, 0, MAX_VOCAB_SIZE - 1)

# Convert `y` to one-hot encoding
y = to_categorical(y, num_classes=MAX_VOCAB_SIZE)

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
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    print("✅ Model built successfully!")
    return model

model = build_model()
print("📊 Model Summary:")
model.summary()

# Train the Model
print("⏳ Starting training...")
history = model.fit(
    X, y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)

# Save the Model
model.save("story_generator_model.h5")
print("✅ Model saved successfully!")
