from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Load test data
df_test = pd.read_csv("/content/test.csv")

X_test = tokenizer.texts_to_sequences(df_test['prompt'])
y_test = tokenizer.texts_to_sequences(df_test['story'])

X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
y_test = pad_sequences(y_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Load trained model
model.load_weights("story_generator.h5")

# Predict
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)

# Compute Metrics
accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())
f1 = f1_score(y_test.flatten(), y_pred.flatten(), average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
