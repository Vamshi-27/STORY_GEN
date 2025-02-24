# AI Story Generator

## Overview
The **AI Story Generator** is a deep learning-based project that generates creative stories based on writing prompts. This project leverages NLP techniques and a custom **CNN-LSTM** deep learning model to generate coherent and engaging narratives.

## Features
- **Preprocessing:** Cleans and tokenizes input data from a dataset.
- **Training:** Uses a **CNN-LSTM** neural network model to learn from story prompts.
- **Generation:** Produces unique and engaging stories based on input prompts.
- **Evaluation:** Assesses the model's performance using metrics like accuracy and F1-score.
- **Deployment:** Provides a simple web UI using **Streamlit** for easy interaction.

## Dataset
The project uses the **WritingPrompts dataset**, stored in `data/writingPrompts/`. The dataset consists of writing prompts and their corresponding stories. The preprocessing script converts the dataset into CSV format, which is then used for model training. You can enhance the model's performance by adding more prompts.

## File Structure
```
📁 Story_Generator
│── 📁 data
│   └── 📁 writingPrompts  # Raw dataset files
│   └── 📁 csv
│       ├── train.csv  # Training data
│       ├── valid.csv  # Validation data
│       ├── test.csv  # Test data
│── 📁 models
│   ├── story_generator.h5  # Trained model
│   ├── tokenizer.pkl  # Tokenizer for text processing
│── 📁 src
│   ├── preprocess.py  # Convert dataset to CSV format
│   ├── train.py  # Train the CNN-LSTM model
│   ├── build_model.py  # Defines the CNN-LSTM architecture
│   ├── generate.py  # Story generation script
│   ├── evaluate.py  # Model evaluation script
│── 📁 deployment
│   ├── app.py  # Streamlit web UI for text generation
│── requirements.txt  # List of dependencies
│── README.md  # Project documentation
```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Vamshi-27/SIC_Project.git
   cd SIC_Project
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### 1. Preprocess Data
Convert the raw dataset into CSV format:
```sh
python src/preprocess.py
```
### 2. Train the Model
Train the **CNN-LSTM** model using the processed dataset:
```sh
python src/train.py
```
### 3. Generate Stories
Generate a new story based on a writing prompt:
```sh
python src/generate.py --prompt "Once upon a time..."
```
### 4. Evaluate Model
Evaluate the trained model on test data:
```sh
python src/evaluate.py
```
### 5. Deploy with Streamlit
Run the Streamlit web app:
```sh
streamlit run deployment/app.py
```

## Dependencies
Ensure you have all required Python packages installed by referring to `requirements.txt`. The project mainly uses:
- TensorFlow/Keras
- NLTK
- Pandas
- NumPy
- Streamlit

## Future Enhancements
- Improve model accuracy with advanced NLP techniques.
- Introduce a feedback mechanism to refine generated stories.
- Deploy as a web application with user authentication.
- Explore integration with voice input for interactive storytelling.

