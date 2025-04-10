# Fake-News-Detection

This project detects fake news using machine learning techniques on a pretrained dataset. It includes data preprocessing, feature extraction with TF-IDF, model training, evaluation, and deployment with Streamlit. The model also supports predicting unseen news articles.

## Topics Covered:
- Text preprocessing (stopword removal, stemming, lemmatization, etc.)
- Feature extraction using TF-IDF
- Machine learning models (Logistic Regression, Naive Bayes, Random Forest, SVM)
- Fake news detection using a pretrained dataset
- Prediction for unseen news articles (real-time predictions)
- Evaluation of models based on accuracy, F1-score, and confusion matrix
- Deployment using Streamlit for real-time predictions

## Tools Used:
- **Programming Language**: Python
- **Libraries**:
  - Pandas, NumPy
  - Scikit-learn, Re, String
  - NLTK (Natural Language Toolkit)
  - Streamlit
  **Models**:
  - Logistic Regression
  - Naive Bayes
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - Voting Classifier (Ensemble model)

## Features:
- **Preprocessing**: The text is preprocessed by removing stopwords, punctuation, and performing stemming/lemmatization.
- **TF-IDF Feature Extraction**: Converts text data into numerical features that the machine learning models can understand.
- **Multiple Models**: Implements various machine learning models to classify news articles as real or fake.
- **Support for Unseen News**: The app can predict whether a new (unseen) news article is real or fake based on the trained models.
- **Model Evaluation**: Evaluate models using accuracy, F1-score, confusion matrix, and other metrics.
- **Streamlit App**: Deploys a user-friendly interface where users can paste a news article and get predictions in real-time from multiple models.

## Running the Streamlit App

To run the Streamlit app on your local machine, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/santosh2580/Fake-News-Detection.git
    ```

2. Navigate into the project directory:

    ```bash
    cd Fake-News-Detection
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

5. Access the app in your web browser at `http://localhost:8501`.

6. ## Model Training

1. **Data Preprocessing**:
    - The dataset is cleaned by removing irrelevant information (stopwords, punctuation) and preparing the text for feature extraction.

2. **TF-IDF Vectorization**:
    - TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text into numerical representations for machine learning models.

3. **Model Training**:
    - Models like Logistic Regression, Naive Bayes, Random Forest, and SVM are trained on the dataset to predict whether news is real or fake.

4. **Support for Unseen News**:
    - Once the model is trained, it is capable of predicting unseen news articles that were not part of the training dataset. Users can paste any news article into the app, and the model will classify it as either real or fake.

5. **Evaluation**:
    - The models are evaluated using accuracy, F1-score, and confusion matrix to determine their performance.

