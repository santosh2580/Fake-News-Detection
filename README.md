**Fake News Detection Using Machine Learning**

This project detects fake news using machine learning techniques on a dataset of real and fake news articles. It includes data preprocessing, feature extraction with TF-IDF, model training, evaluation, and model deployment with Streamlit.

### **Topics Covered:**

- **Text Preprocessing:**
  - Stopword removal
  - Lemmatization
  - URL and punctuation removal
  - Lowercasing and noise reduction
    
- **Feature Extraction:**
  - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
    
- **Machine Learning Models Used:**
  - Logistic Regression
  - Naive Bayes (MultinomialNB)
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - Voting Classifier (ensemble of multiple models)
    
- **Model Evaluation:**
  - Accuracy and F1-score comparison
  - Confusion matrix visualization

### **Tools & Libraries Used:**
- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Matplotlib, Seaborn**
- **NLTK (Natural Language Toolkit)**
- **Joblib (for model saving & loading)**
- **Streamlit (for deployment)**

### **Run the Project:**
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train and save models by running the script:
   ```bash
   python train.py
   ```
3. Run the Streamlit app for model deployment:
   ```bash
   streamlit run app.py
   ```

