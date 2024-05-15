# Hate Speech Detection using Decision Trees

## Overview
This project aims to detect hate speech and offensive language in tweets using a Decision Tree classifier. The system analyzes the text content of tweets and classifies them into one of three categories: Hate Speech, Offensive Language, or No Hate and Offensive.

## Features
- **Data Preprocessing:** Cleans the text data by removing URLs, special characters, punctuation, stopwords, and numbers. It also performs stemming to reduce words to their root form.
  
- **Model Training:** Utilizes a Decision Tree classifier to train the hate speech detection model on a labeled dataset of tweets.
  
- **Accuracy Evaluation:** Evaluates the performance of the trained model using test data and calculates the classification accuracy.
  
- **Web Application:** Implements a simple web application using Streamlit, allowing users to input a tweet and receive a prediction on whether it contains hate speech or offensive language.

## Installation
1. **Clone the Repository:**
   ```
   git clone [https://github.com/your_username/hate_speech_detection.git](https://github.com/Premaramkarthik/Hate-Speech-Detection)
   ```

2. **Install Dependencies:**
   ```
   pip install pandas numpy scikit-learn nltk streamlit
   ```

3. **Run the Web Application:**
   ```
   streamlit run app.py
   ```

4. **Open the Web Browser:**
   Access the web application by navigating to the URL provided by Streamlit in your web browser.

## Usage
1. **Input Tweet:**
   Enter any tweet into the text area provided on the web application interface.
  
2. **Prediction:**
   Click on the button to submit the tweet. The application will display the predicted category for the tweet (Hate Speech, Offensive Language, or No Hate and Offensive).

3. **Repeat:**
   You can input multiple tweets and receive predictions for each one.

## Dataset
- The dataset used for training and testing the model can be found in the file `twitter.csv`.
  
- It contains labeled tweets categorized into three classes: 0 (Hate Speech), 1 (Offensive Language), and 2 (No Hate and Offensive).

## Contributions
Contributions to this project are welcome! If you have any suggestions, improvements, or feature requests, please open an issue or create a pull request.


## Acknowledgments
Special thanks to the developers of the libraries and tools used in this project, including Pandas, NumPy, scikit-learn, NLTK, and Streamlit.

---

Feel free to customize the README file with additional information, such as project setup instructions, usage examples, or acknowledgments to contributors and collaborators.
