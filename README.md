# Email Spam Classification Project

## Introduction and Objective

In the digital era, email is a crucial communication tool, but it faces the persistent challenge of spam. Spam emails, or junk emails, are unsolicited messages sent in bulk, often for advertising, phishing, or spreading malware. The key issues with spam emails include: <br>

* Security: They can contain malicious links, attachments, or phishing attempts.
* Productivity: Sorting through spam wastes time and distracts users.
* Storage: They consume storage space, increasing data management costs.
* User Experience: A cluttered inbox makes it difficult to locate important emails. <br>

This project aims to build a robust and efficient Email Spam Classifier using machine learning techniques. By training the model on a comprehensive dataset of labeled emails from Kaggle, we aim to adapt to new and evolving spam techniques, providing a reliable solution for maintaining a spam-free inbox.

<img src = "https://github.com/somaksanyal97/Email-Spam-Classifier/blob/main/pics/readme%20pic.png" style="width:1000px; height:300px;">

## Data Cleaning

In our preprocessing pipeline for the email spam classification dataset, we began by cleaning the data. This involved removing unnecessary columns ('Unnamed: 0' and 'Unnamed: 0.1') from multiple dataframes and filtering out rows where the 'Body' column was empty. We then merged the cleaned dataframes and removed any duplicate entries. After merging, we reset the index of the dataframe to ensure it was properly ordered. <br>

## Text Preprocessing

Subsequently, we prepared the text data by iterating through each row, removing non-alphabetical characters, converting the text to lowercase, splitting it into individual words, removing English stopwords, and applying stemming using the Porter Stemmer. We used stemming instead of lemmatization for this project because it is faster and we do not need the root form of the words to be meaningful for this project. The processed text was then compiled into a new list called corpus. <br>

## Feature Extraction

Following text preprocessing, we transformed the cleaned text data into numerical feature representations suitable for machine learning models. First, we utilized a CountVectorizer to convert the text data into a bag-of-words model with a maximum of 6000 features, resulting in the feature matrix X. Concurrently, we assigned the target labels from the original dataframe to the variable y. In parallel, we applied a TfidfVectorizer to generate a TF-IDF (Term Frequency-Inverse Document Frequency) representation of the text data, also with a maximum of 6000 features, resulting in the feature matrix X1. The corresponding labels for this representation were stored in y1. This dual approach provided two different sets of features for subsequent machine learning model training and evaluation.<br>

## Train-Test Split

After transforming the text data into numerical feature representations, we proceeded with splitting the data into training and testing sets. This was done separately for the features derived from the CountVectorizer and the TfidfVectorizer. For the bag-of-words representation, the feature matrix X and labels y were split into training and testing sets. Similarly, the TF-IDF features in X1 and their corresponding labels y1 were also divided into training and testing subsets. This split ensured that both feature sets could be independently evaluated for model performance. <br>

## Oversampling for Class Imbalance

Following the train-test split, we applied oversampling to address any class imbalance in the training data. This technique was used to augment the minority class, ensuring that the training sets had a more balanced representation of both classes. For the CountVectorizer-based features, oversampling was performed on the training subset derived from 'X' and 'y'. Similarly, for the TF-IDF features, oversampling was applied to the training data from 'X1' and 'y1'. This step aimed to improve the robustness and accuracy of the classification models by mitigating potential biases caused by imbalanced classes. <br>

## Model Training and Evaluation

In this section of the code, we evaluated several machine learning classifiers for the email spam classification task, using two different feature extraction methods: CountVectorizer and TfidfVectorizer. First, we defined a dictionary containing various classifiers (Multinomial Naive Bayes, Logistic Regression, K-Nearest Neighbors, Decision Tree, and Random Forest) along with their hyperparameter grids for tuning. Two dictionaries were then initialized: one to store the best-performing model for each classifier and another to store evaluation metrics (Accuracy, Precision, Recall, and F1 Score). We iterated over each classifier, performing hyperparameter tuning using GridSearchCV with 5-fold cross-validation on the oversampled training data. The best model was then used to make predictions on the test data, and performance metrics were calculated and stored. Confusion matrices and performance metrics were printed and visualized for each classifier. This entire process was repeated separately for both the CountVectorizer and TfidfVectorizer feature sets, allowing for a comprehensive comparison of classifier performance across different feature representations.

## Results

## Performance Metrics of ML Algorithms with CountVectorizer

| Model        | Accuracy   | Precision | Recall   | F1 Score | Best Parameters |
|----------------|-----------|--------------------|----------------|-----------|--------------------|
| Naive Bayes | 0.95 | 0.95     | 0.95 | 0.95 |  {'alpha': 0.5}   |
| Logistic Regression | 0.97 | 0.97    | 0.97 | 0.97 |   {'C': 1}   |
| K-Nearest Neighbors | 0.83 | 0.88     | 0.83 | 0.83 |   {'n_neighbors': 3}   |
| Decision Tree | 0.93 | 0.93     | 0.93 | 0.93 |  {'max_depth': None}   |
| Random Forest | 0.97 | 0.97     | 0.97 | 0.97 |   {'max_depth': None, 'n_estimators': 200} |

<div style="display: flex; justify-content: space-around;">
  <div style="text-align: center;">
    <img src="https://github.com/somaksanyal97/Email-Spam-Classifier/blob/main/pics/readme%20pic.png" alt="CountVectorizer Performance" width="400"/>
    <p>CountVectorizer Performance</p>
  </div>
  <div style="text-align: center;">
    <img src="https://github.com/somaksanyal97/Email-Spam-Classifier/blob/main/pics/readme%20pic.png" alt="TfidfVectorizer Performance" width="400"/>
    <p>TfidfVectorizer Performance</p>
  </div>
</div>

## Performance Metrics of ML Algorithms with TfidfVectorizer

| Model        | Accuracy   | Precision | Recall   | F1 Score | Best Parameters |
|----------------|-----------|--------------------|----------------|-----------|--------------------|
| Naive Bayes | 0.97 | 0.97    | 0.97 | 0.97 |  {'alpha': 0.5}   |
| Logistic Regression | 0.98 | 0.98    | 0.98 | 0.98 |   {'C': 10}   |
| K-Nearest Neighbors | 0.62 | 0.81    | 0.62 | 0.60 |    {'n_neighbors': 3}   |
| Decision Tree | 0.93 | 0.93     | 0.93 | 0.93 |  {'max_depth': None}   |
| Random Forest | 0.98 | 0.98     | 0.98 | 0.98 |   {'max_depth': None, 'n_estimators': 200} |
