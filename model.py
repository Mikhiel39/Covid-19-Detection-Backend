from colorama import Fore, Style
import colorama
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Initialize Colorama
colorama.init(autoreset=True)

# Load dataset
covid_data = pd.read_csv("./Covid_Dataset.csv")

# Create a table with data missing
missing_values = covid_data.isnull().sum()  # Missing values
percent_missing = covid_data.isnull().sum(
) / covid_data.shape[0] * 100  # Missing value %

# Create DataFrame with missing values information
value = {
    'missing_values': missing_values,
    'percent_missing %': percent_missing
}
frame = pd.DataFrame(value)
print(frame)

# Encode categorical variables
encoder = LabelEncoder()
columns_to_encode = [
    'Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat', 'Running Nose', 'Asthma',
    'Chronic Lung Disease', 'Headache', 'Heart Disease', 'Diabetes', 'Hyper Tension',
    'Abroad travel', 'Contact with COVID Patient', 'Attended Large Gathering',
    'Visited Public Exposed Places', 'Family working in Public Exposed Places',
    'Wearing Masks', 'Sanitization from Market', 'COVID-19', 'Gastrointestinal ', 'Fatigue '
]

for column in columns_to_encode:
    covid_data[column] = encoder.fit_transform(covid_data[column])

# Prepare features and target
x = covid_data.drop('COVID-19', axis=1)
y = covid_data['COVID-19']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=101)

# Initialize dictionaries to store metrics
accuracies = {}
algo_time = {}
r2_scores = {}
mean_squared_errors = {}
roc_auc_scores = {}

# Function to print performance metrics


def print_performance(yt, clf, clf_name):
    y_pred = clf.predict(x_test)
    roc_auc_scores[clf_name] = roc_auc_score(yt, y_pred) * 100
    mean_squared_errors[clf_name] = mean_squared_error(yt, y_pred) * 100
    r2_scores[clf_name] = r2_score(yt, y_pred) * 100
    accuracies[clf_name] = clf.score(x_train, y_train) * 100


# Train K-Nearest Neighbors with GridSearchCV
start = time.time()
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 50)}
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(x_train, y_train)

# Save the model to disk
pickle.dump(knn_cv, open('model.pkl', "wb"))
