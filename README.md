Importing Libraries


import numpy as np
import pandas as pd
import tensorflow as tf

These lines import the necessary libraries. NumPy is used for numerical operations, Pandas for data manipulation, and TensorFlow (tf) for building and training neural networks.
Data Preprocessing


dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

This section loads a dataset (Churn_Modelling.csv) using Pandas, extracts features (X) and labels (y) from the dataset.
Encoding Categorical Data



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

The "Gender" column is label-encoded using LabelEncoder to convert categorical data into numerical format.


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

The "Geography" column is one-hot-encoded using ColumnTransformer and OneHotEncoder to handle categorical data properly.
Splitting the Dataset



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

The dataset is split into training and testing sets.
Feature Scaling



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

Feature scaling is applied to standardize the feature values, which is important for neural networks to converge efficiently during training.
Building the ANN



ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

The Sequential model is created, and layers are added to it. The architecture consists of two hidden layers with ReLU activation and an output layer with a sigmoid activation function.
Compiling the ANN



ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

The model is compiled with the Adam optimizer, binary crossentropy loss (suitable for binary classification), and accuracy as a metric.
Training the ANN

python

ann.fit(X_train, y_train, batch_size=32, epochs=100)

The model is trained on the training set for 100 epochs using mini-batch gradient descent (batch size = 32).
Making Predictions

python

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

A sample prediction is made on a new data point using the trained model.



y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

Predictions are made on the test set.
Evaluating the Model



from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

The model is evaluated using a confusion matrix and accuracy score.

Concept: 

