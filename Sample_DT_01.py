import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

data = pd.read_csv("Training_Set.csv")  # Import the CSV file
print(data.head())

y = data['ClassLabel']  # Save dependent variable into y
data.drop(columns='ClassLabel', inplace=True)  # Drop the dependent var from the dataframe
X = data  # The list of independent vars is saved to X
print(y.head())
print(X.head())

# Use the train_test_split function to randomly segment the training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
print(X_train.shape)

dt = tree.DecisionTreeClassifier()  # Create a decision tree with default Hyper parameters
dt.fit(X_train, y_train)  # Fit the model into the training data
y_pred_test = dt.predict(X_test)  # Calculate the y-pred values using the predict function
print("Accuracy score: ", accuracy_score(y_test, y_pred_test))  # Get the accuracy of the model
