from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the Iris dataset
iris = load_iris()
x = iris.data
y = iris.target

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = RandomForestClassifier(n_jobs=-1)
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

# Save the model
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
