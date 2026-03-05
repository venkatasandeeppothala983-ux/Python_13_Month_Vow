import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
X = np.array([[8.0, 2], [8.5, 3], [7.0, 1], [9.2, 5], [6.5, 0], [8.8, 4]])
y = np.array([1, 1, 0, 1, 0, 1]) # 1 = Hired, 0 = Not Hired
clf = LogisticRegression()
clf.fit(X, y)
my_stats = np.array([[8.68, 3]])
prediction = clf.predict(my_stats)
probability = clf.predict_proba(my_stats)[0][1]
print(f"Prediction for 8.68 CGPA: {'HIRED' if prediction[0] == 1 else 'NOT HIRED'}")
print(f"Confidence Level: {probability*100:.2f}%")
plt.figure(figsize=(10, 6))
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='red', label='Not Hired', s=100)
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='green', label='Hired', s=100)
m1, m2 = clf.coef_[0]
c = clf.intercept_[0]
x_vals = np.linspace(6, 10, 100)
y_vals = -(m1 * x_vals + c) / m2
plt.plot(x_vals, y_vals, '--', color='black', label='Decision Boundary')
plt.fill_between(x_vals, y_vals, 6, color='green', alpha=0.1)
plt.xlabel("CGPA", fontsize=12)
plt.ylabel("Projects", fontsize=12)
plt.title("AI Internship Selection: Logic & Boundary", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
print(f"KNN Result: {knn.predict([[8.68, 3]])}")
