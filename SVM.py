# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification

# X, y = make_classification(
#     n_samples=200,
#     n_features=2,
#     n_redundant=0,
#     n_clusters_per_class=1,
#     class_sep=1.2,
#     flip_y=0.05,
#     random_state=42
# )

# outliers = np.random.uniform(low=-6, high=6, size=(10, 2))
# X = np.vstack((X, outliers))
# y = np.hstack((y, np.random.randint(0, 2, 10)))

# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.title("Realistic Data with Noise and Outliers")
# plt.show()


# import numpy as np

# class SVM:
#     def __init__(self, lr=0.001, lambda_param=0.01, epochs=1000):
#         self.lr = lr
#         self.lambda_param = lambda_param
#         self.epochs = epochs

#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         y = np.where(y <= 0, -1, 1)

#         self.w = np.zeros(n_features)
#         self.b = 0

#         for _ in range(self.epochs):
#             for idx, x_i in enumerate(X):
#                 condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1

#                 if condition:
#                     self.w -= self.lr * (2 * self.lambda_param * self.w)
#                 else:
#                     self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
#                     self.b -= self.lr * y[idx]

#     def predict(self, X):
#         linear_output = np.dot(X, self.w) - self.b
#         return np.sign(linear_output)
    

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC

# log_model = LogisticRegression()
# svm_model = SVC(kernel="linear")

# log_model.fit(X, y)
# svm_model.fit(X, y)


# def plot_boundary(model, X, y, title):
#     plt.figure()
#     plt.scatter(X[:, 0], X[:, 1], c=y)

#     ax = plt.gca()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()

#     xx = np.linspace(xlim[0], xlim[1], 200)
#     yy = np.linspace(ylim[0], ylim[1], 200)
#     YY, XX = np.meshgrid(yy, xx)
#     xy = np.vstack([XX.ravel(), YY.ravel()]).T
#     Z = model.decision_function(xy)
#     Z = Z.reshape(XX.shape)

#     plt.contour(XX, YY, Z, levels=[0])
#     plt.title(title)
#     plt.show()

# plot_boundary(log_model, X, y, "Logistic Regression Boundary")
# plot_boundary(svm_model, X, y, "SVM Boundary")


# import numpy as np
# from sklearn.datasets import make_classification

# X, y = make_classification(
#     n_samples=2000,
#     n_features=100,
#     n_informative=10,
#     n_redundant=20,
#     n_repeated=0,
#     n_clusters_per_class=2,
#     flip_y=0.08,
#     class_sep=0.8,
#     random_state=42
# )
# print(X.shape)

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42
# )

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# log_model = LogisticRegression(max_iter=5000)
# svm_model = SVC(kernel="linear")

# log_model.fit(X_train, y_train)
# svm_model.fit(X_train, y_train)

# log_pred = log_model.predict(X_test)
# svm_pred = svm_model.predict(X_test)

# print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
# print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# pca = PCA(n_components=2)
# X_vis = pca.fit_transform(X_test)

# plt.figure(figsize=(12,5))

# plt.subplot(1,2,1)
# plt.scatter(X_vis[:,0], X_vis[:,1], c=log_pred)
# plt.title("Logistic Regression Predictions (PCA view)")

# plt.subplot(1,2,2)
# plt.scatter(X_vis[:,0], X_vis[:,1], c=svm_pred)
# plt.title("SVM Predictions (PCA view)")

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

np.random.seed(42)

# Class 0
X0 = np.random.randn(50,3) + np.array([0,0,0])
# Class 1
X1 = np.random.randn(50,3) + np.array([5,5,5])
# Add outliers
outliers = np.array([[2,2,8],[3,1,7],[4,0,6]])
X1 = np.vstack([X1, outliers])

X = np.vstack([X0,X1])
y = np.array([0]*50 + [1]*53)

# 3D plotting
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c=y, s=50)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
ax.set_title("3D Data with Outliers")
plt.show()


# Train models
log_model = LogisticRegression().fit(X, y)
svm_model = SVC(kernel='linear').fit(X, y)

print("Log Model : ",log_model.coef_)

w1 = log_model.coef_[0]
b1 = log_model.intercept_[0]

x1,y1,=np.meshgrid(np.linspace(-2,7,10),np.linspace(-2,7,10))
zz = -(w1[0]*x1 + w1[1]*y1 -b1)/w1[2]

fig = plt.figure(figsize=(8,6))
ap = fig.add_subplot(111,projection='3d')
ap.scatter(X[:,0],X[:,1],X[:,2],c=y,s=30)
ap.plot_surface(x1,y1,zz,color='cyan',alpha=0.3)

# Conceptual: in 3D, the plane equation is w1*x + w2*y + w3*z + b = 0
# We can plot a mesh plane for SVM
w = svm_model.coef_[0]
b = svm_model.intercept_[0]

xx, yy = np.meshgrid(np.linspace(-2,7,10), np.linspace(-2,7,10))
zz = (-w[0]*xx - w[1]*yy - b)/w[2]
print("SVM Model : ",svm_model.coef_[0])

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c=y, s=50)
ax.plot_surface(xx, yy, zz, color='cyan', alpha=0.3)
ax.set_title("SVM Decision Plane in 3D")
plt.show()
