import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class KNN:

    def __init__(self,k=3):
        self.K=k

    def fit(self,X,y):
        self.X_train=X
        self.y_train=y

    def euclid(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def  predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self,x):
        dist = [self.euclid(x,x_train) for x_train in self.X_train]
        k_indices = np.argsort(dist)[:self.K]
        k_labels = [self.y_train[i] for i in k_indices]
        common = np.bincount(k_labels).argmax()
        return common
    

X, y = make_classification(n_samples = 50,
                                       n_features = 2,
                                       n_informative = 2,
                                       n_redundant = 0,
                                       n_classes = 2,
                                       weights = [0.51, .49])

# print(X.shape)
# print(X)
X_train, X_test,y_train, y_test = train_test_split(X,y ,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)
print(X_train)
print(X_test)

knn = KNN(k=2)
knn.fit(X_train, y_train)

X_new = np.array([[5.5, 3.5]])
prediction = knn.predict(X_test)

print("Predicted class:", prediction)
print("Actual class:", y_test)