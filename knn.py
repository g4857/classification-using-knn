import pandas as pd

#Importing data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#Splitting data
X_train = train_data.drop(['label'], axis=1).values
y_train = train_data['label'].values
X_test = test_data.values

#Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(X_train)

# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# apply PCA to the data
from sklearn.decomposition import PCA
pca = PCA(.95)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# KNN Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7,n_jobs=-1)

# Traning the model.
knn.fit(X_train,y_train)

# Making Prediction
y_pred = knn.predict(X_test)

# result to csv file
a=[]
for i in range(28000):
    a.append(i+1)

df1 = pd.DataFrame(a)
df2 = pd.DataFrame(y_pred)
df1.columns = ['ImageId']
df2.columns = ['Label']
df = pd.concat([df1,df2],axis=1)
result = df.to_csv('result.csv', index = None, header=True)
#print(df.head(5))
