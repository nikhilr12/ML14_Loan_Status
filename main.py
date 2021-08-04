import pandas as pd
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.python.keras.api._v1.keras
from tensorflow.python.keras.api._v1.keras.layers import Dense
from tensorflow.python.keras.api._v1.keras.models import Sequential
data = pd.read_csv("C:/Users/safu1/Downloads/archive/train_u6lujuX_CVtuZ9i.csv")
data.drop(columns=['Loan_ID'], inplace=True)
data = data.dropna()
LS = pd.get_dummies(data['Loan_Status'], drop_first=True)
Gndr = pd.get_dummies(data['Gender'], drop_first=True)
Mrrd = pd.get_dummies(data['Married'], drop_first=True)
EduStat = pd.get_dummies(data['Education'], drop_first=True)
Self = pd.get_dummies(data['Self_Employed'], drop_first=True)
Prop = pd.get_dummies(data['Property_Area'], drop_first=True)
Deps = pd.get_dummies(data['Dependents'], drop_first=True)
data = data.drop(
columns=['Loan_Status', 'Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents'], axis=1)
data = pd.concat([LS, data, Gndr, Mrrd, EduStat, Self, Prop, Deps], axis=1)
y = data['Y']
x = data.drop(columns=['Y'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(14,)))
model.add(Dense(35, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)
#model = LogisticRegression(max_iter=5000)
#model.fit(x_train, y_train)
pred = model.predict(x_test)
confusion_matrix(y_test, pred)
print(precision_score(y_test, pred))
print(recall_score(y_test, pred))
print(f1_score(y_test, pred))
print(cohen_kappa_score(y_test, pred))
#print('score1:', model.score(x_test, y_test))
#print('score2:', model.score(x_train, y_train))