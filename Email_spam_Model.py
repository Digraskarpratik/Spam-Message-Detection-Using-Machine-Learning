import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

data = pd.read_csv("spam.csv", encoding = "latin-1")#for language 
data.head(5573)
#print(data.head(5576))
data.columns
#print(data.columns)
data.drop (['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
data.head()
#print(data.head(5576))
#data is row
data['class']=data['class'].map({'ham':0, 'spam':1})
data.head()
#print(data.head(5573))
cv=CountVectorizer()
x=data['message']
y=data['class']
x.shape
#print(x.shape)
y.shape
#print(y.shape)
x=cv.fit_transform(x)
#print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
#print(x_train.shape)
model = MultinomialNB()
model.fit(x_train, y_train)
#print(model.fit(x_train, y_train))
result = model.score(x_test, y_test)
#print(result = model.score(x_test, y_test))
#overall accuracy score
result = result*100
print("Accuracy of Naive Based algorithm is :- ",result)
pickle.dump(model, open("spam.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))
clf = pickle.load(open("spam.pkl", "rb"))
#print(clf)
# we can check the message is spam means 1 or genuine message for 0
msg = "Hello There"
data = [msg]
vect = cv.transform(data).toarray()
result = model.predict(vect)
print(result)