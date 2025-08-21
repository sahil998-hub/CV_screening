import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

df = pd.read_csv('UpdatedResumeDataSet.csv')

def cleanResume(txt):
    txt = re.sub('http\S+', ' ', txt)
    txt = re.sub('@\S+', ' ', txt)
    txt = re.sub('#\S+', ' ', txt)
    txt = re.sub('RT|cc', ' ', txt)
    txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    return txt

df['Resume'] = df['Resume'].apply(cleanResume)

le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['Resume'])
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_train, y_train)

pickle.dump(clf, open('clf.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))

print("✅ Model, vectorizer, and label encoder saved successfully.")
