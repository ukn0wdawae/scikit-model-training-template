import numpy as np  
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Copy data
!cp -r /content/drive/MyDrive/ICU_Dataset/ICUDatasetProcessed .

# Read and preprocess data
path = 'ICUDatasetProcessed/'
csvs = os.listdir(path)

df1 = pd.DataFrame()
for csv in csvs:
    df = pd.read_csv(path + csv)
    df.fillna(0, inplace=True)
    df1 = df1.append(df, ignore_index=True)

# Feature selection
feats = ['ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport','mqtt.topic', 'mqtt.msg', 'tcp.payload',
         'mqtt.clientid', 'mqtt.conflags', 'mqtt.conack.flags', 'class']
df1.drop(labels=feats, axis=1, inplace=True)

fs2 = ['frame.time_delta', 'tcp.time_delta', 'tcp.flags.ack', 'tcp.flags.push', 'tcp.flags.reset',
       'mqtt.hdrflags', 'mqtt.msgtype', 'mqtt.qos', 'mqtt.retain', 'mqtt.ver', 'label']
df1 = df1[fs2]

# Encode categorical features
label_encoder = preprocessing.LabelEncoder()
df1['mqtt.hdrflags'] = label_encoder.fit_transform(df1['mqtt.hdrflags'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df1.drop(labels=['label'], axis=1), df1['label'], test_size=0.3, random_state=100
)

# Logistic Regression Feature Selection
embeded_LR_selector = SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear'), 
                                      threshold='0.9*median', max_features=10)
embeded_LR_selector.fit(X_train, y_train)
embeded_LR_support = embeded_LR_selector.get_support()
embeded_LR_feature = X_train.loc[:, embeded_LR_support].columns.tolist()

# Train and evaluate classifiers
models = {
    "GaussianNB": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(max_depth=10, random_state=100),
    "AdaBoost": AdaBoostClassifier(),
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=10, min_samples_leaf=5)
}

results = [("Classifier", "Accuracy", "Precision", "Recall", "F1-Score")]
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results.append((
        name,
        accuracy_score(y_test, predictions) * 100,
        precision_score(y_test, predictions) * 100,
        recall_score(y_test, predictions) * 100,
        f1_score(y_test, predictions) * 100
    ))

# Display results
df_results = pd.DataFrame(results[1:], columns=results[0])
print(df_results)

# Confusion Matrices
for name, model in models.items():
    predictions = model.predict(X_test)
    print(f"Confusion Matrix for {name}:")
    print(confusion_matrix(y_test, predictions))