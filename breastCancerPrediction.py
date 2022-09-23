import sklearn
from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing
import pickle
data = pd.read_csv('./data/breast-cancer-wisconsin.data')
le = preprocessing.LabelEncoder()

cls = le.fit_transform(list(data['class']))

clumpThickness = list(data['clumpThickness'])
UniFormCellSize = list(data['UniFormCellSize'])
UniFormCellShape = list(data['UniFormCellShape'])
marginalAdhesion = list(data['marginalAdhesion'])
SingleEpithCellSize = list(data['SingleEpithCellSize'])
bareNuclei = list(data['bareNuclei'])
blandCromatin = list(data['blandCromatin'])
normalNucleoli = list(data['normalNucleoli'])
mitoses = list(data['mitoses'])

classes = ["benign","malignant"]

X = list(zip(clumpThickness,UniFormCellSize,UniFormCellShape,marginalAdhesion,SingleEpithCellSize,bareNuclei,blandCromatin,normalNucleoli,mitoses))
Y = list(cls)
best =0
# training the model
for _ in range(10):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.05)

    classifier = svm.SVC(kernel="linear", C=2)

    classifier.fit(x_train,y_train)
    predictions = classifier.predict(x_test)
    acc = metrics.accuracy_score(y_test, predictions)

    if acc > best:
        best = acc
        with open("./models/breastCancerPrediction.model","wb") as f:
            pickle.dump(classifier,f)