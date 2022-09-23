import sklearn
from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing
import pickle
# scn,clumpThickness,UniFormCellSize,
# UniFormCellShape,marginalAdhesion,
# SingleEpithCellSize,bareNuclei,
# blandCromatin,normalNucleoli,mitoses,
# class

data = pd.read_csv('./data/breast-cancer-wisconsin.data')
le = preprocessing.LabelEncoder()

cls = le.fit_transform(list(data['class']))

# uncls = le.inverse_transform(list(cls))
# print(cls[0])
# print(uncls[0])
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

pickled_in = open("./models/breastCancerPrediction.model","rb")
clsfr = pickle.load(pickled_in)

predictions = clsfr.predict([(5, 1, 1, 1, 2, 0, 3, 1, 1)])
print(classes[predictions[0]])


# predictions = classifier.predict([(5, 1, 1, 1, 2, 0, 3, 1, 1)])
# acc = metrics.accuracy_score(y_test,predictions)

# # print(predictions)
# for x in range(len(predictions)):
#     print(classes[predictions[x]])