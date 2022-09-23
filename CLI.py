import sklearn
from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing
import pickle

pickled_in = open("./models/breastCancerPrediction.model","rb")
clsfr = pickle.load(pickled_in)

clumpThickness = int(input("Clump Thickness (1-10): "))
UniFormCellSize = int(input("uniforminty of cell size (1-10): "))
UniFormCellShape = int(input("Uniformity of cell shape (1-10): "))
marginalAdhesion = int(input("marginal adhesion (1-10): "))
SingleEpithCellSize = int(input("single epithelial cell size (1-10): "))
bareNuclei = input("Bare Nuclei (1-10): ")
blandCromatin = int(input("bland cromatin (1-10): "))
normalNucleoli = int(input("Normal Nucleoli (1-10): "))
mitoses = int(input("mitoses (1-10): "))

classes = ["benign","malignant"]
# 5,1,1,1,2,1,3,1,1 --> test benign
# 8,7,4,4,5,3,5,10,1 --> test malignant
predictThis = [(clumpThickness,UniFormCellSize,UniFormCellShape,marginalAdhesion,SingleEpithCellSize,bareNuclei,blandCromatin,normalNucleoli,mitoses)]
print(predictThis)
predictions = clsfr.predict(predictThis)

print("\n\nPrediction: ",classes[predictions[0]])