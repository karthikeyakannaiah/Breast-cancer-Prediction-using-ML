import sklearn
from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing
from tkinter import *
import tkinter
import pickle
pickled_in = open("./models/breastCancerPrediction.model","rb")
Model = pickle.load(pickled_in)
classes = ["benign","malignant"]
def predict():
    var1 = int(e1.get())
    var2 = int(e2.get())
    var3 = int(e3.get())
    var4 = int(e4.get())
    var5 = int(e5.get())
    var6 = int(e6.get())
    var7 = int(e7.get())
    var8 = int(e8.get())
    var9 = int(e9.get())
    predictThis = [(var1,var2,var3,var4,var5,var6,var7,var8,var9)]
    pred = Model.predict(predictThis)
    myText.set(classes[pred[0]])
window = Tk()
window.title("Breast Cancer Prediction")
window.geometry("800x600")
myText=StringVar()
Label(window, text="Clump Thickness (1-10): ", font="times 14 bold").grid(row=1,sticky=W)
Label(window, text="uniforminty of cell size (1-10): ", font="times 14 bold").grid(row=2,sticky=W)
Label(window, text="Uniformity of cell shape (1-10): ", font="times 14 bold").grid(row=3,sticky=W)
Label(window, text="marginal adhesion (1-10): ", font="times 14 bold").grid(row=4,sticky=W)
Label(window, text="single epithelial cell size (1-10): ", font="times 14 bold").grid(row=5,sticky=W)
Label(window, text="Bare Nuclei (1-10): ", font="times 14 bold").grid(row=6,sticky=W)
Label(window, text="bland cromatin (1-10): ", font="times 14 bold").grid(row=7,sticky=W)
Label(window, text="Normal Nucleoli (1-10): ", font="times 14 bold").grid(row=8,sticky=W)
Label(window, text="mitoses (1-10): ", font="times 14 bold").grid(row=9,sticky=W)
e1 = Entry(window)
e1.grid(row=1,column=1)
e2 = Entry(window)
e2.grid(row=2,column=1)
e3 = Entry(window)
e3.grid(row=3,column=1)
e4 = Entry(window)
e4.grid(row=4,column=1)
e5 = Entry(window)
e5.grid(row=5,column=1)
e6 = Entry(window)
e6.grid(row=6,column=1)#
e7 = Entry(window)
e7.grid(row=7,column=1)
e8 = Entry(window)
e8.grid(row=8,column=1)
e9 = Entry(window)
e9.grid(row=9,column=1)

bt = Button(window,text="PREDICT",command=predict, padx=10, pady=10,activebackground="light green", font="times 14 bold")
bt.grid(row=11,column=6,sticky=W)
Label(window,text="Prediction is:",font="times 12 bold").grid(row=12,column=0, sticky=W)
result=Label(window, text="", textvariable=myText,font="times 14 bold").grid(row=13,column=0, sticky=W)
window.mainloop()