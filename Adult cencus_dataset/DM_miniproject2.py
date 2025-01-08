# initialize
import csv
XA = [] #sample
XB = []
X = []
Y = [] #classes

# random string perpocessing
import random
import string

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

# open CSV file and read training data
with open('adult.csv', newline='') as csvfile:
	rows = csv.reader(csvfile)
	for row in rows:
		X1 = []
		X2 = []
		if row[0]=="age" :
			continue

		if row[14]=="<=50K" : #Class: income >50K or not
			Y.append(0)
		else :
			Y.append(1)

		X1.append(int(row[0])) #dataA1: age

		if row[1]=="?" : #dataB1: workclass
			X2.append(randomString())
		else :
			X2.append(row[1])

		X2.append(row[3]) #dataB2: education

		X1.append(int(row[4])) #dataA2: education num

		X2.append(row[5]) #dataB3: marital status

		if row[6]=="?" : #dataB4: occupation
			X2.append(randomString())
		else :
			X2.append(row[6])

		X2.append(row[7]) #dataB5: relationship
		
		if row[9]=="Male" : #dataA3: gender
			X1.append(0)
		elif row[9]=="Female" :
			X1.append(1)

		X1.append(int(row[10])) #dataA4: capital gain
		X1.append(int(row[11])) #dataA5: capital loss
		X1.append(int(row[12])) #dataA6: hours per week

		X.append(X1)
		XB.append(X2)

# open test file and read testing data
testX = []
testY = []
testXB = []
f = open("adult.test","r")
test_data = f.readlines()
for data in test_data:
	X1 = []
	X2 = []
	if data[2]=='x' :
		continue

	data = data.split(", ")

	if data[14]=="<=50K.\n" :
		testY.append(0)
	elif data[14]==">50K.\n" :
		testY.append(1)

	X1.append(int(data[0]))
	if data[1]=="?" :
		X2.append(randomString())
	else :
		X2.append(data[1])
	X2.append(data[3])
	X1.append(int(data[4]))
	X2.append(data[5])
	if data[6]=="?" :
		X2.append(randomString())
	else :
		X2.append(data[6])
	X2.append(data[7])
	#sample.append(row[8])
	if data[9]=="Male" :
		X1.append(0)
	elif data[9]=="Female" :
		X1.append(1)
	X1.append(int(data[10]))
	X1.append(int(data[11]))
	X1.append(int(data[12]))
	testX.append(X1)
	XB.append(X2)
f.close()

# building model
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
enc.fit(XB)
XP = enc.transform(XB)

for i in range(len(X)):
	for j in range(5):
		X[i].append(XP[i][j])
for i  in range(len(testX)):
	for j in range(5):
		testX[i].append(XP[i+len(X)][j])

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
n = [1,3,5,7,9,12,15,20,30,50,70,100]

#f = open('result.txt', 'w')
fout=open("result_show.txt","w")

for i in range(len(n)) :
	clf1 = KNeighborsClassifier(n_neighbors=n[i],weights="uniform")
	clf1.fit(X,Y)
	cv_results = cross_validate(clf1, X, Y, cv=5)
	print("uniform classifier result with",n[i],"neighbors:",file=fout)
	print(cv_results['test_score'],file=fout)
	y_pred1 = clf1.predict(testX)
	print(accuracy_score(testY,y_pred1),file=fout)
	print(confusion_matrix(testY, y_pred1),file=fout)
	print(classification_report(testY, y_pred1),file=fout)
	
	clf2 = KNeighborsClassifier(n_neighbors=n[i],weights="distance")
	clf2.fit(X,Y)
	cv_results = cross_validate(clf2, X, Y, cv=5)
	print("distance classifier result with",n[i],"neighbors:",file=fout)
	print(cv_results['test_score'],file=fout)
	y_pred2 = clf2.predict(testX)
	print(accuracy_score(testY, y_pred2),file=fout)
	print(confusion_matrix(testY, y_pred2),file=fout)
	print(classification_report(testY, y_pred2),file=fout)
fout.close()