# initialize
import csv
X = [] #sample
Y = [] #classes
Result = [["PassengerId","Survived"]]
Gender = {"male":0, "female":1}
Fare_threshold = [72.0, 35.2, 26.0, 16.5, 13.0, 8.66, 7.89, 7.73, -1]
Embarked = {"":1, "S":1, "C":2, "Q":3}

# open CSV file and read training data
with open('train.csv', newline='') as csvfile:
	rows = csv.reader(csvfile)
	for row in rows:
		if row[0]=="PassengerId" :
			continue

		Y.append(int(row[1])) #Class: survived or not

		sample = []
		sample.append(int(row[2])) #data1: Pclass

		sample.append(Gender[row[4]]) #data2: Gender

		fare = float(row[9]) #data3: Fare
		for i in range(len(Fare_threshold)):
			if fare > Fare_threshold[i]:
				sample.append(i+1)
				break

		sample.append(Embarked[row[11]]) #data4: Embarked

		X.append(sample)

# building models
classification_models = []
from sklearn import tree
classification_models.append( tree.DecisionTreeClassifier(max_depth=5,min_samples_leaf=3,max_leaf_nodes=30,max_features=2) )
from sklearn.neural_network import MLPClassifier
classification_models.append( MLPClassifier(max_iter=500) )
from sklearn.neighbors import KNeighborsClassifier
classification_models.append( KNeighborsClassifier(n_neighbors=5) )
from sklearn.naive_bayes import GaussianNB
classification_models.append( GaussianNB() )
from sklearn.svm import LinearSVC
classification_models.append( LinearSVC(random_state=0, tol=1e-5,max_iter=5000) )

from sklearn.model_selection import cross_validate

for clf in classification_models:
	clf.fit(X, Y)
	print( "test_score: ", cross_validate(clf, X, Y, cv=3)['test_score'] )

# prediction
with open('test.csv', newline='') as csvfile:
	rows = csv.reader(csvfile)
	for row in rows:
		if row[0]=="PassengerId" :
			continue

		Y.append(int(row[1])) #Class: survived or not

		to_predict = []
		to_predict.append(int(row[1])) #data1: Pclass

		to_predict.append(Gender[row[3]]) #data2: Gender

		if row[8]=="":
			to_predict.append(0)
		else:
			fare = float(row[8]) #data3: Fare
			for i in range(len(Fare_threshold)):
				if fare > Fare_threshold[i]:
					to_predict.append(i+1)
					break

		to_predict.append(Embarked[row[10]]) #data4: Embarked

		output = []
		output.append(row[0])
		survival = 0
		for clf in classification_models:
			survival += clf.predict([to_predict])[0]
		if survival >= 3 :
			output.append(1)
		else :
			output.append(0)

		Result.append(output)

# output as csv file
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(Result)):
    	writer.writerow(Result[i])