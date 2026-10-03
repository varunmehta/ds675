from sklearn import svm
import svm_cross_validate
from sklearn import preprocessing

#datafile = sys.argv[1]
datafile = "data2.txt"
f = open(datafile, 'r')
data = []
i = 0
l = f.readline()

##################
### Read data ####
##################
while(l != ''):
	a = l.split()
	l2 = []
	for j in range(0, len(a), 1):
		l2.append(float(a[j]))
	data.append(l2)
	l = f.readline()

rows = len(data)
cols = len(data[0])

###########################
### Read all labels ##
###########################
#labelfile = sys.argv[3]
labelfile = "labels2.txt"
f = open(labelfile)
labels_d = {}
labels = f.readlines()
for i in range(0, len(labels), 1):
        labels[i] = int(labels[i])
        if(labels[i] == -1):
                labels[i] = 0
        labels_d[i] = labels[i]

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(data)
clf = svm.LinearSVC(C=100, max_iter=100000)
Y = labels
clf.fit(X,Y)
w = clf.coef_
w0 = clf.intercept_
print(w,w0)
