from PIL import Image
from PIL import ImageFilter
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
import svmcrossvalidate

f = open("flower/flower_images/flower_labels.csv")
names = f.readlines()

data = []
labels = []
for i in range(1, len(names), 1):
	names[i] = names[i].strip('\n')
	l = names[i].split(',')
	if(l[1] == "0"):
		labels.append(0)
		img = Image.open("flower/flower_images/"+l[0])
		img = img.convert('L')
		img1 = np.array(img.filter(ImageFilter.FIND_EDGES)).flatten()
		img2 = np.array(img.filter(ImageFilter.DETAIL)).flatten()
		img3 = np.array(img.filter(ImageFilter.CONTOUR)).flatten()
		img4 = np.array(img.filter(ImageFilter.SHARPEN)).flatten()
		img5 = np.array(img.filter(ImageFilter.EDGE_ENHANCE)).flatten()
		img6 = np.array(img.filter(ImageFilter.EDGE_ENHANCE_MORE)).flatten()
		img7 = np.array(img.filter(ImageFilter.SMOOTH_MORE)).flatten()
		img8 = np.array(img.filter(ImageFilter.BLUR)).flatten()
		img9 = np.array(img.filter(ImageFilter.EMBOSS)).flatten()
		data.append(np.concatenate((img1,img2,img3,img4,img5,img6,img7,img8,img9)))
#		img_array = np.array(img)
#		data.append(img_array.flatten())
	elif(l[1] == "5"):
		labels.append(1)
		img = Image.open("flower/flower_images/"+l[0])
		img = img.convert('L')
		img1 = np.array(img.filter(ImageFilter.FIND_EDGES)).flatten()
		img2 = np.array(img.filter(ImageFilter.DETAIL)).flatten()
		img3 = np.array(img.filter(ImageFilter.CONTOUR)).flatten()
		img4 = np.array(img.filter(ImageFilter.SHARPEN)).flatten()
		img5 = np.array(img.filter(ImageFilter.EDGE_ENHANCE)).flatten()
		img6 = np.array(img.filter(ImageFilter.EDGE_ENHANCE_MORE)).flatten()
		img7 = np.array(img.filter(ImageFilter.SMOOTH_MORE)).flatten()
		img8 = np.array(img.filter(ImageFilter.BLUR)).flatten()
		img9 = np.array(img.filter(ImageFilter.EMBOSS)).flatten()
		data.append(np.concatenate((img1,img2,img3,img4,img5,img6,img7,img8,img9)))
#		img_array = np.array(img)
#		data.append(img_array.flatten())                

print(len(data))
print(len(data[0]))

clf = svm.LinearSVC()
#clf = svm.SVC(kernel="poly", degree=3)
scores = cross_val_score(clf, data, labels, cv=10)
print(scores)
print(sum(scores)/len(scores))
#svmcrossvalidate.getbestC(data, labels)
