from sklearn import svm
import random


def get_best_C(train, labels):

    random.seed()
    allCs = [0.001, 0.01, 0.1, 1, 10, 100]
    error = {}
    for j in range(0, len(allCs), 1):
        error[allCs[j]] = 0
    rowIDs = []
    for i in range(0, len(train), 1):
        rowIDs.append(i)
    n_splits = 10
    for x in range(0, n_splits, 1):
        #### Making a random train/validation split of ratio 90:10
        new_train = []
        new_labels = []
        validation = []
        validation_labels = []

        random.shuffle(rowIDs)  # randomly reorder the row numbers
        # print(rowIDs)

        for i in range(0, int(0.9 * len(rowIDs)), 1):
            new_train.append(train[i])
            new_labels.append(labels[i])
        for i in range(int(0.9 * len(rowIDs)), len(rowIDs), 1):
            validation.append(train[i])
            validation_labels.append(labels[i])

        #### Predict with SVM linear kernel for values of C={.001, .01, .1, 1, 10, 100} ###
        for j in range(0, len(allCs), 1):
            C = allCs[j]
            clf = svm.LinearSVC(C=C)
            clf.fit(new_train, new_labels)
            prediction = clf.predict(validation)

            err = 0
            for i in range(0, len(prediction), 1):
                if prediction[i] != validation_labels[i]:
                    err = err + 1

            err = err / len(validation_labels)
            error[C] += err
            # print("err=",err,"C=",C,"split=",x)

    bestC = 0
    min_error = 100
    keys = list(error.keys())
    for i in range(0, len(keys), 1):
        key = keys[i]
        error[key] = error[key] / n_splits
        print("key=", key, " error[key]=", error[key])
        if error[key] < min_error:
            min_error = error[key]
            bestC = key

    # print(bestC,min_error)
    return [bestC, min_error]
