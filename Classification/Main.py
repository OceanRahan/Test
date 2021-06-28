import pandas as pd
from sklearn import svm
import pandas.plotting as pdplot
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

df_raw=pd.read_csv("example_data.csv")
df=df_raw.loc[ : , df_raw.columns != 'class_label']

x = df.values
x = np.array(x)
y = df_raw['class_label'].values
for i in range(0, len(y)):
    if 'R' in y:
        y[i] = 1
    else:
        y[i] = 0


def clustering():

    pca=PCA(n_components=10)
    df_transformed=pca.fit_transform(df)
    df_transformed=pd.DataFrame(df_transformed)
    pdplot.scatter_matrix(df_transformed)
    kmeans = KMeans(2)
    df_transformed['cluster'] = kmeans.fit_predict(df_transformed[[1,3]])
    print(df_transformed)
    centroid=kmeans.cluster_centers_
    print(centroid)
    plt.scatter(centroid[0][0],centroid[0][1])
    plt.scatter(centroid[1][0],centroid[1][1])
    plt.scatter(df_transformed[1],df_transformed[3],c=df_transformed['cluster'])
    plt.show()


def classification_svm_kfold():

    kf=KFold(n_splits=5)
    classifier = svm.SVC(kernel='linear')
    scores=[]
    for index_train, index_test in kf.split(x):
        x_train = x[index_train]
        x_test = x[index_test]
        y_train = y[index_train]
        y_test = y[index_test]
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        print(classifier.score(x_test,y_test))


def svm_train_test_split():

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=109)
    y_train=y_train.astype(int)
    y_test=y_test.astype(int)
    classifier=svm(kernel='linear')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(classifier.score(X_test, y_test))

def Neural_Network():

    kf = KFold(n_splits=5)
    i=1
    for index_train, index_test in kf.split(x):
        x_train = x[index_train]
        x_test = x[index_test]
        y_train = y[index_train]
        y_test = y[index_test]
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        ''''
        model = Sequential()
        model.add(Dense(128, input_shape=(x_train.shape[1],), activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(64, activation='relu'))
        model.add((Dropout(0.4)))
        model.add(Dense(32, activation='relu'))
        model.add((Dropout(0.4)))
        model.add(Dense(1, activation='sigmoid'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        trained_model = model.fit(x_train, y_train, epochs=100, batch_size=5, verbose=1)

        model.save("model"+str(i)+".h", trained_model)
        i+=1
'''''
        model1=load_model('model2.h')
        res = model1.evaluate(x_test, y_test)
        y_pred=model1.predict_classes(x_test)
        y_pred=y_pred.reshape(1,-1)

        count=0

        for i in range(0,len(y_test)):
            if y_test[i]==y_pred[0][i]:
                count+=1
        acc=(count/len(y_test))*100
        print(acc,"%")
        print("actual: ",y_test)
        print("Predicted: ", y_pred)







#clustering()
#classification_svm_kfold()
#svm_train_test_split()
Neural_Network()