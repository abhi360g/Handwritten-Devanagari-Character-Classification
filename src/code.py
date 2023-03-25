'''
Abhijeet Ghadge
G01274854
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from numpy import asarray
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from PIL import Image
from numpy import asarray
from sklearn.metrics import accuracy_score
from skimage.morphology import skeletonize
from skimage import morphology
from sklearn.model_selection import RepeatedStratifiedKFold
from keras.utils import np_utils


k_values = [3,5,7,9,11,15,21,25,27,31,35,37,45,55,65,75,85,105,125,145]
# k_values = [5]

def image_thinning(img):
    
    kernel = np.ones((2,2),np.uint8)
    #thin_image = cv2.erode(img,kernel,iterations = 1)
    thin_image = cv2.dilate(img,kernel,iterations = 1)
    

    
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)

    # ax = axes.ravel()
    
    # ax[0].imshow(img, cmap=plt.cm.gray)
    # ax[0].axis('off')
    # ax[0].set_title('Original image', fontsize=20)
    
    # ax[1].imshow(thin_image, cmap=plt.cm.gray)
    # ax[1].axis('off')
    # ax[1].set_title('Modified image', fontsize=20)
    
    # fig.tight_layout()
    # plt.show()
    # print("Exiting thinning")
    return thin_image
    

def reader(dir_path):
    print("Reading data")
    counter = 0
    pixel_data_list = []
    labels = []
    os.chdir(dir_path)
    for char_dir in os.listdir():
        print(char_dir)
        os.chdir(char_dir)
        for image in os.listdir():
            #img = Image.open(image)
            counter += 1
            print(counter)
            img = cv2.imread(image,0)
            # thin_image = img
            thin_image = image_thinning(img)                    #Thinning or skeletonization of the image
            image_pixel_data = asarray(thin_image)
            pixel_data_list.append(image_pixel_data.flatten().tolist())
            split_dir_name = char_dir.split('_')
            labels.append(split_dir_name[-1])
            # plt.imshow(image_pixel_data, cmap='gray')
            # plt.show()
        os.chdir("..")
    os.chdir("..")
    return pixel_data_list,labels

def data_scaler(x_train,x_test):
    print('Data scaler')
    scaler = StandardScaler()              #for TSNE
    #scaler = RobustScaler()                
    #scaler = MinMaxScaler()                 #for PCA
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print("Done scaling")
    return x_train,x_test

def data_PCA(train_pixel_data_list, test_pixel_data_list):
    #PCA for dimensionality reduction
    print("PCA")
    pca_train = PCA(n_components = 600)
    
    print("train")
    pca_train.fit(train_pixel_data_list)
    train_pixel_data_list = pca_train.transform(train_pixel_data_list)
    train_pixel_data_list = train_pixel_data_list.tolist()
    
    
    print("test")
    pca_test = PCA(n_components = 600)
    pca_test.fit(test_pixel_data_list)
    test_pixel_data_list = pca_test.transform(test_pixel_data_list)
    test_pixel_data_list = test_pixel_data_list.tolist()
    
    print("Done with PCA")
    return train_pixel_data_list,test_pixel_data_list


def data_TSNE(train_pixel_data_list, test_pixel_data_list,labels_train, labels_test):
    print("TSNE")
    #n_sne = 10000
    #labels_unique = list(set(labels_train))
    
    feat_cols = ['Pixel'+ str(i) for i in range(600)]
    
    print("train")
    df_train_pixel_data_list = pd.DataFrame(train_pixel_data_list,columns = feat_cols)
    # df_train_pixel_data_list['y'] = labels_train
    # df_train_pixel_data_list['label'] = df_train_pixel_data_list['y'].apply(lambda i: str(i))
    # y = None
    
    randpermutation = np.random.permutation(df_train_pixel_data_list.shape[0])
    df_tsne_train = df_train_pixel_data_list.loc[randpermutation[:],:].copy()
    tsne = TSNE(n_components=150, method = 'exact', verbose=1, perplexity=50, n_iter=1000)
    df_train_pixel_data_list = tsne.fit_transform(df_train_pixel_data_list.loc[randpermutation[:],feat_cols].values)
    
    df_test_pixel_data_list = pd.DataFrame(test_pixel_data_list,columns = feat_cols)
    randpermutation = np.random.permutation(df_test_pixel_data_list.shape[0])
    df_tsne_test = df_test_pixel_data_list.loc[randpermutation[:],:].copy()
    tsne = TSNE(n_components=150, method = 'exact', verbose=1, perplexity=50, n_iter=1000)
    df_test_pixel_data_list = tsne.fit_transform(df_test_pixel_data_list.loc[randpermutation[:],feat_cols].values)
    
    ''' #Scatterplot for TSNE
    df_tsne_train['x-tsne'] = df_train_pixel_data_list[:,0]
    df_tsne_train['y-tsne'] = df_train_pixel_data_list[:,1]
        
    plt.figure(figsize=(16,10))
    sns.scatterplot(x="x-tsne", y="y-tsne", hue = "y", palette = sns.color_palette("hls", 46), data = df_tsne_train, legend = "full", alpha = 0.3)
    '''
    print("Done TSNE")
    return df_train_pixel_data_list.tolist(),df_test_pixel_data_list.tolist()
    
def label_encoder(data):
    le = preprocessing.LabelEncoder()
    data = le.fit_transform(data)
    return data

def data_UMAP(train_pixel_data_list, test_pixel_data_list,labels_train, labels_test):
    print("UMAP dimensionality reduction")

    train_umap = umap.UMAP(n_components = 500,n_neighbors = 5, min_dist = 0.3, metric='correlation').fit_transform(train_pixel_data_list)
    test_umap = umap.UMAP(n_components = 500,n_neighbors = 5, min_dist = 0.3, metric='correlation').fit_transform(test_pixel_data_list)
    
    '''#Plot for UMAP
    plt.figure(figsize=(12,12))
    plt.scatter(train_umap[:,0],train_umap[:,1], c = label_encoder(labels_train), edgecolor='none', alpha=0.80, s=10)
    plt.scatter(train_umap[:,0],train_umap[:,1], c = label_encoder(labels_train), cmap = 'Spectral', s = 5)
    '''
    print("Done UMAP")
    return train_umap.tolist(),test_umap.tolist()

def KNN(k_values,train_pixel_data_list,test_pixel_data_list,labels_train,labels_test):
    print("KNN classifier")
    predictions = []
    accuracy_list = []
    for k in k_values:              #Calcualte accuracies for different values of k using KNN
        print("Current value of k: ",k)
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(train_pixel_data_list,labels_train)
        predictions = classifier.predict(test_pixel_data_list)
        accuracy_list.append(accuracy_calculator(labels_test,predictions))
    return accuracy_list


def SVM(train_pixel_data_list,test_pixel_data_list,labels_train,labels_test):
    print("SVM classifier")
    predictions = []
    clf = svm.SVC(kernel='linear', random_state = 42)
    print("clf done")
    clf.fit(train_pixel_data_list,labels_train)
    print("Fit done")
    predictions = clf.predict(test_pixel_data_list)
    print("SVM predictions done")
    accuracy_calculator(labels_test,predictions)
    
def random_forest(x_train,x_test,y_train,labels_test):
    print('Random forest')
    accuracy_list = []
    # n_estimators_list = [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100]

    n_estimators_list = [1100]    
    for trees in n_estimators_list:
        print("Trees ",trees)
        # clf1 = RandomForestClassifier(n_estimators = trees, max_depth = 46, random_state=42, max_features = 100, criterion='gini')
        clf1 = RandomForestClassifier(n_estimators = trees)
        clf1.fit(x_train, y_train)
        print("Done with fitting")
        predictions = clf1.predict(x_test)
        print("Done with predictions")
        accuracy = accuracy_calculator(labels_test,predictions)
        accuracy_list.append(accuracy)
        
    '''#Graph
    print("Plotting graph")
    print("Accuracy ", accuracy_list)
    plt.plot(n_estimators_list,accuracy_list)
    plt.xlabel('No of estimators')
    plt.ylabel('Accuracy')
    plt.title('Graph of Accuracy against estimators')
    plt.show()
    '''    
    
def conv_nets(train_pixel_data_list,test_pixel_data_list,labels_train,labels_test):
    print('CNN')
    labels_train = label_encoder(labels_train)
    labels_test = label_encoder(labels_test)
    
    train_pixel_data_list = train_pixel_data_list.reshape(np.shape(train_pixel_data_list)[0],32,32,1)
    test_pixel_data_list = test_pixel_data_list.reshape(np.shape(test_pixel_data_list)[0],32,32,1)
        
    x_train,x_test,y_train,y_test = train_test_split(train_pixel_data_list,labels_train,test_size = 0.1, random_state = 42)

    model = Sequential()

    model.add(Conv2D(30, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    print("1st Conv2d")
    model.add(MaxPooling2D())
    print("1st MaxPooling2D")
    model.add(Conv2D(15, (3, 3), activation='relu'))
    print("2nd Conv2d")
    model.add(MaxPooling2D())
    print("2nd MaxPooling2D")
    model.add(Dropout(0.2))
    print("Dropout")
    model.add(Flatten())
    print("Flatten done")
    model.add(Dense(128, activation='relu'))
    print("Added 1st Dense")
    model.add(Dense(80, activation='relu'))
    print("Added 2st Dense")
    model.add(Dense(46, activation='softmax'))      # 46 = number of classes
    print("Added 3rd Dense")
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Model compiled")
    
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=35, batch_size=200)
    print("Model fitted")
    predictions = model.predict(test_pixel_data_list)
    print("Model predications done")
    scores = model.evaluate(test_pixel_data_list, labels_test, verbose=0)
    print("Model evaluated")
        
    #Plot graph
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1,36)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    
    loss_train1 = history.history['accuracy']
    loss_val1 = history.history['val_accuracy']
    epochs = range(1,36)
    plt.plot(epochs, loss_train1, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val1, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    print(labels_test)
    print(np.shape(labels_test))
    print(predictions)
    print(np.shape(predictions))
    predictions = np.argmax(predictions,axis = 1)
    print(predictions)
    print(np.shape(predictions))
    
    # labels_test_set = []
    # labels_test_set = list(set(labels_test))
    # print("labels",labels_test)
    # print("Predictions",predictions)
    # conf_matrix = confusion_matrix(labels_test,predictions,labels=labels_test_set)
    # print('Confusion matrix : \n',conf_matrix)
    # conf_matrix_display=ConfusionMatrixDisplay(conf_matrix,display_labels=labels_test_set).plot()
    # conf_matrix_display.ax_.set(title='Confusion Matrix', xlabel='Predicted Labels', ylabel='Actual Labels')
    
    accuracy_calculator(labels_test,predictions)
    
def random_forest_RandomSearch(train_pixel_data_list,test_pixel_data_list,labels_train):
    print("Hyperparameter optimization for random forest")
    space = dict()
    model = RandomForestClassifier()
    
    space['n_estimators'] = [100,200,400,1000]
    space['max_features'] = ['auto', 'sqrt']
    space['max_depth'] = [int(x) for x in np.linspace(10, 110, num = 11)]
    space['max_depth'].append(None)
    space['min_samples_split'] = [2, 5, 10]
    space['min_samples_leaf'] = [1, 2, 4]
    space['bootstrap'] = [True, False]
    search = RandomizedSearchCV(estimator = model, param_distributions = space, n_iter = 3, scoring='accuracy', n_jobs = -1, cv = 3, random_state = 42)
    print("Now fitting...")
    result = search.fit(train_pixel_data_list, labels_train)
    
    print('Best Score for SVM: %s' % result.best_score_)
    print('Best Hyperparameters for SVM: %s' % result.best_params_)

def accuracy_calculator(labels_test,predictions):
    print("Classification Report:")
    accuracy = []
    print(classification_report(labels_test,predictions))      #Classification report
    
    print("Recall")
    # print(recall_score(labels_test,predictions,average = 'None'))
    print(recall_score(labels_test,predictions,average = 'macro'))
    print(recall_score(labels_test,predictions,average = 'micro'))
    print(recall_score(labels_test,predictions,average = 'weighted'))
    
    print("Precision")
    # print(precision_score(labels_test,predictions,average = 'None'))
    print(precision_score(labels_test,predictions,average = 'macro'))
    print(precision_score(labels_test,predictions,average = 'micro'))
    print(precision_score(labels_test,predictions,average = 'weighted')) 
    
    
    print("F-1 score")
    # print(f1_score(labels_test,predictions,average = 'None'))
    print(f1_score(labels_test,predictions,average = 'macro'))
    print(f1_score(labels_test,predictions,average = 'micro'))
    print(f1_score(labels_test,predictions,average = 'weighted')) 
        
    
    
    print("Accuracy")
    print(accuracy_score(labels_test,predictions))            #Accuracy
    # accuracy = accuracy_score(labels_test,predictions)          #Accuracy
    # print(accuracy)
    return accuracy

def plot_graph(k_values,accuracy_list):
    print("Plotting graph")
    plt.plot(k_values,accuracy_list)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Graph of Accuracy against K')
    plt.show()
    
def dimensionality_reduction_component_variance_plot(train_pixel_data_list, test_pixel_data_list):
    print("Plotting variance graph")
    pca = PCA()
    #pca.fit(train_pixel_data_list)     #Training data
    pca.fit(test_pixel_data_list)       #Test data
    plt.figure(figsize = (10,8))
    plt.plot(range(1,1025),pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
    plt.title('Explained Variance by Components for Test data')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative explained variance')

if __name__=="__main__":
    train_pixel_data_list = []
    test_pixel_data_list = []
    labels_train = []
    labels_test = []
    predictions = []
    accuracy_list = []

    train_pixel_data_list,labels_train = reader("Train")
    print("Training data read complete")
    test_pixel_data_list,labels_test = reader("Test")
    print("Test data read complete")
    
  
    
    '''#Scaling'''
    train_pixel_data_list,test_pixel_data_list = data_scaler(train_pixel_data_list, test_pixel_data_list)
    
    '''#Dimensionality reduction'''
    #train_pixel_data_list,test_pixel_data_list = data_PCA(train_pixel_data_list, test_pixel_data_list)
    #train_pixel_data_list,test_pixel_data_list = data_TSNE(train_pixel_data_list, test_pixel_data_list, labels_train, labels_test)
    #train_pixel_data_list,test_pixel_data_list = data_UMAP(train_pixel_data_list, test_pixel_data_list,labels_train, labels_test)
    
    
    
    '''#Hyperparameter optimzation'''
    #random_forest_RandomSearch(train_pixel_data_list,test_pixel_data_list,labels_train)
    
    '''#Cross validation'''
    # x_train,x_test,y_train,y_test = train_test_split(train_pixel_data_list,labels_train,test_size = 0.2, random_state = 42)
    # accuracy_list = KNN(k_values,x_train,x_test,y_train,y_test)
    # SVM(x_train,x_test,y_train,y_test)
    # random_forest(x_train,x_test,y_train,y_test)
    
    
    '''#Models'''
    # accuracy_list = KNN(k_values,train_pixel_data_list,test_pixel_data_list,labels_train,labels_test)
    # SVM(train_pixel_data_list,test_pixel_data_list,labels_train,labels_test)
    # random_forest(train_pixel_data_list,test_pixel_data_list,labels_train,labels_test)
    conv_nets(train_pixel_data_list,test_pixel_data_list,labels_train,labels_test) 

 
    '''#Graphs'''
    # plot_graph(k_values,accuracy_list)      #Plot graph of Accuracy againts values of K
    #dimensionality_reduction_component_variance_plot(train_pixel_data_list, test_pixel_data_list)
    