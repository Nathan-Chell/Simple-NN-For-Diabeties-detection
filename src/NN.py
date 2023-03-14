#   Train and test a neural network
#   To identify diabeties in patients

import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier

def get_data():

    cur_path = os.getcwd()
    parent_path = os.path.dirname(cur_path)
    data_path = os.path.join(parent_path, 'data')
    
    data = pd.read_csv(os.path.join(data_path, 'diabetes.csv'))
    
    X = data.iloc[:, 2:13].values
    y = data.iloc[:, 13].values
    
    df_X = pd.DataFrame(X, columns=['Gender', 'Age', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI'])
    #   Male: 0, Female: 1
    df_X['Gender'].replace(['M', 'm', 'F', 'f'], [0, 0, 1, 1], inplace=True)
    
    
    df_Y = pd.DataFrame(y, columns=['Class'])
    
    #   Normal: 0, Pre-diabetes: 1, Diabetes: 2
    df_Y['Class'].replace(['N', 'N ', 'P', 'P ', 'Y', 'Y '], [0, 0, 1, 1, 2, 2], inplace=True)
    
    return df_X, df_Y

def Split_data(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    return X_train, X_test, y_train, y_test

def train_MLP(X_train, y_train, X_test, y_test, activation='relu', layers=(100,)):
    
    model = MLPClassifier(hidden_layer_sizes=layers, max_iter=2000, solver='adam', activation=activation, random_state=0)
    model.fit(X_train, y_train.values.ravel())
    
    print("Hidden layer size: {}".format(layers))
    print("Test accuracy: {}".format(accuracy_score(y_test, model.predict(X_test))))
    print("Train accuracy: {}".format(accuracy_score(y_train, model.predict(X_train))))
    #print("Confusion matrix:\n {}".format(confusion_matrix(y_test, model.predict(X_test))))

    return accuracy_score(y_test, model.predict(X_test))
    
def main():

    X, y = get_data()
    X_train, X_test, y_train, y_test = Split_data(X, y)
    
    
    activations = ['identity', 'logistic', 'tanh', 'relu']
    best = []
    for j in activations:
        print("{}\n".format(j))
        
        highest_accuracy = 0
        for i in range(8, 30):
            accuracy = train_MLP(X_train, y_train, X_test, y_test, activation=j, layers=(i,))
            print("\n")
            
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_layer = i
                
        best.append("Activation: {} Best accuracy: {} for layer size: {}".format(j, highest_accuracy, best_layer))

    for i in best:
        print(i)
    
if __name__ =='__main__':
    main()