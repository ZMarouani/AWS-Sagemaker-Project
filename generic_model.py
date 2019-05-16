import pandas as pd 
import numpy as np 
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.utils import normalize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def training(df , test_data):
    
    # filling missing values
    col_names = df.columns
    for c in col_names:
    df[c] = df[c].replace("?", np.NaN)

    df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

    n_cols = df.shape[1]
    X = df.iloc[:,0:n_cols-1] 
    y = df['y']
    # Get dummies
    X = pd.get_dummies(X, prefix_sep='_', drop_first=True)
    # Split the data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, shuffle= True)
    #from keras.utils import to_categorical
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)

    num_classes = 2 
    input_dim = X_train.shape[1]
    #Build the model 
    model = models.Sequential()
    model.add(layers.Dense(8 , activation='relu', input_dim=input_dim))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(2,activation='softmax'))

    model.compile(optimizer='adam' , loss='categorical_crossentropy' , metrics=['accuracy'])
    model_fit = model.fit(X_train, y_train , epochs=10 , validation_data = ( X_valid , y_valid ))

    score = model.evaluate(X_valid, y_valid, verbose=0)
    loss=score[0]
    acc=score[1]

    #Test set part 
    # filling missing values
    test_df = test_data
    col_names = test_data.columns
    for c in col_names:
        test_data[c] = test_data[c].replace("?", np.NaN)
    test_data = test_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
    n_cols = test_data.shape[1]
    X = test_data.iloc[:,0:n_cols] 
    # Get dummies
    X = pd.get_dummies(X, prefix_sep='_', drop_first=True)
    #prediction
    predictions = model.predict(X)
    pred = []
    for i in predictions:
        pred.append(i[0])
    myarray = np.asarray(pred)
    vect = (myarray==1)
    test_df['y']=vect
    return (acc , loss , test_df )


