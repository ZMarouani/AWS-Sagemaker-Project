import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
# cleaning test! data

def clean(df):

    #h1 cleaning part

    # filling missing values
    test_data = df 
    col_names = test_data.columns
    for c in col_names:
        test_data[c] = test_data[c].replace("?", np.NaN)

    test_data = test_data.apply(lambda x:x.fillna(x.value_counts().index[0]))

    #discretisation
    test_data.replace(['pdays'],
                ['days'],
            inplace = True)

    #label Encoder
    category_col =['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome'] 
    labelEncoder = preprocessing.LabelEncoder()

    # creating a map of all the numerical values of each categorical labels.
    mapping_dict={}
    for col in category_col:
        test_data[col] = labelEncoder.fit_transform(test_data[col])
        le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
        mapping_dict[col]=le_name_mapping
    print(mapping_dict)




    X_test = test_data
    X_test_norm = (X_test - X_test.mean()) / (X_test.max() - X_test.min())
    #X_test_norm.head()

    return (X_test_norm)

def rf_clean(test_data):
    # filling missing values
    col_names = test_data.columns
    for c in col_names:
        test_data[c] = test_data[c].replace("?", np.NaN)

    test_data = test_data.apply(lambda x:x.fillna(x.value_counts().index[0]))

    #discretisation
    test_data.replace(['pdays'],
                ['days'],
            inplace = True)

    #label Encoder
    category_col =['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome'] 
    labelEncoder = preprocessing.LabelEncoder()

    # creating a map of all the numerical values of each categorical labels.
    mapping_dict={}
    for col in category_col:
        test_data[col] = labelEncoder.fit_transform(test_data[col])
        le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
        mapping_dict[col]=le_name_mapping
    print(mapping_dict)


    sc = StandardScaler()
    X_train = sc.fit_transform(test_data)
    X_pred = sc.transform(test_data)

    return(X_pred)

def xls_clean(df):
    #drop ID and labels 
    df = df.drop('ID')
    #set needed arrays  
    names = df['X24']
    last = df['X25']
    email = df['X26']
    city = df['X27']
    #drop the previously mentioned columns
    X_pred = df.drop(['X24','X25','X26','X27'] , axis=1)
    return(names,last,email,city , df)
