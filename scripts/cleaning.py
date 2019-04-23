# cleaning test! data


def clean(file):

    #h1 cleaning part

    # filling missing values
    test_data = file 
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