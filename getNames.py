from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import calendar
from collections import Counter
import json



    #get months in order 
    #months= pd.read_csv('banking-batch.csv')['month']

    # make a prediction
    #ynew = model.predict_classes(X_test_norm)
def getNames (ynew , names , last , email , city) : 
    value_name = []
    value_last = []
    value_email = []
    value_city = []

    for i in range(len(ynew)):
        if(ynew[i]==1):
            value_name.append(names[i])
            value_last.append(last[i])
            value_email.append(email[i])
            value_city.append(city[i])


    return (value_name,value_last,value_email,value_city )

def getCity (ynew , value_city):
    p = Counter(value_city)

    results_list = []
    for i in p :
        results_list.append(p[i])

    return(results_list)