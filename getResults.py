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
def getResults (ynew , months) : 
    
    month =['jan','feb','mar','apr' ,'may','jun' , 'jul','aug','sep','oct','nov','dec' ] 
    value = []

    for i in range(len(ynew)):
        if (ynew[i] == 1 ):
            month.append(months[i])
                    

    #data_dict = [{t, s} for t, s in zip(month, value)]

    p = Counter(month)

    results_list =[]
    for i in p:
        results_list.append(p[i])

    #results = {"results" : results_list}

    # Printing in JSON format
    #print (json.dumps(results))

    return (results_list)

