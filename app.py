from flask import Flask , jsonify , render_template , request ,flash, redirect, url_for

from random import sample
import numpy as np
import pandas
import pickle
 
from cleaning import clean , rf_clean , xls_clean
from getResults import  getResults
from getNames import getNames , getCity 
from api import create_client , invoke_endpoint_xgb , invoke_endpoint_fm , invoke_endpoint_ll

#from insertTable import insertTable

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

#do we need an upload folder ? // app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
TEMPLATES_AUTO_RELOAD = True

#an attempt to write the html with python , the problem is that its not dynamic neither would be errased 
@app.route('/data_table')
def dataTable(name=None):
    return render_template('examples/dataTable.html', name=name)


@app.route('/credit_risk')
def test(name=None):
	return render_template('examples/credit_risk.html', name=name)

@app.route('/generic_model')
def generic_model(name=None):
	return render_template('examples/generic_model.html', name=name)
@app.route('/accept_generic_model')
def accept_generic_model(name=None):
    global results 
    if request.method == 'POST':
        file = request.files['file']
        df = pandas.read_csv(file)
        #generic cleaning 
        clean_df =  clean(df).values
        #Build neural network // parameters ??
        nn = build_nn()
        #use predict_classes not predict

        print ("results : " , results)
        return jsonify({
		'results' : results
        
        		})
    else:
        print('GET the problem')
        return jsonify({
            'results' : results
        }) 



@app.route('/')
def hello_world(name=None):
	return render_template('examples/dashboard.html', name=name)

@app.route('/adult_pred')
def adult_pred(name=None):
	return render_template('examples/adult_pred.html', name=name)


#prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,12)
    loaded_model = pickle.load(open("model.pkl","rb"))
    results = loaded_model.predict(to_predict)
    return results[0]

# the submit button => /results
@app.route('/result',methods = ['POST'])
def results():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        results = ValuePredictor(to_predict_list)
        
        if int(results)==1:
            prediction='Income more than 50K'
        else:
            prediction='Income less that 50K'
        return render_template("examples/result.html",prediction=prediction)

# submit file to neural networks prediction 
@app.route('/nn_results', methods = ['POST' , 'GET'])
def nn_results():
    global results 
    if request.method == 'POST':
        file = request.files['file']
        df = pandas.read_csv(file)
        #set months and clean the file 
        months = df['month']
        clean_df =  clean(df).values

        loaded_model = pickle.load(open("nn_model.pkl", "rb"))
        #use predict_classes not predict
        print("model loaded")
        ynew = loaded_model.predict_classes(clean_df)
        print("prediction step ")

        #call getResults function or module : Returns json data for charts
        # create var results or data for getResults()
        
        results = getResults(ynew , months)
        print ("results : " , results)


        #df = pandas.DataFrame(nn_results)
        #dlist = df.values.list() 
        #return render_template("examples/exp.html", data=results )
        return jsonify({
		'results' : results
        
        		})
    else:
        print('GET the problem')
        return jsonify({
            'results' : results
        })

#submit file to Random Forest prediction
@app.route('/rf_results', methods = ['POST' , 'GET'])
def rf_results():
    global rf_results 
    if request.method == 'POST':
        file = request.files['file']
        df = pandas.read_csv(file)
        #set months and clean the file 
        months = df['month']

        #clean_df =  clean(df).values
        #maybe we should use the standardScaler ?
        X_pred =  rf_clean(df)
        print('step 1 ')
        loaded_model = pickle.load(open("rf_model.pkl", "rb"))
        print('step 2 ')
        ynew = loaded_model.predict(X_pred)
       
        print('model.predict' , ynew)
        #call getResults function or module : Returns json data for charts
        # create var results or data for getResults()
        print('step 3 ')
        rf_results = getResults(ynew , months)
        print ("results : " , rf_results)


        #df = pandas.DataFrame(nn_results)
        #dlist = df.values.list() 
        #return render_template("examples/exp.html", data=results )
        return jsonify({
		'results' : rf_results
        
        		})
    else:
        print('GET the problem')
        return jsonify({
            'results' : rf_results
        })

#submit file to SVM lineair model
@app.route('/svm_results', methods = ['POST' , 'GET'])
def svm_results():
    global svm_results 
    if request.method == 'POST':
        file = request.files['file']
        df = pandas.read_csv(file)
        #set months and clean the file 
        months = df['month']

        #clean_df =  clean(df).values
        #maybe we should use the standardScaler ?
        X_pred =  rf_clean(df)
        loaded_model = pickle.load(open("svm_model.pkl", "rb"))
        ynew = loaded_model.predict(X_pred)
       
        print('probably the error place ynew:' , ynew)
        #call getResults function or module : Returns json data for charts
        # create var results or data for getResults()
        print('step 3 ')
        svm_results = getResults(ynew , months)
        print ("results : " , svm_results)


        #df = pandas.DataFrame(nn_results)
        #dlist = df.values.list() 
        #return render_template("examples/exp.html", data=results )
        return jsonify({
		'results' : svm_results
        
        		})
    else:
        print('GET the problem')
        return jsonify({
            'results' : svm_results
        })

#submit file to Logistic regression predictions
@app.route('/lr_results', methods = ['POST' , 'GET'])
def lr_results():
    global lr_results 
    if request.method == 'POST':
        file = request.files['file']
        df = pandas.read_csv(file)
        #set months and clean the file 
        months = df['month']

        #maybe we should use the standardScaler ?
        X_pred =  rf_clean(df)
        print('step 1 ')
        loaded_model = pickle.load(open("lr_model.pkl", "rb"))
        print('step 2 ')
        ynew = loaded_model.predict(X_pred)
       
        print('model.predict' , ynew)
        #call getResults function or module : Returns json data for charts
        # create var results or data for getResults()
        print('step 3 ')
        lr_results = getResults(ynew , months)
        print ("results : " , lr_results)


        #df = pandas.DataFrame(nn_results)
        #dlist = df.values.list() 
        #return render_template("examples/exp.html", data=results )
        return jsonify({
		'results' : lr_results
        
        		})
    else:
        print('didnt GET the problem')
        return jsonify({
            'results' : lr_results
        })



#submit file to XGBoost prediction
@app.route('/xgb_results', methods = ['POST' , 'GET'])
def xgb_results():
    global results 
    if request.method == 'POST':
        file = request.files['file']
        df = pandas.read_csv(file)
        #set months and clean the file 
        months = df['month']
        clean_df =  rf_clean(df) 

        loaded_model = pickle.load(open("xgb_model.pkl", "rb"))
        #use predict_classes not predict
        print("model loaded clean_df" , clean_df)

        ynew = loaded_model.predict(clean_df)
        print("prediction step ynew: " , ynew)

        #call getResults function or module : Returns json data for charts
        # create var results or data for getResults()
        
        results = getResults(ynew , months)
        print ("results : " , results)


        #df = pandas.DataFrame(nn_results)
        #dlist = df.values.list() 
        #return render_template("examples/exp.html", data=results )
        return jsonify({
		'results' : results
        
        		})
    else:
        print('GET the problem')
        return jsonify({
            'results' : results
        })

@app.route('/data')
def data():
	return jsonify({
		'results' : sample(range(1,20),12)
		 })



#CREDIT CARD XLS risk part 

#submit file to Random Forest prediction
@app.route('/rf_xls_results', methods = ['POST' , 'GET'])
def rf_xls_results():
    global rf_xls_results 
    if request.method == 'POST':
        file = request.files['file']
        df = pandas.read_excel(file)
        
        #drop ID and labels 
        df = df.drop('ID')
        #set names  
        names = df['X24']
        last = df['X25']
        email = df['X26']
        city = df['X27']

        #drop the mentioned columns
        X_pred = df.drop(['X24','X25','X26','X27'] , axis=1)
        loaded_model = pickle.load(open("rf_xls_model.pkl", "rb"))
        ynew = loaded_model.predict(X_pred)
        print('model.predict' , ynew)

        #call getResults function or module : Returns json data for charts
        # create var results or data for getResults()
        # A for Array ex: array names 
        (anames,alast,aemail,acity) = getNames(ynew ,names ,last , email , city)
        #we will decide if we want to show them in a table or not 

        rf_xls_results = getCity(ynew , acity)


        return jsonify({
		'results' : rf_xls_results
        
        		})
    else:
        print('GET the problem')
        return jsonify({
            'results' : rf_xls_results
        })

#Invoke endpoint for XGB prediction with SAGE
@app.route('/xgb_xls_results', methods = ['POST' , 'GET'])
def xgb_xls_results():
    global xgb_xls_results 
    if request.method == 'POST':
        file = request.files['file']
        df = pandas.read_excel(file)
        #does this function changes DF ?
        #(names,last,email,city,df) = xls_clean(df)
        
        #drop ID and labels 
        df = df.drop('ID')
        #set needed arrays  
        names = df['X24']
        last = df['X25']
        email = df['X26']
        city = df['X27']
        #drop the previously mentioned columns
        X_pred = df.drop(['X24','X25','X26','X27'] , axis=1)
        
        #time to invoke the sage endpoint
        client = create_client()
        ynew = invoke_endpoint_xgb( client , X_pred )
        print('model.predict' , ynew)

        #call getNames function or module : Returns json data for  RADAR charts
        # create var results or data for getResults()
        # A for Array ex: array names 
        (anames,alast,aemail,acity) = getNames(ynew ,names ,last , email , city)
        #we will decide if we want to show them in a table or not 

        xgb_xls_results = getCity(ynew , acity)


        return jsonify({
		'results' : xgb_xls_results
        
        		})
    else:
        print('GET the problem')
        return jsonify({
            'results' : xgb_xls_results
        })

#Invoke endpoint for Linear Learner prediction with SAGE
@app.route('/ll_xls_results', methods = ['POST' , 'GET'])
def ll_xls_results():
    global ll_xls_results 
    if request.method == 'POST':
        file = request.files['file']
        df = pandas.read_excel(file)
        #does this function changes DF ?
        #(names,last,email,city,df) = xls_clean(df)
        
        #drop ID and labels 
        df = df.drop('ID')
        #set needed arrays  
        names = df['X24']
        last = df['X25']
        email = df['X26']
        city = df['X27']
        #drop the previously mentioned columns
        X_pred = df.drop(['X24','X25','X26','X27'] , axis=1)
        
        #time to invoke the sage endpoint
        client = create_client()
        ynew = invoke_endpoint_ll( client , X_pred )
        print('model.predict' , ynew)

        #call getNames function or module : Returns json data for  RADAR charts
        # create var results or data for getResults()
        # A for Array ex: array names 
        (anames,alast,aemail,acity) = getNames(ynew ,names ,last , email , city)
        #we will decide if we want to show them in a table or not 

        ll_xls_results = getCity(ynew , acity)


        return jsonify({
		'results' : ll_xls_results
        
        		})
    else:
        print('GET the problem')
        return jsonify({
            'results' : ll_xls_results
        })


#Invoke endpoint for Factorization machhine prediction with SAGE
@app.route('/fm_xls_results', methods = ['POST' , 'GET'])
def fm_xls_results():
    global fm_xls_results 
    if request.method == 'POST':
        file = request.files['file']
        df = pandas.read_excel(file)
        #does this function changes DF ?
        #(names,last,email,city,df) = xls_clean(df)
        
        #drop ID and labels 
        df = df.drop('ID')
        #set needed arrays  
        names = df['X24']
        last = df['X25']
        email = df['X26']
        city = df['X27']
        #drop the previously mentioned columns
        X_pred = df.drop(['X24','X25','X26','X27'] , axis=1)
        
        #time to invoke the sage endpoint
        client = create_client()
        ynew = invoke_endpoint_fm( client , X_pred )
        print('model.predict' , ynew)

        #call getNames function or module : Returns json data for  RADAR charts
        # create var results or data for getResults()
        # A for Array ex: array names 
        (anames,alast,aemail,acity) = getNames(ynew ,names ,last , email , city)
        #we will decide if we want to show them in a table or not 

        fm_xls_results = getCity(ynew , acity)


        return jsonify({
		'results' : fm_xls_results
        
        		})
    else:
        print('GET the problem')
        return jsonify({
            'results' : fm_xls_results
        })






# Python code to flat a nested list with 
# multiple levels of nesting allowed. 

# output list 
output = [] 

# function used for removing nested 
# lists in python. 
def reemovNestings(l): 
	for i in l: 
		if type(i) == list: 
			reemovNestings(i) 
		else: 
			output.append(i) 



#if __name__ == '__main__':
#    app.run(debug=False)