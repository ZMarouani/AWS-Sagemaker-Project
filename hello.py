from flask import Flask , jsonify , render_template , request ,flash, redirect, url_for

from random import sample
import numpy as np
import pandas
import pickle
 
from cleaning import clean
from cleaning import rf_clean 
from getResults import  getResults

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

#do we need an upload folder ? // app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
TEMPLATES_AUTO_RELOAD = True


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
    global results 
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
       
        print('probably the error place ynew:' , ynew)
        #call getResults function or module : Returns json data for charts
        # create var results or data for getResults()
        print('step 3 ')
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