import pandas as pd 
import numpy as np 
import boto3
import io 
import json

def create_client():
    return(boto3.client('sagemaker-runtime',
    aws_access_key_id= ' ' ,
    aws_secret_access_key= ' ' ,
    region_name='eu-west-1')
    )


def invoke_endpoint_xgb(client , df):
    test_file = io.StringIO()
    df.to_csv(test_file , header=None , index=None)
    
    response = client.invoke_endpoint(
    EndpointName='xgboost-2019-04-27-14-45-18-653',
    Body=test_file.getvalue(),
    ContentType='text/csv',
    Accept='Accept')

    predictions = response['Body'].read().decode('ascii')
    predictions = np.fromstring(predictions[1:] , sep=',')
    predictions = np.round(predictions)

    return(predictions)

def invoke_endpoint_ll(client , df):
    test_file = io.StringIO()
    df.to_csv(test_file , header=None , index=None)
    
    response = client.invoke_endpoint(
    EndpointName='linear-learner-2019-05-12-17-36-25-645',
    Body=test_file.getvalue(),
    ContentType='text/csv',
    Accept='Accept')

    pred = response['Body'].read().decode('ascii')
    predloads = json.loads(pred)
    prediction_list = []
    for i in predloads['predictions']:
        prediction_list.append(i['predicted_label'])

    return(prediction_list)

def invoke_endpoint_fm(client , df):
    test_file = io.StringIO()
    df.to_csv(test_file , header=None , index=None)
    
    response = client.invoke_endpoint(
    EndpointName='factorization-machines-2019-05-12-17-35-35-473',
    Body=test_file.getvalue(),
    ContentType='text/csv',
    Accept='Accept')

    predictions = response['Body'].read().decode('ascii')
    predictions = np.fromstring(predictions[1:] , sep=',')
    predictions = np.round(predictions)

    return(predictions)
