import pandas as pd 
import numpy as np 
import boto3
import io 

def create_client():
    return(boto3.client('sagemaker-runtime',
    aws_access_key_id= 'AKIAYDH4EJCQ6KWMEBGE' ,
    aws_secret_access_key= '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' ,
    region_name='eu-west-1')
    )


def invoke_endpoint(client , df):
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



