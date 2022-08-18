from utils import part3Plots

import json

models=['mlp_1','mlp_2','cnn_3','cnn_4','cnn_5']#Models that will be printed
results=list()
for model in models:
    f = open('q3_'+model+'.json')
     
    # returns JSON object as
    # a dictionary
    results.append(json.load(f))
    f.close()
part3Plots(results, save_dir=r'resultQ3', filename="part3Result")