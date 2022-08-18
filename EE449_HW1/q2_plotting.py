from utils import part2Plots

import json
models=['mlp_1','mlp_2','cnn_3','cnn_4','cnn_5']#Models that will be printed
results=list()
for model in models:
    if(model=='cnn_4'):
        f = open('DUMMY_q2_'+model+'.json')
         
        # returns JSON object as
        # a dictionary
        results.append(json.load(f))
        f.close()        
    else:
        f = open('q2_'+model+'.json')
         
        # returns JSON object as
        # a dictionary
        results.append(json.load(f))
        f.close()
part2Plots(results, save_dir=r'resultQ2', filename='part2Plots___________')
#weight=[data3['weights']]
#visualizeWeights(weight, save_dir='some\location\to\save', filename='input_weights')
