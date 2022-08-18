from utils import part4Plots

import json



# After examination of first plot I decided to use change learning rate at 432nd step
# as it can be seen in the report
#
# After examination of first plot I decided to use change learning rate at 1728th step
# as it can be seen in the report


# Q4cnn3_1.json == It is json file containing of learning rates 0.1 0.01 0.001
# Q4cnn3_01json == It is json file containing of the single graph where LR changes from 0.1 to 0.01
# Q4cnn3_001json == It is json file containing of the single graph where LR changes from 0.1 to 0.01 and from 0.01 to 0.001
f = open('Q4cnn3_1.json')
 
# returns JSON object as
# a dictionary
data1 = json.load(f)
f.close()

part4Plots(data1, save_dir=r'resultQ4', filename="CNN3_1")