import numpy as np
import pandas as pd 
import pickle
from utilis import cleaning,embed

model = pickle.load(open("model.pkl","rb"))

test_input=input()
li=[]
li.append(test_input)
test_embed=embed(li)

predicted=model.predict(test_embed)
print(predicted)