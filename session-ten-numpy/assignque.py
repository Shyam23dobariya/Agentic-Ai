import numpy as np 

arr = np.array([10,20,30,40])

print("original data :",arr)

armean = np.mean(arr)

print("Mean :", armean)

stadiv = np.std(arr)

print("standard daviation: ",stadiv)

nor = (arr-armean)/stadiv

print("normalized data :", nor)

shaped = arr.shape
print("your shaped data is here : ",shaped)

reshaped = arr.reshape(2,2)
print("reshaped :",reshaped)