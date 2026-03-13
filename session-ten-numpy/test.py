import numpy as np 

# a = np.array([1,2,3])
# b = np.array([4,5,6])

# c = a+b
# print(c)

#when we used numpy we need same parameters in the both the array then we can make calculation 
# d = np.array([1,2])
# e = np.array([3,4,5])
# f = d+e
# print(f)

# reshape

arr = np.array([1,2,3,4,5,6])

reshaped = arr.reshape(2,3)
print(reshaped)
print(type(arr))

convert_to_list = list(arr)
print(type(convert_to_list))