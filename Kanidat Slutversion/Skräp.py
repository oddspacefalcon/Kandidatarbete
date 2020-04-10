import numpy as np 

a = np.array([[-2,-1,-2,0,-1], [-1,-3,-2,-1,-0.1]])
unique, counts = np.unique(a, return_counts=True)
unique_dict = dict(zip(unique, counts))
length_unique_dict = len(unique_dict)


temp = -float('inf')
for i in range(len(a)):
    for j in a[i]:
        if j > temp and j !=0:
            temp = j
index_max = np.where(a==temp)
index_max = (index_max[0][0], index_max[1][0])

print('Index',index_max, ' Max value', a[index_max[0]][index_max[1]])
