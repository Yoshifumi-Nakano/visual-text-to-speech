import numpy as np
a=np.array([1,2,3,4,5,6,7])


s=2
e=4
a=a.tolist()
for i in range(len(a)):
    if a[i]<s or a[i]>e:
        a[i]=a[i]/2
a=np.array(a)
print(a)
