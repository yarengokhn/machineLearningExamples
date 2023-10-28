import numpy as np
from numpy.linalg import inv

#Question 9.a
A=np.matrix([[0, 2,4], [2,4,2],[3,3,1]])
Ainv = inv(A)
#print(Ainv)


#Question 9.b.1
b = np.matrix([-2, -2,-4]).transpose()
#print(b)
result=np.matmul(Ainv, b)
#print(result)


#Question 9.b.2
c =np.matrix([1,1,1]).transpose()
result=np.matmul(A, c)
print(result)


