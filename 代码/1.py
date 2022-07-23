import numpy as np 
x1 = np.random.rand()
x2 =  np.random.rand()
p1 =  np.random.rand()
p2 =  np.random.rand()
alpha =  np.random.rand()
l =  np.random.rand()
M = np.array([
    [0,-p1, -p2],
    [-p1, -alpha/x1**2, 0],
    [-p2, 0, -(1-alpha)/x2**2]
])
b = np.array([x1,l, 0])


answer = np.linalg.solve(M,b)
# x = -(x1*(alpha-1)*(alpha - l*p1*x1))/(alpha * p2**2 * x2**2 - p1**2 * x1**2 * alpha + p1**2 * x1 **2)
# y = (x1**2)*(-l * p2**2 * x2**2 + p1*x1*(alpha-1))/(alpha * p2**2 * x2**2 - p1**2 * x1**2 * alpha + p1**2 * x1 **2)
z = -p2*x1 * x2**2 *(alpha - l*p1*x1) / (alpha * p2**2 * x2**2 - p1**2 * x1**2 * alpha + p1**2 * x1 **2)
print(answer[2] -z )