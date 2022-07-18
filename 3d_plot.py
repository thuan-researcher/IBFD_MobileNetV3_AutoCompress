from turtle import color
import matplotlib.pyplot as plt
import numpy as np

def lin(start, stop, num):
    step = (start-stop)/num
    rst = []
    for i in range (num):
        rst.append(start+step)
    return rst


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x=[]
y=[]
z=[]

x.append([11.75,	10.339,	9.189	,8.040	,6.887,	5.743,	4.612])
y.append([328.352	,309.343,	292.896	,268.905	,249.326	,227.050	,178.420])
z.append ([ 99.85 ,	 99.76, 	 99.71 ,	 99.42 	, 99.18, 	 89.16 ,	 78.42])
x.append([ 11.75, 	 10.34, 	 9.19, 	 8.04, 	 6.89, 	 5.74, 	 4.59  ])
y.append([ 85.90 	, 81.56 ,	 76.62 	, 67.29 	, 66.25 	 ,59.05 ,	 11.75 ])
z.append([ 99.86 ,	 99.77 ,	 99.58, 	 99.35 	 ,98.04 	, 96.21 	, 46.68 ])
x.append([ 11.61 	, 10.21 	, 9.08 ,	 7.94 	, 6.81 ,	 5.67 ,	 4.55 ])
y.append([ 180.90 ,	 176.47 ,	 172.08 ,	 167.22 	, 158.97 	, 150.93 	, 137.73 ])
z.append([ 99.91 	 ,99.72 	, 99.76 	, 99.81 ,	 99.15 ,	 85.17 	, 80.51 ])
x.append([ 11.61 	 ,10.22 ,	 9.09 	, 7.94 ,	 6.81, 	 5.67 ,	 4.54 ])
y.append([ 49.04 ,	 46.18 	, 43.35 ,	 42.72 ,	 39.98 ,	 38.82 ,	 37.18 ])
z.append([ 99.86 	, 99.91 	, 99.34 	, 96.43 	, 92.20 ,	 89.95 	, 84.17 ])
s = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for i in range(4):
    for j in range(7):
        ax.scatter(x[i][j], y[i][j], z[i][j], marker='o', color='r' )
        if j<6:
            xs = np.linspace(np.minimum(x[i][j], x[i][j+1]), np.maximum(x[i][j], x[i][j+1]), 100)
            ys = np.linspace(np.minimum(y[i][j], y[i][j+1]), np.maximum(y[i][j], y[i][j+1]), 100)
            zs = np.linspace(np.minimum(z[i][j], z[i][j+1]), np.maximum(z[i][j], z[i][j+1]), 100)
            ax.plot(xs, ys, zs, color='b')
ax.set_xlabel('Parameters')
ax.set_ylabel('MACs')
ax.set_zlabel('Accuracy')
ax.plot(xs, ys, zs, color='b')
plt.show()