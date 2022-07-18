import matplotlib.pyplot as plt
import numpy as np

def ToPercentage(x):
    return x/800
x=[]
y=[]
z=[]

x.append([  11754 ,	10338,	9185,	8044,	6890,	5744])  #param
y.append([ 85904,	81564,	76616,	67291,	66251,	59045, ]) #macs
z.append([ 99860 ,	 99770 ,	 99580, 	 99350 	 ,98040 	, 96210 ]) #acc

x.append([ 11610,	10223,	9091,	7942,	6806,	5668])
y.append([ 49040,	46175,	43351,	42721,	39980,	38818 ])
z.append([ 99860 	, 99910 	, 99340 	, 96430 	, 92200 ,	 89950  ])
annotation = ['0', '0.1', '0.2', '0.3', '0.4', '0.5']
''''''
cm = 1/2.54
fig, ax = plt.subplots()
#fig.set_size_inches(12*cm, 10*cm)

p = np.arange(len(x[0]))
width = 0.35
rects1 = ax.bar(p - width/2, y[0], width, label='MACs')
rects2 = ax.bar(p + width/2, x[0], width, label='Parameters')

plt.plot(p, z[0], 'o-', label='Accuracy')
for i in range(5):
    plt.annotate(z[0], (p, z[0]))

fc = ToPercentage
seaxis = ax.secondary_yaxis('right', functions=(fc, fc))
seaxis.set_ylabel('Accuracy (%)')
plt.legend()
plt.grid(linestyle='--')
plt.title('Pruning Result for C32 Dataset')
plt.xlabel('Sparsity Rate')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylabel('Quantity')
#plt.ylim((0, 110))
plt.show()
