import matplotlib.pyplot as plt
import numpy as np

"""
Here x, y are numpy arrays and N is an integer value.
Values in x, y represent coordinates in Cartesian plane and N is the number 
of such points.

"""
def linfit(x,y,N):
     
    a = (N*(np.sum(np.multiply(x,y))) - np.sum(x)*np.sum(y))/(N*(np.sum(np.power(x,2))) - np.power(np.sum(x),2))
    b = (np.sum(y) -a*np.sum(x))/N
    params = a,b
    return(params)
    
def test_linfit():
    N = 10    
    plt.axis([-10, 10, -10, 10])
    plt.grid(True);
    inputList = plt.ginput(N,timeout=-1,show_clicks=True)
           
    x_coord = []
    y_coord = []
    for coord in inputList:
        x_coord.append(coord[0])
        y_coord.append(coord[1])

    x = np.array(x_coord)
    y = np.array(y_coord)
    
    plt.scatter(x,y,c ='r',marker='+');
    plt.show();
    
    learningParams = linfit(x,y,N)
    
    x = np.linspace(-10,10,num=10);
    y = learningParams[0]*x + learningParams[1] # y = ax + b
    plt.plot(x,y)
    


test_linfit()







