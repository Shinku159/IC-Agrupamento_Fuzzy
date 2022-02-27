import numpy as np
from sklearn import datasets as Datas

#TRÊS CIRCULOS BEM PRÓXIMOS----------------------------------------------
def Circles01(n_samples=60):
    n_samples = 60
    
    linspace_out = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    outer1_x = np.cos(linspace_out)
    outer1_y = np.sin(linspace_out)

    outer2_x = np.cos(linspace_out)
    outer2_x += 2.1
    outer2_y = np.sin(linspace_out) 

    outer3_x = np.cos(linspace_out)
    outer3_x += 1.05
    outer3_y = np.sin(linspace_out)
    outer3_y += 1.85

    data = np.vstack([np.append(np.append(outer1_x, outer2_x), outer3_x),
                    np.append(np.append(outer1_y, outer2_y), outer3_y)]).T

    return data
#-----------------------------------------------------------------------=

#DOIS CIRCULOS INTERLAÇADOS E UM AFASTADO--------------------------------
def Circles02(n_samples=60):
    n_samples = 60

    data,_ = Datas.make_circles(n_samples=n_samples, factor=0.999)
    linspace_out = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    outer1_x = np.cos(linspace_out)
    outer1_y = np.sin(linspace_out)

    outer2_x = np.cos(linspace_out)
    outer2_y = np.sin(linspace_out)
    outer2_y += 1.9

    outer3_x = np.cos(linspace_out)
    outer3_x += 4
    outer3_y = np.sin(linspace_out)
    outer3_y += 1

    data = np.vstack([np.append(np.append(outer1_x, outer2_x), outer3_x),
                    np.append(np.append(outer1_y, outer2_y), outer3_y)]).T

    return data
#------------------------------------------------------------------------

#DOIS CIRCULOS INTERLAÇADOS----------------------------------------------
def Circles03(n_samples=60):
    n_samples = 60

    data,_ = Datas.make_circles(n_samples=n_samples, factor=0.999)
    linspace_out = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    outer1_x = np.cos(linspace_out)
    outer1_y = np.sin(linspace_out)

    outer2_x = np.cos(linspace_out)
    outer2_x += 0.5
    outer2_y = np.sin(linspace_out)

    data = np.vstack([np.append(outer1_x, outer2_x),
                    np.append(outer1_y, outer2_y)]).T
    
    return data
#------------------------------------------------------------------------

#2 PARES DE CIRCULOS INTERLAÇADOS----------------------------------------
def Circles04(n_samples=60):
    n_samples = 60
    #data,_ = Datas.make_circles(n_samples=n_samples, factor=0.999)
    linspace_out = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    outer1_x = np.cos(linspace_out)
    outer1_y = np.sin(linspace_out)

    outer2_x = np.cos(linspace_out)
    outer2_x += 1.9
    outer2_y = np.sin(linspace_out)

    outer3_x = np.cos(linspace_out)
    outer3_y = np.sin(linspace_out)
    outer3_y += 3

    outer4_x = np.cos(linspace_out)
    outer4_x += 1.9
    outer4_y = np.sin(linspace_out)
    outer4_y += 3

    data = np.vstack([np.append(np.append(np.append(outer1_x, outer2_x), outer3_x), outer4_x),
                    np.append(np.append(np.append(outer1_y, outer2_y), outer3_y), outer4_y)]).T

    return data
#------------------------------------------------------------------------

#NOISE POINTS

def Noise():
    data,_ = Datas.make_blobs(n_samples=[100, 300, 20, 40, 150, 100],
                                    n_features=6,
                                    centers=[[1,3],[5,5],[-5,0],[-5,7],[1,-8],[-6,-6]],
                                    random_state=30,
                                    cluster_std=0.8);

    data1,_ = Datas.make_blobs(n_samples=[30, 30, 30, 30, 30, 30],
                                    n_features=6,
                                    centers=[[1,3],[5,5],[-5,0],[-5,7],[1,-8],[-6,-6]],
                                    random_state=30,
                                    cluster_std=4);

    data2 = np.concatenate((data, data1));

    return data2;

def Luas():
    data,_ = Datas.make_moons(n_samples=1000,
                              noise=0.1,
                              random_state=30);

    return data;

def Luas():
    data,_ = Datas.make_moons(n_samples=1000,
                              noise=0.1,
                              random_state=30);

    return data;

def Circles():
    data,_ = Datas.make_circles(n_samples=1000, noise=.08, random_state=30, factor=.5)

    return data;