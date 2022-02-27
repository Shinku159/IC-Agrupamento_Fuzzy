
#Clustering Figures.

#IMPORTS----------------------------
import numpy as np
import cv2 as cv

from sklearn.cluster import MeanShift
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
#-----------------------------------
def main():
  print("Hi I'm your assistent for image cluster")
  print("Make sure that the figure that you want to cluster has the name 'FigC.jpg'.")
  print('...\n')

  image = cv.imread("FigC.jpg") #Reading the image
  print("The type of this input is {}".format(type(image)))
  print("Shape: {}".format(image.shape))
  # image = cv.cvtColor(image, cv.COLOR_BGR2RGB) #converting the normal image color type to classic RGB


  
  print("Agrupando Imagem... \n.. \n.")
  number_of_colors = 2 #defina aqui o numero de cores final da imagem
  image1 = image.reshape(image.shape[0]*image.shape[1], 3) # reforma a imagem para o formato ideal para o agrupamento
  # clf = KMeans(n_clusters = number_of_colors)
  # labels = clf.fit_predict(image1)
  MScluster = MeanShift(bandwidth=1.2)
  MScluster.fit(image1)
  print("DONE!!")

  center =  np.uint8(MScluster.cluster_centers_)
  res = center[MScluster.labels_.flatten()]
  res2 = res.reshape((image.shape))

  cv.imshow('res2',res2)
  cv.waitKey(0)
  cv.destroyAllWindows()

  # grayImage = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  # plt.imshow(image)
  # plt.show()
  # plt.clf()
  # plt.imshow(grayImage, cmap='Greys')
  # plt.show()


if(__name__ == "__main__"):
  main()


'''
import numpy as np
import cv2 as cv
img = cv.imread('FigC.jpg')
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
print(label)
print(center)
center = np.uint8(center)
print(center)
res = center[label.flatten()]
print(res)
res2 = res.reshape((img.shape))
cv.imshow('res2',res2)
cv.waitKey(0)
cv.destroyAllWindows()
'''