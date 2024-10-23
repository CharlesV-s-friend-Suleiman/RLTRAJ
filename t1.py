import matplotlib.pyplot as plt
import matplotlib.image as mpimg

background_img = mpimg.imread('figur/GG_realworld.jpg')
slicepg = background_img[0:100, 0:1000]
plt.imshow(slicepg)
plt.show()

