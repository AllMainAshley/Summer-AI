import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

i=misc.ascent() # 灰度图像
# plt.grid(False) # 生成网络
# plt.gray() # 灰度图像
# plt.axis('off')
# plt.imshow(i)
# plt.show()



i_trainsformed = np.copy(i)     # 此时图像被存储为numpy array
size_x = i_trainsformed.shape[0]
size_y = i_trainsformed.shape[1]

# filter = [[0,1,0],[1,-4,1],[0,1,0]]
# filter = [[-1,-2,-1],[0,0,0],[1,2,1]]
filter = [[-1,0,1],[-2,0,2],[-1,0,1]]
weight = 1

# 构造一个3×3的filter
for x in range(1,size_x-1):
    for y in range(1,size_y-1):
        convolution = 0.0
        convolution = convolution + (i[x-1,y-1] * filter[0][0])
        convolution = convolution + (i[x,y-1] * filter[0][1])
        convolution = convolution + (i[x+1,y-1] * filter[0][2])
        convolution = convolution + (i[x-1,y] * filter[1][0])
        convolution = convolution + (i[x,y] * filter[1][1])
        convolution = convolution + (i[x+1,y] * filter[1][2])
        convolution = convolution + (i[x-1,y+1] * filter[2][0])
        convolution = convolution + (i[x,y+1] * filter[2][1])
        convolution = convolution + (i[x+1,y+1] * filter[2][2])
        # relu层：
        if(convolution<0):
            convolution=0
        if(convolution>255):
            convolution=255
        i_trainsformed[x,y]=convolution

# plt.grid(False)
# plt.gray()
# plt.imshow(i_trainsformed)
# plt.show()

# 池化
new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x,new_y))
for x in range(0,size_x,2):
    for y in range(0,size_y,2):
        pixels = []
        pixels.append(i_trainsformed[x,y])
        pixels.append(i_trainsformed[x+1,y])
        pixels.append(i_trainsformed[x,y+1])
        pixels.append(i_trainsformed[x+1,y+1])
        pixels.sort(reverse=True)
        newImage[int(x/2),int(y/2)] = pixels[0]

plt.grid(False)
plt.gray()
plt.imshow(newImage)
plt.show()