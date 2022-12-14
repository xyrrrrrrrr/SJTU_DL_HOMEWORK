import matplotlib.pyplot as plt
# import numpy as np
import cv2

image_root = 'example.png'

image = cv2.imread(image_root)
# cut image into 4 part
image_length = image.shape[0]
image_width = image.shape[1]
image_1 = image[0:image_length//2, 0:image_width//2]
image_2 = image[0:image_length//2, image_width//2:image_width]
image_3 = image[image_length//2:image_length, 0:image_width//2]
image_4 = image[image_length//2:image_length, image_width//2:image_width]

# save image
cv2.imwrite('./example_1.png', image_1)
cv2.imwrite('./example_2.png', image_2)
cv2.imwrite('./example_3.png', image_3)
cv2.imwrite('./example_4.png', image_4)

# plot 5 sub-graph with 4 parts and 1 original image
for j in range(4):
    ax = plt.subplot2grid((2,4),(int(j/2), j%2))
    ax.axis('off')
    ax.imshow(eval('image_'+str(j+1)))
    
ax = plt.subplot2grid((2,4),(0,2), colspan=2, rowspan=2)
ax.axis('off')
ax.imshow(image)

plt.savefig('example?.png')

#  for j in range(4):
#                 ax = plt.subplot2grid((2, 4), (int(pred[i][j]/2), pred[i][j]%2))
#                 img = jt.array(inputs[i][j]*0.5+0.5)#
#                 ax.imshow(img)
#                 ax.axis("off")
#                 # ax.imshow(inputs[i][j].numpy().transpose(1,2,0))
#             ax = plt.subplot2grid((2,4), (0,2), colspan=2, rowspan=2)
#             img = jt.array(val_loader_n[batch_idx*64+i][0]*0.5+0.5)#
#             ax.axis("off")
#             ax.imshow(img.numpy().transpose(1,2,0))