from scipy.misc import imread
import numpy as np
import os
a = os.listdir("./edges2handbags/train")
#check 0-255 or 0-1
content_data=np.ones([20000, 256, 256, 3], np.float32)
style_data = np.ones([20000, 256, 256, 3], np.float32)
b=(imread("./edges2handbags/train/"+a[0])/255).astype(np.float32)
content_data[0, :, :, :] = b[np.newaxis, :, 0:256, :]
style_data[0, :, :, :] = b[np.newaxis, :, 256:512, :]
for i in range(1, len(a)):
    b=(imread("./edges2handbags/train/"+a[i])/255).astype(np.float32)
    content_data[i%20000, :, :, :] = b[np.newaxis, :, 0:256, :]
    style_data[i%20000, :, : , :] = b[np.newaxis, :, 256:512, :]
    if i % 1000 == 0:
        print(i)
    if i % 20000 == 0:
        np.save("style_data"+str(int(i/10000))+"w", style_data)
        np.save("content_data"+str(int(i/10000))+"w", content_data)
np.save("style_data", style_data)
np.save("content_data", content_data)
