import numpy as np
import pdb
import matplotlib
import cv2
np.set_printoptions(threshold=np.inf)
#pdb.set_trace()
h, w = 256, 256
r = 4
key_list = [(116,125),(147,121),(151,92),(107,61),(57,30),(146,171),(133,239),(89,191),(253,97),(None,None),(None,None),(254,140),(None,None),(None,None),(94,116),(92,135),(103,109),(96,153)]

target_pose = np.zeros((h,w,18), dtype=np.float32)
j = 0
for i in key_list:
	if i == (None,None):
		target_pose[:,:,j] = np.ones((h, w), dtype=np.float32)*-1
	else:
		a, b = i
		y, x = np.ogrid[-a:h-a, -b:w-b]
		mask = x*x + y*y <= r*r
		array = np.ones((h, w), dtype=np.float32)*-1
		array1 = np.ones((h, w), dtype=np.float32)*2
		arrayt = array1*mask + array
		target_pose[:,:,j] = arrayt
	j+=1
#pdb.set_trace()
#target_pose.astype(np.float32)
#print(target_pose.dtype)

np.save('ultraman1.npy', target_pose)
pose_test = target_pose.sum(axis=2)
pose_test = pose_test + 18
pose_test = pose_test * 200
cv2.imwrite('ultraman1.png', pose_test)
print("target pose generated!")
