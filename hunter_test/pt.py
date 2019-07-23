import numpy as np
import pdb
import matplotlib
import cv2
np.set_printoptions(threshold=np.inf)
#pdb.set_trace()
h, w = 128, 64
r = 4
key_list = [(53,47),(63,39),(55,31),(57,17),(61,8),(66,52),(78,49),(77,37),(67,20),(45,18),(22,18),(83,24),(100,24),(116,24),(45,46),(46,52),(49,41),(52,54)]

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

np.save('kungfu1.npy', target_pose)
pose_test = target_pose.sum(axis=2)
pose_test = pose_test + 18
pose_test = pose_test * 200
cv2.imwrite('kungfu1.png', pose_test)
print("target pose generated")
