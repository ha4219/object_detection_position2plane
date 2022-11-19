import numpy as np
import cv2


h_matrix = np.load('data/h.npy')

x = np.array([
    [378, 450, 1],
    [378+581, 450, 1],
    [378, 450+270, 1],
    [378+581, 450+270, 1]
])

dst_list = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    
];

src_pts = np.array(x[:, :2]).reshape(-1, 1, 2)
dst_pts = np.array(dst_list).reshape(-1, 1, 2)

h, mask = cv2.findHomography(src_pts, dst_pts)

dst = cv2.perspectiveTransform(np.array([
    [[0, 0]],
    [[0, 270 - 1]],
    [[581 - 1, 270 - 1]],
    [[581 - 1, 0]],
], dtype=np.float32), h)
print(x.shape, h_matrix.shape)

print(np.matmul(x, h_matrix))

print(np.matmul(x, h)[0])

print(src_pts)
print(dst_pts)
print(dst - dst[0])

np.save('data/h.npy', h)
np.save('data/tl0.npy', dst[0])
