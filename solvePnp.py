import cv2
import numpy as np
img = cv2.imread("assets/images/data.png")
size = img.shape
image_points_2D = np.array([
        (378, 450),
        (378+581, 450),
        (378, 450),
        (378+581, 450+270),
    ], dtype="double")

figure_points_3D = np.array([
    (0.0, 1.0, 0.0),             # left top
    (1.0, 1.0, 0.0),        # right top
    (0.0, 0.0, 0.0),     # left bottom
    (1.0, 0.0, 0.0),      # right bottom
])

distortion_coeffs = np.zeros((4,1))
focal_length = size[1]
center = (size[1]/2, size[0]/2)
matrix_camera = np.array(
    [[focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]], dtype = "double"
)
success, vector_rotation, vector_translation = cv2.solvePnP(figure_points_3D, image_points_2D, matrix_camera, distortion_coeffs, flags=0)
nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), vector_rotation, vector_translation, matrix_camera, distortion_coeffs)
for p in image_points_2D:
    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

for i in range(len(image_points_2D)):
    point1 = ( int(image_points_2D[i][0]), int(image_points_2D[i][1]))

    point2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(img, point1, point2, (255,255,255), 2)


cv2.imwrite('tmp/projection.png', img)
print(vector_rotation, vector_translation)
print(vector_rotation.shape, vector_translation.shape)
np.save('data/vr.npy', vector_rotation)
np.save('data/vt.npy', vector_translation)
# cv2.imshow("Final",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()