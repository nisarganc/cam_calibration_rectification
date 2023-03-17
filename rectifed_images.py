import numpy as np
import cv2 as cv
import glob

#camera projection matrix and distortion co-efficients
camera_omni = np.array([[1442.56071669,   0.,            850.22934855], 
                          [0.,            1442.84866378, 716.87844446], 
                          [0.,            0.,            1.          ]], dtype=np.float32)

camera_pinhole = np.array([[517.47696058, 0.,             846.9816226  ], 
                          [0.,            517.48621258,   740.3092986  ], 
                          [0.,            0.,             1.          ]], dtype=np.float32)              

dist_radtan = np.array([-0.02219753, 0.14768812, -0.00052205, -0.00107954], dtype=np.float32)
dist_equi = np.array([0.00980522,  -0.00095657,  -0.00080178, -0.00025478], dtype=np.float32)
xi = np.array([[1.754305]], dtype=np.float32)

#read image
path = r'~/data/rectify/input.png'
img = cv.imread('input.png')
img_dim = img.shape[:2][::-1]  
print(img_dim)

'''
print(img_dim )
scaled_K = camera_pinhole * 1.0
print(scaled_K)
scaled_K[2][2] = 1.0  
'''

#1. Rectify using cv-fisheye function
new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(camera_pinhole, dist_equi,
    img_dim, np.eye(3), fov_scale=1)
print(new_K)
new_K = np.array([[ 200.994316,   0.,       872.93585 ],
                [  0.,        200.995705,   525.7371  ],
                [  0.,         0.,         1.      ]], dtype=np.float32)
mapx3, mapy3 = cv.fisheye.initUndistortRectifyMap(camera_pinhole, dist_equi, np.eye(3), new_K, img_dim, cv.CV_16SC2)
undistorted_img2 = cv.remap(img, mapx3, mapy3, interpolation=cv.INTER_LINEAR)
cv.imwrite('fisheye_pinhole.png', undistorted_img2)


#2. Rectify using cv-omni function
un_frame = cv.omnidir.undistortImage(img, camera_omni, dist_radtan, xi, 1, np.eye(3))
cv.imwrite('fisheye_omniradtan.png', un_frame)



'''
mapx, mapy = cv.fisheye.stereoRectify(camera_pinhole, dist_equi, new_K, np.eye(3), img_dim, cv.CV_16SC2, validPixROI1=0)
undistorted_img = cv.remap(img, mapx, mapy, interpolation=cv.INTER_LINEAR)
cv.imwrite('fisheye_pinhole_stereo.png', undistorted_img)

# 1. Rectify using function1
mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult1.png', dst)

# 2. Rectify using function2
dst = cv.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult2.png', dst)
mapx2, mapy2 = cv.fisheye.initUndistortRectifyMap(camera_matrix, dist_radtan, np.eye(3), camera_matrix, (w,h), cv.CV_16SC2)
undistorted_img = cv.remap(img, mapx2, mapy2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
cv.imwrite('fisheye.png', undistorted_img)

'''