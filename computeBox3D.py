import cv2
import numpy as np
import os.path
def computeBox3D(object,P):
    # takes an object and a projection matrix (P) and projects the 3D
    # bounding box into the image plane.

    # index for 3D bounding box faces
    face_idx = np.array([[ 1,2,6,5],   # front face
                 [2,3,7,6],   # left face
                 [3,4,8,7],   # back face
                 [4,1,5,8]]) # right face

    # compute rotational matrix around yaw axis
    R = np.array([[np.cos(object.ry), 0, np.sin(object.ry)],
                       [0, 1,               0],
         [-np.sin(object.ry), 0, np.cos(object.ry)]]);

    #% 3D bounding box dimensions
    l = object.l
    w = object.w
    h = object.h

    # 3D bounding box corners
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    # rotate and translate 3D bounding box
    corners_3D = R.dot(np.array([x_corners,y_corners,z_corners]))
    corners_3D[1,:] = corners_3D[1,:] + object.t(1)
    corners_3D[2,:] = corners_3D[2,:] + object.t(2)
    corners_3D[3,:] = corners_3D[3,:] + object.t(3)

    # only draw 3D bounding box for objects in front of the camera
    if any(corners_3D[3,:]<0.1):
        corners_2D = []
    else:
        # project the 3D bounding box into the image plane
        corners_2D = projectToImage(corners_3D, P)

        return (corners_2D,face_idx)

def projectToImage(pts_3D, P):
    pts_2D = P.dot(np.array( [pts_3D, np.ones((1,pts_3D.shape[1]))]))
    # scale projected points
    pts_2D[1,:] = pts_2D[1,:]/pts_2D[3,:]
    pts_2D[2,:] = pts_2D[2,:]/pts_2D[3,:]
    pts_2D[3,:] = []
    return pts_2D