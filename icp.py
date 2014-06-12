import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def icp(a, b, init_pose=(0,0,0), no_iterations = 13):
    '''
    The Iterative Closest Point estimator.
    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint a to the cloudpoint b.
    Note:
        (1) This method works for cloudpoints with minor
        transformations. Thus, the result depents greatly on
        the initial pose estimation.
        (2) A large number of iterations does not necessarily
        ensure convergence. Contrarily, most of the time it
        produces worse results.
    '''

    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)

    #Initialise with the initial pose estimation
    Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
                   [0,                    0,                   1          ]])
    #import pdb; pdb.set_trace()
    src = cv2.transform(src, Tr[0:2])
    #empty = np.zeros((730, 1300, 3))
    for i in range(no_iterations):
        empty = np.zeros((720, 1280, 3))
        #Find the nearest neighbours between the current source and the
        #destination cloudpoint
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst[0])
        distances, indices = nbrs.kneighbors(src[0])
        qualified = (distances<5).ravel()
        indices = indices[qualified, :]
        
        #Compute the transformation between the current source
        #and destination cloudpoint
        T = cv2.estimateRigidTransform(src[:, qualified, :], dst[0, indices.T], True)
        #T = cv2.getPerspectiveTransform(src, dst[0, indices.T])
        #Transform the previous source and update the
        #current source cloudpoint
        src = cv2.transform(src, T)
        
        #empty[a[0,:], a[1,:], :] = (255, 0, 0)
        empty[b[0,:], b[1,:], :] = (255, 0, 0)
        idx = src[:, qualified, :].astype(int)
        idx2 = dst[0, indices.T].astype(int)
        
        #import pdb; pdb.set_trace()
        for (y, x) in idx[0, :]:
            cv2.circle(empty, (x, y), 8, (0,255,0), 1)
        for (y, x) in idx2[0, :]:
            cv2.circle(empty, (x, y), 3, (0,0,255), 1)

        cv2.imshow('icp', empty)
        k = cv2.waitKey(50) & 0xFF
        if k == 27: break
        #Save the transformation from the actual source cloudpoint
        #to the destination
        Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
        print T
        #import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()
    return Tr, (0, Tr[0,2], Tr[1, 2])

if __name__ == '__main__':
    ang = np.linspace(-np.pi/2, np.pi/2, 320)
    a = np.array([ang, np.sin(ang)])
    th = np.pi/2
    rot = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    b = np.dot(rot, a) + np.array([[0.2], [0.3]])

    #Run the icp
    M2 = icp(a, b, [0.1,  0.33, np.pi/2.2], 30)

    #Plot the result
    src = np.array([a.T]).astype(np.float32)
    res = cv2.transform(src, M2)
    plt.figure()
    plt.plot(b[0],b[1])
    plt.plot(res[0].T[0], res[0].T[1], 'r.')
    plt.plot(a[0], a[1])
    plt.show()