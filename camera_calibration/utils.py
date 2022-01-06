import cv2

def save_coefficients(mtx, dist, rvecs, tvecs, path):
    '''Save the camera matrix and the distortion coefficients to given path/file.'''
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    cv_file.write('R', rvecs)
    cv_file.write('T', tvecs)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_coefs = cv_file.getNode('D').mat()
    rot_vecs = cv_file.getNode('R').mat()
    tran_vecs = cv_file.getNode('T').mat()

    cv_file.release()
    return (camera_matrix, dist_coefs, rot_vecs, tran_vecs)