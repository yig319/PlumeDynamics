import cv2
import numpy as np

def transform_image(image, frame_view, frame_view_ref):
    transformation_matrix = cv2.getPerspectiveTransform(frame_view.astype(np.float32), frame_view_ref.astype(np.float32))
    transformed_image = cv2.warpPerspective(image, transformation_matrix, (image.shape[1], image.shape[0]))
    return transformed_image

def align_plumes(plumes, frame_view, frame_view_ref):
    align_plumes = np.zeros(plumes.shape, dtype=plumes.dtype)
    n_plume, n_frame, h, w = plumes.shape
    for n1 in range(n_plume):
        for n2 in range(n_frame):
            align_plumes[n1, n2] = transform_image(plumes[n1, n2], frame_view, frame_view_ref)
    return align_plumes