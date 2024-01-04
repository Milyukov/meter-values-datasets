import cv2
import numpy as np
import random

def resize_image_keypoints(im, keypoints, width, height):
    h, w, c = im.shape
    delta_top = 0
    delta_bottom = 0
    delta_left = 0
    delta_right = 0
    if h > w:
        delta = h - w
        delta_left = delta // 2
        delta_right = delta // 2
        if delta % 2 != 0:
            delta_right += 1
    else:
        delta = w - h
        delta_top = delta // 2
        delta_bottom = delta // 2
        if delta % 2 != 0:
            delta_top += 1
    keypoints_resized = []
    for point in keypoints:
        keypoints_resized.append([point[0], point[1]])
    for point in keypoints_resized:
        point[0] += delta_left
        point[1] += delta_top
    im_resized = cv2.copyMakeBorder(
        im, delta_top, delta_bottom, delta_left, delta_right, cv2.BORDER_CONSTANT, value=0)
    side = im_resized.shape[0]
    scale = 1024.0 / side
    for point_idx, point in enumerate(keypoints_resized):
        keypoints_resized[point_idx] = (point[0] * scale, point[1] * scale)
    im_resized = cv2.resize(im_resized, (width, height))
    keypoints_resized = np.array(keypoints_resized)
    return im_resized, keypoints_resized

def resize_image_bboxes(im, bboxes, width, height):
    h, w, c = im.shape
    delta_top = 0
    delta_bottom = 0
    delta_left = 0
    delta_right = 0
    # make image rectangular
    if h > w:
        delta = h - w
        delta_left = delta // 2
        delta_right = delta // 2
        if delta % 2 != 0:
            delta_right += 1
    else:
        delta = w - h
        delta_top = delta // 2
        delta_bottom = delta // 2
        if delta % 2 != 0:
            delta_top += 1
    bboxes_resized = []
    for box in bboxes:
        bboxes_resized.append(box[:])
    for box in bboxes_resized:
        box[0] += delta_left
        box[1] += delta_top
    im_resized = cv2.copyMakeBorder(
        im, delta_top, delta_bottom, delta_left, delta_right, cv2.BORDER_CONSTANT, value=0)
    side = im_resized.shape[0]
    scale = height / side
    for box_idx, box in enumerate(bboxes_resized):
        bboxes_resized[box_idx] = [box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale]
    im_resized = cv2.resize(im_resized, (width, height))
    bboxes_resized = np.array(bboxes_resized)
    return im_resized, bboxes_resized

def sort_keypoints(keypoints):
    # assumption: left points are on the left
    keypoints_sorted = sorted(keypoints, key=lambda x: x[0])
    if keypoints_sorted[0][1] < keypoints_sorted[1][1]:
        top_left = keypoints_sorted[0]
        bottom_left = keypoints_sorted[1]
    else:
        top_left = keypoints_sorted[1]
        bottom_left = keypoints_sorted[0]
    if keypoints_sorted[2][1] < keypoints_sorted[3][1]:
        top_right = keypoints_sorted[2]
        bottom_right = keypoints_sorted[3]
    else:
        top_right = keypoints_sorted[3]
        bottom_right = keypoints_sorted[2]
    return np.array([top_left, top_right, bottom_right, bottom_left])

def process_keypoints(im, keypoints, width, height):
    # sort keypoints: top-left, top-right, bottom-rightm bottom-left
    #keypoints = sort_keypoints(keypoints) # they are sorted by design
    # resize image and keypoints
    im_resized, keypoints_resized = resize_image_keypoints(im, keypoints, width, height)
    # create bounding box
    bbox = cv2.boundingRect(keypoints_resized.astype(np.int32))
    bbox = np.array(bbox)
    return im_resized, bbox, keypoints_resized 

def process_bboxes(im, bboxes, width, height):
    # resize image and bboxes
    h_orig = 2 * im.shape[0] // 7
    x_extend = h_orig // 2
    y_extend = 2 * h_orig // 3
    im = im[y_extend:-y_extend, x_extend:-x_extend, :]
    for index, bbox in enumerate(bboxes):
        bboxes[index] = [bbox[0] - x_extend, bbox[1] - y_extend, bbox[2], bbox[3]]
    im_resized, bboxes_resized = resize_image_bboxes(im, bboxes, width, height)
    return im_resized, bboxes_resized 

def extract_rectangle_area(im_resized, bbox, keypoints):
    # evaluate homography transform to warp and crop image inside keypoints
    # crop keypoints coordinates
    x_min = np.min(keypoints[:, 0])
    y_min = np.min(keypoints[:, 1])
    keypoints[:, 0] -= x_min
    keypoints[:, 1] -= y_min
    width = int(np.sqrt((keypoints[0][0] - keypoints[1][0]) ** 2 + (keypoints[0][1] - keypoints[1][1]) ** 2))
    height = int(np.sqrt((keypoints[0][0] - keypoints[3][0]) ** 2 + (keypoints[0][1] - keypoints[3][1]) ** 2))
    keypoints_planar = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)

    h, status = cv2.findHomography(keypoints.astype(np.int32), keypoints_planar)
    # for extended area
    h_inv, status = cv2.findHomography(keypoints_planar, keypoints.astype(np.int32))

    # extend area
    keypoints_planar_extended = keypoints_planar + np.array(
        [[-height//2, -2 * height//3], 
        [height//2, -2 * height//3], 
        [height//2, 2 * height//3], 
        [-height//2, 2 * height//3]])
    # use inverse homography to choose new points on the original image
    keypoints_planar_extended = keypoints_planar_extended.reshape(-1,1,2).astype(np.float32)
    keypoints_extended = cv2.perspectiveTransform(keypoints_planar_extended, h_inv)
    keypoints_extended = keypoints_extended.reshape(-1, 2)
    keypoints_extended[:, 0] += x_min
    keypoints_extended[:, 1] += y_min

    # get bbox
    bbox = cv2.boundingRect(keypoints_extended.astype(np.int32))
    width = int(np.sqrt((keypoints_extended[0][0] - keypoints_extended[1][0]) ** 2 + (keypoints_extended[0][1] - keypoints_extended[1][1]) ** 2))
    height = int(np.sqrt((keypoints_extended[1][0] - keypoints_extended[2][0]) ** 2 + (keypoints_extended[1][1] - keypoints_extended[2][1]) ** 2))

    # compute new homography matrix for extended area
    x_min = np.min(keypoints_extended[:, 0])
    y_min = np.min(keypoints_extended[:, 1])
    keypoints_extended[:, 0] -= x_min
    keypoints_extended[:, 1] -= y_min
    keypoints_planar = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)
    h, status = cv2.findHomography(keypoints_extended.astype(np.int32), keypoints_planar)
    keypoints_extended[:, 0] += x_min
    keypoints_extended[:, 1] += y_min

    # warp image area
    # extend image if bbox is out of image's plane
    top = 0 if bbox[1] > 0 else -bbox[1]
    bottom = 0 if bbox[3] < im_resized.shape[0] else bbox[3] - im_resized.shape[0]
    left = 0 if bbox[0] > 0 else -bbox[0]
    right = 0 if bbox[2] < im_resized.shape[1] else bbox[2] - im_resized.shape[1]
    im_resized = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=np.mean(im_resized))

    im_dst = cv2.warpPerspective(im_resized[bbox[1] + top:bbox[1] + top + bbox[3], 
                                            bbox[0] + left:bbox[0] + left + bbox[2]], h, (width, height))
    #im_dst = cv2.cvtColor(im_dst, cv2.COLOR_BGR2GRAY)
    #im_dst_eq = cv2.equalizeHist(im_dst)
    return im_dst

def augment_data_stage2(im, labels, bboxes):
    choice = random.choice([0, 1])
    im_augmented = im.copy()
    labels_augmented = [label for label in labels]
    bboxes_augmented = bboxes.copy()
    # randomly do one of the following augmentations:
    # rotation
    if choice == 0:
        angle = (random.random() - 0.5) * 45
        # using cv2.getRotationMatrix2D() to get the rotation matrix
        height, width = im_augmented.shape[:2]
        center = (width // 2, height // 2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
        # rotate the image using cv2.warpAffine
        im_augmented = cv2.warpAffine(src=im_augmented, M=rotate_matrix, dsize=(width, height))
        for index, box in enumerate(bboxes_augmented):
            lu = box[:2]
            ru = box[:2] + np.array([box[2], 0])
            rb = box[:2] + box[2:]
            lb = box[:2] + np.array([0, box[3]])
            lu_hom = np.concatenate((lu, np.ones((1,))))
            ru_hom = np.concatenate((ru, np.ones((1,))))
            rb_hom = np.concatenate((rb, np.ones((1,))))
            lb_hom = np.concatenate((lb, np.ones((1,))))
            lu_rot = rotate_matrix.dot(lu_hom)
            #lu_rot = lu_rot[:2] / lu_rot[2]
            ru_rot = rotate_matrix.dot(ru_hom)
            #ru_rot = ru_rot[:2] / ru_rot[2]
            rb_rot = rotate_matrix.dot(rb_hom)
            #rb_rot = rb_rot[:2] / rb_rot[2]
            lb_rot = rotate_matrix.dot(lb_hom)
            #lb_rot = lb_rot[:2] / lb_rot[2]
            minx = int(np.min([lu_rot[0], ru_rot[0], rb_rot[0], lb_rot[0]]))
            maxx = int(np.max([lu_rot[0], ru_rot[0], rb_rot[0], lb_rot[0]]))
            miny = int(np.min([lu_rot[1], ru_rot[1], rb_rot[1], lb_rot[1]]))
            maxy = int(np.max([lu_rot[1], ru_rot[1], rb_rot[1], lb_rot[1]]))
            bboxes_augmented[index] = np.array([minx, miny, maxx-minx, maxy-miny], dtype=int)
    # scaling
    elif choice == 1:
        scale = 1.0 + (random.random() - 0.5)
        im_augmented = cv2.resize(im, (int(im.shape[0] * scale), int(im.shape[1] * scale)))
        for index, box in enumerate(bboxes_augmented):
            bboxes_augmented[index] = (box * scale).astype(np.int32)

    return im_augmented, labels_augmented, bboxes_augmented
