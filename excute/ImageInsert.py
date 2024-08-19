import cv2

def extend_alpha(image, interval, imgsize):
    right = imgsize[0]-interval[0]-image.shape[0]
    bottom = imgsize[1]-interval[1]-image.shape[1]
    return cv2.copyMakeBorder(image, interval[1], bottom, interval[0], right, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])

def mask_operate(Frame, overlay_img):
    overlay_bgr = overlay_img[:, :, 0:3]
    overlay_img = overlay_img[:, :, 3:]
    overlay_out = cv2.bitwise_and(overlay_bgr, overlay_bgr, mask=overlay_img)
    original_out = cv2.bitwise_and(Frame, Frame, mask=cv2.bitwise_not(overlay_img))
    return cv2.add(original_out, overlay_out)