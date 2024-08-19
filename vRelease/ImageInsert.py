from cv2 import copyMakeBorder, BORDER_CONSTANT, bitwise_and, bitwise_not, add

def extend_alpha(image, interval, imgsize):
    right = imgsize[0]-interval[0]-image.shape[0]
    bottom = imgsize[1]-interval[1]-image.shape[1]
    return copyMakeBorder(image, interval[1], bottom, interval[0], right, BORDER_CONSTANT, value=[0, 0, 0, 0])

def mask_operate(Frame, overlay_img):
    overlay_bgr = overlay_img[:, :, 0:3]
    overlay_img = overlay_img[:, :, 3:]
    overlay_out = bitwise_and(overlay_bgr, overlay_bgr, mask=overlay_img)
    original_out = bitwise_and(Frame, Frame, mask=bitwise_not(overlay_img))
    return add(original_out, overlay_out)