from gimpfu import *
import sys

baseLoc = "/home/paul/Projects/gimp_deep-learning/my_plugins"

sys.path.extend([baseLoc + '/gimpenv/lib/python2.7', baseLoc + '/gimpenv/lib/python2.7/site-packages',
                 baseLoc + '/gimpenv/lib/python2.7/site-packages/setuptools', baseLoc + '/PoolNet/networks'])

import cv2
import torch
from torch.autograd import Variable
import numpy as np
from joint_poolnet import build_model

def gimp_to_npy(image):
    _, x1, y1, x2, y2 = pdb.gimp_selection_bounds(image)
    width = x2 - x1
    height = y2 - y1
    # Enlarging the selected image a bit
    x1 = x1 - min(x1 - 0, width // 20)
    x2 = x2 + min(image.width - x2, width // 20)
    y1 = y1 - min(y1 - 0, height // 20)
    y2 = y2 + min(image.height - y2, height // 20)
    width = x2 - x1
    height = y2 - y1

    layer = image.active_layer
    region = layer.get_pixel_rgn(x1, y1, width, height, False)
    pixChars = region[:, :]
    img_array = np.frombuffer(pixChars, dtype=np.uint8).reshape(region.h, region.w, region.bpp)
    img_array = convert_to_rgb(img_array, image)
    coordinates = {'x1': x1, 'y1': y1, 'x2': x2, 'y2':y2}
    return img_array, coordinates, width, height

def convert_to_rgb(img_array, image):
    if image.active_layer.type == RGBA_IMAGE:
        img_array = img_array[:, :, :3]
    elif image.active_layer.type == GRAY_IMAGE:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)[:,:,:]
    elif image.active_layer.type == GRAYA_IMAGE:
        img_array = cv2.cvtColor(img_array[:, :, :1], cv2.COLOR_GRAY2RGB)[:,:,:]

    return img_array

def poolnet_inference(image, use_gpu):
    image = np.array(image, dtype=np.float32)
    image -= np.array((104.00699, 116.66877, 122.67892))
    image = image.transpose((2,0,1))
    image = torch.Tensor(image[np.newaxis, ...])
    with torch.no_grad():
        image = Variable(image)
    if use_gpu:
        image = image.cuda()
    
    net = build_model("resnet")
    if use_gpu:
        net.load_state_dict(torch.load(baseLoc + "/PoolNet/trained.pth"))
    else:
        net.load_state_dict(torch.load(baseLoc + "/PoolNet/trained.pth", map_location=torch.device('cpu')))
    net.eval()

    pred = net(image, mode=1)
    pred = np.squeeze(torch.sigmoid(pred).cpu().data.numpy())

    return pred

def update_selection(image, result, coordinates):
    mask = np.zeros((image.height, image.width))
    x1, y1, x2, y2 = coordinates['x1'], coordinates['y1'], coordinates['x2'], coordinates['y2']
    result[result > 1.0] = 1.0
    result[result < 0.0] = 0.0
    result[result < 0.2] = 0.0
    mask[y1:y2, x1:x2] = result * 255
    rlBytes = np.uint8(mask).tobytes()
    selection_mask = image.selection
    region = selection_mask.get_pixel_rgn(0, 0, selection_mask.width, selection_mask.height, True) #True means replace in region
    region[:, :] = rlBytes
    pdb.gimp_selection_feather(image, 2.0)

def iselect(imggimp):
    use_gpu = False
    if imggimp.base_type == INDEXED:
        pdb.gimp_message("Doesn't work with INDEXED color space, please go in RGB mode by selecting Image > Mode > RGB")
        return
    if torch.cuda.is_available():
        use_gpu = True
        gimp.progress_init("(Using GPU) Running Intelligent Select for " + imggimp.name + "...")
    else:
        gimp.progress_init("(Using CPU) Running Intelligent Select for " + imggimp.name + "...")
    
    selected_img, coordinates, width_rect, height_rect = gimp_to_npy(imggimp)

    resize_factor = 1.0
    if min(width_rect, height_rect) < 128:
        resize_factor = 128.0 / min(width_rect, height_rect)
    elif width_rect * height_rect > 512 * 512:
        resize_factor = 512.0 / max(width_rect, height_rect)  
    if resize_factor != 1.0:
        selected_img = cv2.resize(selected_img, None, fx=resize_factor, fy=resize_factor)
    
    pred = poolnet_inference(selected_img, use_gpu)

    if resize_factor != 1.0:
        pred = cv2.resize(pred, (width_rect, height_rect))

    update_selection(imggimp, pred, coordinates)

register(
    "intelligent-select",
    "Intelligent Select",
    "Execute image segmentation in the selected area.",
    "Paul Gicquel",
    "Open Source",
    "2021",
    "Intelligent Select",
    "RGB, RGB*, GRAY, GRAY*",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [
     (PF_IMAGE, "image", "Input image", None),
     ],
    [],
    iselect, menu="<Image>/Select")

main()
