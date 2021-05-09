from gimpfu import *
import sys

baseLoc = "/home/paul/Projects/gimp_deep-learning/my_plugins"

sys.path.extend([baseLoc + '/gimpenv/lib/python2.7', baseLoc + '/gimpenv/lib/python2.7/site-packages',
                 baseLoc + '/gimpenv/lib/python2.7/site-packages/setuptools', baseLoc + '/PoolNet/networks'])

import torch
from torch.autograd import Variable
import numpy as np
from joint_poolnet import build_model

def gimp_to_npy(image):
    _, x1, y1, x2, y2 = pdb.gimp_selection_bounds(image)
    width = x2 - x1
    height = y2 - y1
    layer = image.active_layer
    region = layer.get_pixel_rgn(x1, y1, width, height, False)
    pixChars = region[:, :]  # Take whole layer
    img_array = np.frombuffer(pixChars, dtype=np.uint8).reshape(region.h, region.w, region.bpp)
    img_array = img_array[:, :, :3]
    return img_array

def poolnet_inference(image, use_cpu):
    image = np.array(image, dtype=np.float32)
    image -= np.array((104.00699, 116.66877, 122.67892))
    image = image.transpose((2,0,1))
    image = torch.Tensor(image[np.newaxis, ...])
    with torch.no_grad():
        image = Variable(image)
    if not use_cpu:
        image = image.cuda()
    
    net = build_model("resnet")
    if use_cpu:
        net.load_state_dict(torch.load(baseLoc + "/PoolNet/trained.pth", map_location=torch.device('cpu')))
    else:
        net.load_state_dict(torch.load(baseLoc + "/PoolNet/trained.pth"))
    net.eval()

    pred = net(image, mode=1)
    pred = np.squeeze(torch.sigmoid(pred).cpu().data.numpy())

    return pred

def update_selection(image, result):
    _, x1, y1, x2, y2 = pdb.gimp_selection_bounds(image)
    mask = np.zeros((image.height, image.width, 3))
    mask[y1:y2, x1:x2, 0] = (result >= 0.95) * 255
    mask[y1:y2, x1:x2, 1] = (result >= 0.95) * 255
    mask[y1:y2, x1:x2, 2] = (result >= 0.95) * 255
    rlBytes = np.uint8(mask).tobytes()
    rl = gimp.Layer(image, 'mask', image.width, image.height, image.active_layer.type, 100,
                    NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes

    pdb.gimp_image_add_layer(image, rl, 0)
    pdb.gimp_image_select_color(image, CHANNEL_OP_REPLACE, rl,gimpcolor.RGB(0,0,0))
    pdb.gimp_image_remove_layer(image,rl)
    pdb.gimp_selection_feather(image, 3.0)
    gimp.displays_flush()

def iselect(imggimp, use_cpu):
    if torch.cuda.is_available() and not use_cpu:
        gimp.progress_init("(Using GPU) Running intelligent-select for " + imggimp.name + "...")
    else:
        gimp.progress_init("(Using CPU) Running intelligent-select for " + imggimp.name + "...")
    
    selected_img = gimp_to_npy(imggimp)
    pred = poolnet_inference(selected_img, use_cpu)
    update_selection(imggimp, pred)

register(
    "intelligent-select",
    "intelligent-select",
    "Running image segmentation.",
    "Paul Gicquel",
    "Your",
    "2021",
    "intelligent-select...",
    "*",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [
     (PF_IMAGE, "image", "Input image", None),
     (PF_BOOL, "fcpu", "Force CPU", True),
     ],
    [],
    iselect, menu="<Image>/Filters")

main()
