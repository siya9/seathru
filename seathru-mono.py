from __future__ import absolute_import, division, print_function
import cv2
import os
import sys
import glob
import argparse
import time
import io
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import pynng
from pynng import nng

import torch
from torchvision import transforms, datasets

import deps.monodepth2.networks as networks
from deps.monodepth2.utils import download_model_if_doesnt_exist

from seathru import *

def run(args, filename):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    
    # Load image and preprocess
    img = cv2.imread(args.image + "/" + filename)  # Load image using OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    original_height, original_width = img.shape[:2]
    img = cv2.resize(img, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)
    input_image = cv2.resize(img, (feed_width, feed_height), interpolation=cv2.INTER_LANCZOS4)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    print('Preprocessed image', flush=True)

    # PREDICTION
    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
    mapped_im_depths = ((disp_resized_np - np.min(disp_resized_np)) / (
            np.max(disp_resized_np) - np.min(disp_resized_np))).astype(np.float32)
    print("Processed image", flush=True)
    print('Loading image...', flush=True)
    depths = preprocess_monodepth_depth_map(mapped_im_depths, args.monodepth_add_depth,
                                            args.monodepth_multiply_depth)
    print("The depths data type is ", depths.shape)

    # depth map acquired
    recovered = run_pipeline(np.array(img) / 255.0, depths, args)
    sigma_est = estimate_sigma(recovered, channel_axis=-1, average_sigmas=True) / 10.0
    recovered = denoise_tv_chambolle(recovered, sigma_est, channel_axis=-1)
    im = Image.fromarray((np.round(recovered * 255.0)).astype(np.uint8))
    im.save(args.output + "/" + filename, format='png')
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Input image')
    parser.add_argument('--output', default='output.png', help='Output filename')
    parser.add_argument('--f', type=float, default=2.0, help='f value (controls brightness)')
    parser.add_argument('--l', type=float, default=0.5, help='l value (controls balance of attenuation constants)')
    parser.add_argument('--p', type=float, default=0.01, help='p value (controls locality of illuminant map)')
    parser.add_argument('--min-depth', type=float, default=0.0,
                        help='Minimum depth value to use in estimations (range 0-1)')
    parser.add_argument('--max-depth', type=float, default=1.0,
                        help='Replacement depth percentile value for invalid depths (range 0-1)')
    parser.add_argument('--spread-data-fraction', type=float, default=0.05,
                        help='Require data to be this fraction of depth range away from each other in attenuation estimations')
    parser.add_argument('--monodepth-add-depth', type=float, default=2.0, help='Additive value for monodepth map')
    parser.add_argument('--monodepth-multiply-depth', type=float, default=10.0,
                        help='Multiplicative value for monodepth map')
    parser.add_argument('--model-name', type=str, default="mono_1024x320",
                        help='monodepth model name')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        os.makedirs(args.image)

    # Loop through all files in the folder
    for filename in os.listdir(args.image):
        # if filename.endswith('.png'):  
        run(args, filename)
            # print(filename)
            