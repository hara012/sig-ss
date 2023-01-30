#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import math
import imageio
import argparse

import ImageTransform as it

def make_mask_f(h, w, azimuth, elevation, fov):
    if h > w:
        sz = h
    else:
        sz = w
    imgs = np.ones((1, sz, sz))

    comp1 = it.Composition(azimuth, elevation, fov, 0.0, h, w)
    comps = [comp1]

    mask = it.PerspectiveImageToEqui(imgs, comps, h, w)
    
    return mask

def make_mask_multi(h, w, comps):

    n_comps = len(comps)
    mask = np.zeros([h, w])

    for c in comps:
        mask_c = make_mask_f(h, w, c['lon'], c['lat'], c['fov'])
        mask += mask_c

    mask[mask > 1.0] = 1.0

    return mask

def main(args):
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    path_index = os.path.join(args.out_dir, 'index.csv')
    with open(path_index, mode='w') as f:
        comps = [{'lon':args.lon, 'lat':args.lat, 'fov':args.fov}]
        mask = make_mask_multi(args.img_h, args.img_w, comps)
        filename = os.path.join(args.out_dir, args.out_name)
        imageio.imsave(filename, 255*(1 - mask).astype(np.uint8))
        f.write("{}".format(filename))
        f.write(",{},{},{}".format(comps[0]['lon'], np.pi/2 - comps[0]['lat'], comps[0]['fov']))
        f.write("\n")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='mask.')
    parser.add_argument('--img_h', '-h2', type=int, default=256, help='height of image')
    parser.add_argument('--img_w', '-w', type=int, default=512, help='width of image')
    parser.add_argument('--lon', type=float, default=np.pi, help='longtitude of mask center [rad]')
    parser.add_argument('--lat', type=float, default=0, help='lattitude of mask center [rad]')
    parser.add_argument('--fov', type=float, default=90, help='fov of mask center [deg]')
    parser.add_argument('--rearrange', '-r', action='store_true', help='rearrange mask')
    parser.add_argument('--out_dir', '-o', type=str, default='tmp', help='output directory')
    parser.add_argument('--out_name', '-n', type=str, default='mask.png', help='output directory')

    args = parser.parse_args()
    
    np.random.seed(1)

    main(args)
    