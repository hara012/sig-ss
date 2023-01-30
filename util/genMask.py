#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import math
#from scipy import misc
import imageio
#import random
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

def getRandView():
    fov = 90 * np.random.rand() + 30
    lon = 2 * math.pi * np.random.rand()
    while True:
        lat = math.pi * np.random.rand() - math.pi / 2
        if np.cos(lat) > np.random.rand():
            break
            
    return lon, lat, fov

def main(args):
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    with open("{}/index.csv".format(args.out_dir), mode='w') as f:
        for nv in range(args.num_view):
            for i in range(args.num_img):
                id = i + nv*args.num_img
                print(id)
                comps = []
                for j in range(nv+1):
                    lon, lat, fov =getRandView()
                    if args.rearrange:
                        lon = 0.0
                    comps.append({'lon':lon, 'lat':lat, 'fov':fov})
                mask = make_mask_multi(args.img_h, args.img_w, comps)
                filename = "{}/{:0=5}.png".format(args.out_dir, id)
                imageio.imsave(filename, 255*(1 - mask).astype(np.uint8))
                #f.write("mask/{},{},{},{}\n".format(filename, lon, np.pi/2 - lat, fov))
                f.write("{}".format(filename))
                for j in range(args.num_view):
                    if j <= nv:
                        f.write(",{},{},{}".format(comps[j]['lon'], np.pi/2 - comps[j]['lat'], comps[j]['fov']))
                    else:
                        f.write(",0,0,0")
                f.write("\n")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='mask.')
    parser.add_argument('--img_h', '-h2', type=int, default=256, help='height of image')
    parser.add_argument('--img_w', '-w', type=int, default=512, help='width of image')
    parser.add_argument('--num_img', '-n', type=int, default=1, help='number of images')
    parser.add_argument('--num_view', '-v', type=int, default=1, help='number of groupes')
    parser.add_argument('--rearrange', '-r', action='store_true', help='rearrange mask')
    parser.add_argument('--out_dir', '-o', type=str, default='out', help='output directory')

    args = parser.parse_args()
    
    np.random.seed(1)

    main(args)
    