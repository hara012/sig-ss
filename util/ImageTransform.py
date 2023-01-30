#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from numpy import linalg as LA
import math

#========================================            
# 構図クラス
#========================================
class Composition:
    def __init__(self, _th, _ph, _angle, _dolly, _h, _w):
        self.th = _th
        self.ph = _ph
        self.angle = _angle
        self.dolly = _dolly
        self.h = _h
        self.w = _w
        
#========================================            
# 機能: equiから透視投影画像を切り出す
# 引数: 入力画像(ndarray), 構図(Composition)
#========================================
def ExtractPerspectiveImageFromEqui(iImg, comp):

    #注目方向
    x = math.cos(comp.ph)*math.cos(comp.th)
    y = -math.cos(comp.ph)*math.sin(comp.th)
    z = math.sin(comp.ph)
    
    #画像パラメータ
    w = iImg.shape[1]
    h =  iImg.shape[0]
    ci = comp.h / 2.0
    cj = comp.w / 2.0
    h3div2 = h *3 /2
    wdiv2Pi = w / (2*math.pi)
    hdivPi = h / math.pi
    rate = (2 * math.tan(comp.angle / 2 * math.pi / 180)) / comp.w
    
    #画像軸設定
    ze = np.array([x, y, z])
    g = np.array([0, 0, -1])
    ye = g - ze*(g.dot(ze))
    ye = ye / LA.norm(ye)
    xe = np.cross(ye, ze)

    #画像生成
    I = np.arange(comp.h).reshape([comp.h, 1])
    J = np.arange(comp.w).reshape([1, comp.w])
    di = (I - ci)*rate
    dj = (J - cj)*rate
    a2 = di * di + dj * dj
    D = a2 + 1 - a2*comp.dolly*comp.dolly

    pn = (-comp.dolly +np.sqrt(D)) / (a2 + 1)
    px = pn*dj
    py = pn*di
    pn += comp.dolly
    
    X = pn*ze[0]+ px*xe[0] + py*ye[0]
    Y = pn*ze[1]+ px*xe[1] + py*ye[1]
    Z = pn*ze[2]+ px*xe[2] + py*ye[2]

    #等角座標変換
    th = -np.arctan2(Y, X)
    ph = np.arcsin(Z /np.sqrt(X*X + Y*Y + Z*Z))
    tx = wdiv2Pi*th
    ty = h3div2 - hdivPi*ph

    x0 = np.floor(tx).astype(int)
    y0 = np.floor(ty).astype(int)
    wx = tx - x0
    wy = ty - y0
    w11 = wx*wy
    w10 = wx - w11
    w01 = wy - w11
    w00 = 1 - wx - wy + w11
    x0 = x0 % w
    y0 = y0 % h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h

    w00 = w00.reshape([comp.h, comp.w, 1])
    w01 = w01.reshape([comp.h, comp.w, 1])
    w10 = w10.reshape([comp.h, comp.w, 1])
    w11 = w11.reshape([comp.h, comp.w, 1])
    oImg = iImg[y0, x0] * w00 + iImg[y1, x0] * w01 + iImg[y0, x1] * w10 + iImg[y1, x1] * w11
    
    return oImg

# 部分画像の座標取得
def GetPerspectiveImageCoord(comp):

    #注目方向
    x = math.cos(comp.ph)*math.cos(comp.th)
    y = -math.cos(comp.ph)*math.sin(comp.th)
    z = math.sin(comp.ph)
    
    #画像パラメータ
    ci = comp.h / 2.0
    cj = comp.w / 2.0
    rate = (2 * math.tan(comp.angle / 2 * math.pi / 180)) / comp.w
    
    #画像軸設定
    ze = np.array([x, y, z])
    g = np.array([0, 0, -1])
    ye = g - ze*(g.dot(ze))
    ye = ye / LA.norm(ye)
    xe = np.cross(ye, ze)

    #画像生成
    I = np.zeros((comp.h, comp.w))
    J = np.zeros((comp.h, comp.w))
    for i in range(comp.h):
        for j in range(comp.w):
            I[i][j] = i
            J[i][j] = j
            
    di = (I - ci)*rate
    dj = (J - cj)*rate
    a2 = di * di + dj * dj
    D = a2 + 1 - a2*comp.dolly*comp.dolly

                 
    pn = (-comp.dolly +np.sqrt(D)) / (a2 + 1)
    px = pn*dj
    py = pn*di
    pn += comp.dolly
    
    X = pn*ze[0]+ px*xe[0] + py*ye[0]
    Y = pn*ze[1]+ px*xe[1] + py*ye[1]
    Z = pn*ze[2]+ px*xe[2] + py*ye[2]
    
    return np.array([X, Y, Z])


# 部分画像の座標軸取得
def GetPerspectiveImageAxis(comp):

    #注目方向
    x = math.cos(comp.ph)*math.cos(comp.th)
    y = -math.cos(comp.ph)*math.sin(comp.th)
    z = math.sin(comp.ph)
    
    #画像パラメータ
    rate = (2 * math.tan(comp.angle / 2 * math.pi / 180)) / comp.w
    
    #画像軸設定
    ze = np.array([x, y, z])
    g = np.array([0, 0, -1])
    ye = g - ze*(g.dot(ze))
    ye = ye / LA.norm(ye)
    xe = np.cross(ye, ze)

    return np.array([xe/rate, ye/rate, ze])
    
# 部分画像からequiを構築(dolly未対応) (最近傍版)（180度以上の画角は扱わない）
def PerspectiveImageToEqui(imgs, comps, h, w, offset_col=0):

    h_img = len(imgs[0])
    w_img = len(imgs[0][0])
    
    #部分画像の座標生成    
    num_img = len(imgs)
    c_base = []
    cc = np.zeros([3, num_img])
    for n in range(num_img):
        cc[0][n] = math.cos(comps[n].ph)*math.cos(comps[n].th)
        cc[1][n] = -math.cos(comps[n].ph)*math.sin(comps[n].th)
        cc[2][n] = math.sin(comps[n].ph)
        c_base.append(GetPerspectiveImageAxis(comps[n]))
        
   #equiの座標生成
    I = np.zeros((h, w))
    J = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            I[i][j] = i
            J[i][j] = j
    th = J.astype(np.float64)/(w-2)*math.pi*2
    ph = math.pi/2 - I.astype(np.float64)/(h-1)*math.pi
    x = np.cos(ph)*np.cos(th)
    y = -np.cos(ph)*np.sin(th)
    z = np.sin(ph)
    
    #最も近い部分画像を見つける
    e_dir = np.array([x, y, z]).transpose(1, 2, 0)
    ip = e_dir.dot(cc)
    a0 = np.argmax(ip, axis=2)

    #出力画像を作成
    offset = imgs[0].shape[1]/2.0
    if len(imgs[0].shape) >= 3:
        num_ch = imgs[0].shape[2]
        oImg = np.full([h, w, num_ch], offset_col)
    else:
        oImg = np.full([h, w], offset_col)
    for i in range(h):
        for j in range(w):
            idx = a0[i][j]
            p = c_base[idx].dot(e_dir[i][j]) 
            if p[2] < 0.0: #180度以上の画角
                continue
            x = p[0]/p[2]
            y = p[1]/p[2]
            x = (x + offset).astype(np.int32)
            y = (y + offset).astype(np.int32)
            if 0 <= x and x < w_img and 0<=y and y < h_img:
                oImg[i][j] = imgs[idx][y][x]

    return oImg  
    
# 部分画像からequiを構築(dolly未対応) (線形補間版)
def PerspectiveImageToEquiByLI(imgs, comps, h, w):
    
    #部分画像の座標生成    
    num_img = len(imgs)
    c_base = []
    cc = np.zeros([3, num_img])
    for n in range(num_img):
        cc[0][n] = math.cos(comps[n].ph)*math.cos(comps[n].th)
        cc[1][n] = -math.cos(comps[n].ph)*math.sin(comps[n].th)
        cc[2][n] = math.sin(comps[n].ph)
        c_base.append(GetPerspectiveImageAxis(comps[n]))
        
   #equiの座標生成
    I = np.zeros((h, w))
    J = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            I[i][j] = i
            J[i][j] = j
    th = J.astype(np.float64)/(w-2)*math.pi*2
    ph = math.pi/2 - I.astype(np.float64)/(h-1)*math.pi
    x = np.cos(ph)*np.cos(th)
    y = -np.cos(ph)*np.sin(th)
    z = np.sin(ph)
    
    #equiの各画素のxyz位置と、構図の中心方向の内積
    e_dir = np.array([x, y, z]).transpose(1, 2, 0)
    ip = e_dir.dot(cc)

    #画角内判定用テーブル生成
    thld = np.zeros(num_img)
    for k in range(num_img):
        thld[k] = math.cos(comps[k].angle/360.0*math.pi)
    
    #出力画像を作成
    offset = imgs[0].shape[1]/2.0
    if len(imgs[0].shape) >= 3:
        num_ch = imgs[0].shape[2]
        oImg = np.zeros([h, w, num_ch])
    else:
        oImg = np.zeros([h, w])
    
    #equiの画素ごとに処理
    for i in range(h):
        for j in range(w):
            sum_weight = 0.0
            for k in range(num_img):
                if ip[i][j][k] > thld[k] : #画角内に入っているか判定
                    p = c_base[k].dot(e_dir[i][j]) 
                    x = p[0]/p[2]
                    y = p[1]/p[2]
                    x = (x + offset).astype(np.int32)
                    y = (y + offset).astype(np.int32)
                    weight = (ip[i][j][k] - thld[k]) / (1- thld[k] )
                    oImg[i][j] += imgs[k][y][x] * weight
                    sum_weight  += weight
            if sum_weight  > 0.0:
                oImg[i][j] /= sum_weight

    return oImg  
        
#  np配列を画像のようにリサイズする
def NpResize3(A, hb, wb):
    ha = A.shape[0]
    wa = A.shape[1]
    nch = A.shape[2]
    
    B = np.zeros([hb, wb, nch])
    
    for bi in range(hb):
        for bj in range(wb):
            #入力画像上での座標に変換
            ai = (ha - 1) * bi  / float(hb-1)
            aj = (wa - 1) * bj / float(wb-1)
            ai0 = int(ai)
            aj0 = int(aj)
            aip = ai - ai0
            ajp = aj - aj0
            
            #境界処理
            if(ai0  >= ha-1):
                ai0 -= 1
                aip = 1.0
            if(aj0  >= wa-1):
                aj0 -= 1
                ajp = 1.0
            
            #線形補間
            B[bi][bj] = (1 - aip) * (1 - ajp) * A[ai0][aj0] +   aip * (1 - ajp) * A[ai0+1][aj0] +  (1 - aip) * ajp * A[ai0][aj0 + 1] +  aip *  ajp * A[ai0+1][aj0+1] 
            
    return B
    