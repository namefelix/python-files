import math
import copy
import os
import numpy as np
import tkinter as tk
import skimage.io as io
import matplotlib.pyplot as plt 
from random import random
from PIL import Image, ImageTk
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from skimage.transform import resize
from skimage.feature import canny
from skimage.segmentation import mark_boundaries
from skimage.filters import gaussian
from skimage.feature import corner_harris

def ConstructHistogram(img,height=0, width=0,band=-1):
	if height==0 and width==0:
		height = len(img)
		width = len(img[1])
	level = [0]*512
	for i in range(height):
		for j in range(width):
			if band==-1:
				if img[i][j]>255:
					img[i][j] = 255
				elif img[i][j]< -256:
					img[i][j] = -256
				level[int(img[i][j]+256)] += 1
			else:
				if img[i][j][band]>255:
					img[i][j][band] = 255
				elif img[i][j][band]< -256:
					img[i][j][band] = -256
				level[int(img[i][j][band]+256)] += 1
	return level

def IsUseless(pixel,pixel_eq):
	flag = False
	if 0+pixel_eq[1]-pixel_eq[0] < -20:
		flag = True
	elif pixel_eq[0] > 128:
		flag = True
	elif abs(0+pixel_eq[1]-pixel_eq[0]) < 20 and pixel_eq[0] > 40:
		flag = True
	elif abs(0+pixel_eq[2]-pixel_eq[0]) < 20 and pixel_eq[0] > 40:
		flag = True
	elif abs(0+pixel[2]-pixel[1]) > 30:
		flag = True

	return flag

def Final(im):
	height = im.shape[0]
	width = im.shape[1]
	
	print("    Equalizing...")
	equalizedG_im = copy.deepcopy(im)
	for k in range(3):
		print("    "+str(k+1)+"/"+"3...")
		level = ConstructHistogram(im,band=k)
		accu_level = [0]*256
		for i in range(256):
			accu_level[i]=level[256+i]+accu_level[i-1]
		hmin = 0
		for i in range(256):
			if level[256+i]!=0:
				hmin = i
				break
		hmin = accu_level[hmin]
		siz = height*width-hmin
		for i in range(256):
			accu_level[i] = round( (accu_level[i]-hmin)*255/siz )
		for i in range(height):
			for j in range(width):
				equalizedG_im[i][j][k] = accu_level[im[i][j][k]]

	print("    Thresholding...")
	for i in range(height):
		for j in range(width):
			if IsUseless(im[i][j],equalizedG_im[i][j]):
				equalizedG_im[i][j][0] = 0
				equalizedG_im[i][j][1] = 0
				equalizedG_im[i][j][2] = 0
	g_im = np.zeros((height,width),dtype=np.uint8)
	for i in range(height):
		for j in range(width):
			g_im[i][j] = equalizedG_im[i][j][1]//2 + equalizedG_im[i][j][2]//2 
	g_im = gaussian(g_im)

	print("    Marking...")
	edges = canny(g_im)
	contain_mark = False
	if np.count_nonzero(edges)>500:
		contain_mark = True
	corners = corner_harris(edges)

	out = mark_boundaries(im, edges, (1,0,0))
	uim = img_as_ubyte(out)
	io.imsave("./results/"+oname+".png", uim)
	return contain_mark

def Run():
	end = 15
	arr = []
	for i in range(end):
		print(i+1,"  of  ",end,"...")
		fname = "./sample/test"+str(i+1)+".jpg"

		fn_split = fname.split('.')
		tr = fn_split[len(fn_split)-1]
		if(tr!="png" and tr!="jpeg" and tr!="ppm" and tr!="bmp" and tr!="jpg"):
			return 0

		global oname
		oname = ""
		for j in range(len(fn_split)-1):
			oname += fn_split[j]
			if j != len(fn_split) - 2:
				oname += "."
		im = io.imread(fname)
		arr.append(Final(im))
	print("Contain marks: ", arr)

oname = ""
axe = [0]*512
for i in range(512):
	axe[i]=i-256
FILTER_SIZE = 3
cwd = os.getcwd()
if not os.path.exists(cwd+'/results'):
	os.makedirs(cwd+'/results')
Run()
