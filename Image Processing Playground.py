import math
import numpy as np
import tkinter as tk
import tkinter.filedialog
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib import use
from random import random
from PIL import Image, ImageTk
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from skimage.transform import resize

WAVELET_CONSTANT = 512
NUM_OPERATION = 6
FILTER_SIZE = 3

def savePlot_(axe, level, xmin, xmax):
	plt.clf()
	plt.bar(axe,level,width=1, align='center')
	plt.xlim(xmin,xmax)
	#plt.savefig(name)
	fig = plt.gcf()
	fig.canvas.draw()
	data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	return data
	
def adjustSize_(w,h):
	#key = [[384,216],[384,216],[384,216],[512,512],[384,216]]
	k1 = 464 #key[OPERATION_TYPE][0]
	k2 = 261 #key[OPERATION_TYPE][1]
	if(h*k1//w)>k2:
		return w*k2//h, k2
	else:
		return k1, h*k1//w

def clear_(keep):
	list1 = window.grid_slaves()
	for i in range(len(list1)-keep-1):
		list1[i].destroy()

def showLabel_(c,r,s,colspan=1,rowspan=1):
	text1 = tk.Label(window,text = s)
	text1.grid(column = c, row = r, columnspan = colspan,rowspan = rowspan)

def showImage_(c,r, arr="0", s="", fromfile=False, colspan = 1, rowspan = 1, resize=True):
	if fromfile==True:
		img1 = Image.open(s)
	else:
		img1 = Image.fromarray(arr)

	if resize:
		img1 = img1.resize( adjustSize_(img1.width,img1.height) )
	img1Tk =  ImageTk.PhotoImage(img1)
	lbl_1 = tk.Label(window, image=img1Tk)
	lbl_1.image = img1Tk
	lbl_1.grid(column = c, row = r, columnspan = colspan, rowspan = rowspan)

def histogram_(img,height=0, width=0):
	if height==0 and width==0:
		height = len(img)
		width = len(img[1])
	level = [0]*512
	for i in range(height):
		for j in range(width):
			if img[i][j]>255:
				img[i][j] = 255
			elif img[i][j]< -256:
				img[i][j] = -256
			level[int(img[i][j]+256)] += 1
	return level

def generateNoise_(height,width,sigma):
	noise = np.zeros((height, width))
	for i in range(height):
		for j in range(width-1):
			if j%2==1:
				continue;
			r = random()
			phi = random()
			z1 = sigma * math.cos(2*math.pi*phi) * math.sqrt((-2)*math.log(r))
			z2 = sigma * math.sin(2*math.pi*phi) * math.sqrt((-2)*math.log(r))
			noise[i][j] += round(z1)
			noise[i][j+1] += round(z2)
	return noise

def wavelet_(entry1,g_im):
	sigma =1;
	str1 = entry1.get()
	if(str1==''):
		return
	else:
		sigma = int(str1)

	clear_(6)

	height = 512
	width = 512
	bias = 512
	copy_g_im = np.zeros((512,512))
	for i in range(512):
		for j in range(512):
			copy_g_im[i][j] = g_im[i][j]
	result = np.zeros((512,512))
	for depth in range(sigma):
		bias = bias//2
		for i in range(bias):
			for j in range(bias):
				A_ = copy_g_im[2*i][2*j]
				B_ = copy_g_im[2*i][2*j+1]
				C_ = copy_g_im[2*i+1][2*j]
				D_ = copy_g_im[2*i+1][2*j+1]

				tmp = 0 + A_+B_+C_+D_
				result[i][j] = tmp//4
				tmp = 0 + A_+B_-C_-D_
				result[i+bias][j] = tmp//4
				tmp = 0 + A_-B_+C_-D_
				result[i][j+bias] = tmp//4
				tmp = 0 + A_-B_-C_+D_
				result[i+bias][j+bias] = tmp//4
		for i in range(bias):
			for j in range(bias):
				copy_g_im[i][j] = result[i][j]
	
	showLabel_(1, 3, "Image after wavelet transform", colspan=4)
	for i in range(height):
		for j in range(width):
			if i<bias and j<bias:
				continue
			if result[i][j] < 0:
				result[i][j] = 0
			elif result[i][j]>5:
				result[i][j] += 127
				if result[i][j]>255:
					result[i][j] = 255

	showImage_(1, 2, result.astype(np.uint8), colspan = 4, resize=False)
	return

def update_(entry1,entry2,g_im):
	sigma = 10
	str1 = entry1.get()
	str2 = entry2.get()
	if str1 == '' and str2 == '':
		return
	elif str1 != '':
		sigma = float(str1)
	else:
		sigma = math.sqrt(float(str2))

	if sigma < 0:
		return

	clear_(9)
	height = g_im.shape[0]
	width = g_im.shape[1]
	noise = generateNoise_(height, width, sigma)

	showLabel_(1, 5, "Histogram of noise:",colspan = 2)
	level = histogram_(noise)
	histpic = savePlot_(axe, level, -258, 258)
	showImage_(1, 6, histpic, colspan = 2)

	showLabel_(0, 5, "Noised image:")
	for i in range(height):
		for j in range(width):
			value = noise[i][j]+g_im[i][j]
			if value>255:
				value = 255
			elif value < 0:
				value = 0
			noise[i][j] = value
	showImage_(0, 6, noise.astype(np.uint8))
	return

def Convolution(convolute_filter, g_im):
	height = g_im.shape[0]
	width = g_im.shape[1]
	result = np.zeros((height, width))
	for i in range(height):
		for j in range(width):
			for m in range(FILTER_SIZE):
				for n in range(FILTER_SIZE):
					if i+m-1 < 0 or i+m-1 >= height:
						x = i
					else:
						x = i+m-1
					if j+n-1 <0 or j+n-1 >= width:
						y = j
					else:
						y = j+n-1
					result[i][j] += g_im[x][y]*convolute_filter[FILTER_SIZE-m-1][FILTER_SIZE-n-1]

			result[i][j] = round(result[i][j])
			if result[i][j]>255 :
				result[i][j]=255
			elif result[i][j]<0 :
				result[i][j] = 0
	return result

def ConvoluteResult(entries, g_im):
	#ClearScreen(13)
	convolute_filter = [ [0]*FILTER_SIZE for i in range(FILTER_SIZE)]
	total = 0
	for i in range(FILTER_SIZE):
		for j in range(FILTER_SIZE):
			str1 = entries[i][j].get()
			if str1=='':
				convolute_filter[i][j]=0
			else:
				value = float(str1)
				convolute_filter[i][j]=value
				total += value
	if total != 0:
		for i in range(FILTER_SIZE):
			for j in range(FILTER_SIZE):
				convolute_filter[i][j] /= total
	
	showLabel_(0, 6, "Image after convolution transform:")
	result = Convolution(convolute_filter, g_im)
	result = result.astype(np.uint8)
	showImage_(0, 7, result)
	#io.imsave(oname+"_filtered.png", result)
	return

def changeOperation(x):
	global OPERATION_TYPE
	OPERATION_TYPE = x
	clear_(1)
	return

def operationIO(im):
	oname1 = oname + "_out.png"
	io.imsave(oname1, im)
	showLabel_(0, 1, "Input image:")
	showImage_(0, 2, fromfile=True, s=oname1)
	showLabel_(1, 1, "Output image can be found at "+oname1+".")
	showImage_(1, 2, fromfile=True, s=oname1)
	return

def operationHistogram(im):
	showLabel_(0, 1, "Input image:")
	showImage_(0, 2, im)
	showLabel_(0, 3, "Grayscale image:")
	g_im = img_as_ubyte(rgb2gray(im))
	showImage_(0, 4, g_im)

	showLabel_(1, 3, "Image histogram:")
	level = histogram_(g_im)
	histpic = savePlot_(axe[256:], level[256:], -2, 258)
	showImage_(1, 4, histpic)
	return

def operationNoise(im):
	showLabel_(0, 1, "Grayscale image:")
	g_im = img_as_ubyte(rgb2gray(im))
	showImage_(0, 2, g_im, rowspan=3)

	showLabel_(1, 1, "Input the parameter of Gaussian noise:",2)
	showLabel_(1, 2, "standard deviation = sigma:") 
	input1 = tk.Entry(window)
	input1.grid(column = 2, row = 2)
	showLabel_(1, 3, "or Variance X = sigma squared:") 
	input2 = tk.Entry(window)
	input2.grid(column = 2, row = 3)
	tk.Button(window,text="Generate noise",command = lambda: update_(input1,input2,g_im) ).grid(column = 1, row = 4, columnspan=2)
	return

def operationWavelet(im):
	showLabel_(0, 1, "Grayscale image:")
	g_im = img_as_ubyte(resize(rgb2gray(im), (512,512)))
	showImage_(0, 2, g_im, resize=False)

	showLabel_(1, 1, "Input how many times you want to transform (1~9):")
	input1 = tk.Entry(window)
	input1.grid(column = 2, row = 1)
	tk.Button(window,text="Transform",command = lambda: wavelet_(input1,g_im) ).grid(column = 4, row = 1, columnspan=2)
	return

def operationEqualization(im):
	showLabel_(0, 1, "Grayscale image:")
	g_im = img_as_ubyte(rgb2gray(im))
	showImage_(0, 2, g_im)

	showLabel_(0, 3, "Image histogram:")
	level = histogram_(g_im)
	histpic = savePlot_(axe[256:], level[256:], -2, 258)
	showImage_(0, 4, histpic)
	
	height = g_im.shape[0]
	width = g_im.shape[1]
	equalized_g_im = np.zeros((height,width))
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
			equalized_g_im[i][j] = accu_level[g_im[i][j]]

	showLabel_(1, 1, "Image after equalization:")
	showImage_(1, 2, equalized_g_im)

	showLabel_(1, 3, "Image histogram after equalization:")
	level = histogram_(equalized_g_im)
	histpic = savePlot_(axe[256:], level[256:], -2, 258)
	showImage_(1, 4, histpic)
	return

def operationConvolution(im):
	showLabel_(0, 1, "Grayscale image:")
	g_im = img_as_ubyte(rgb2gray(im))
	showImage_(0, 2, g_im, rowspan=4)

	showLabel_(1, 1, "Input the 3 by 3 convolution kernal:", colspan = 3)
	inputs = []
	for i in range(3):
		input_line = []
		for j in range(3):
			textbox = tk.Entry(window)
			textbox.grid(column = j+1, row = i+2)
			input_line.append(textbox)
		inputs.append(input_line)
	tk.Button(window,text="View result",command = lambda: ConvoluteResult(inputs, g_im) ).grid(column = 1, row = 5, columnspan=3)
	return

def render_():
	fname = tk.filedialog.askopenfilename()
	if fname:
		clear_(1)

		fn_split = fname.split('.')
		tr = fn_split[len(fn_split)-1]
		if(tr!="png" and tr!="jpeg" and tr!="ppm" and tr!="bmp" and tr!="jpg"):
			return 0

		global oname
		oname = ""
		for i in range(len(fn_split)-1):
			oname += fn_split[i]
			if i != len(fn_split) - 2:
				oname += "."

		im = io.imread(fname)
		if OPERATION_TYPE==0:
			operationIO(im)
		elif OPERATION_TYPE==1:
			operationHistogram(im)
		elif OPERATION_TYPE==2:
			operationNoise(im)
		elif OPERATION_TYPE==3:
			operationWavelet(im)
		elif OPERATION_TYPE==4:
			operationEqualization(im)
		elif OPERATION_TYPE==5:
			operationConvolution(im)
		
oname = ""
axe = [0]*512
for i in range(512):
	axe[i]=i-256
OPERATION_TYPE = NUM_OPERATION-1
op_name = ["File I/O", "Plot Histogram" , "Add Gaussian Noise", "Wavelet Transform", "Histogram Equalization", "Convolution Operation"]

window = tk.Tk()
window.title('AIP_61047039S')
window.geometry('1200x800')
but_file = tk.Button(window,text="Open File",command = render_ ).grid(column = 0, row = 0)
frame1 = tk.Frame(window,height=60, width=600)
for i in range(NUM_OPERATION):
	tk.Button(frame1, text=op_name[i], command = lambda x=i: changeOperation(x) ).grid(column = i, row = 0)
frame1.grid(column= 0, row = 9)
window.mainloop()