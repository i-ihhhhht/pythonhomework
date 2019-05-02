from PIL import Image
from numpy import *
from pylab import *
from imtools import imresize
from scipy.ndimage import filters
from scipy.ndimage import measurements,morphology
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
# coding=UTF-8
# coding: utf-8
############################################################################
'''
#1.1

path=r'D:/209/new'
for filename in os.listdir(path):
    outfile=os.path.splitext(filename)[0]+".jpg"
    if filename!=outfile:
        try:
            Image.open(filename).save(outfile)
        except IOError:
            print"cannot convert",filename
'''
################################################################################
'''
#1.2

pil_im=Image.open('cat.jpg')
print(pil_im.size)
pil_im.thumbnail((128,128))
print(pil_im.size)
pil_im.show()

'''
################################################################################
'''
#1.3
pil_im=Image.open('cat.jpg')
box=(100,100,400,400)
region=pil_im.crop(box)
region=region.transpose(Image.ROTATE_180)
pil_im.paste(region,box)
pil_im.show()
region.show()
'''
################################################################################
'''
#1.4
pil_im=Image.open('cat.jpg')
out=pil_im.resize((128,128))
out1=pil_im.rotate(45)
out.show()
out1.show()

'''
################################################################################
'''
#2.1
im=array(Image.open('cat.jpg'))
imshow(im)
x=[100,100,400,400]
y=[200,500,200,500]
plot(x,y,'r*')
plot(x[:2],y[:2])
title('plotting')
show()
'''
################################################################################
#2.2
'''
im=array(Image.open('cat.jpg').convert('L'))
figure()
gray()
contour(im,orifin='image')
axis('equal')
axis('off')
figure()
hist(im.flatten(),128)
show()
'''
################################################################################
#2.3
'''
im=array(Image.open('cat.jpg'))
imshow(im)
print 'please click 3 points'
x=ginput(3)
print 'you clicked:',x
show()
'''
################################################################################
#3.1
'''

im=array(Image.open('cat.jpg'))
print im.shape,im.dtype
im=array(Image.open('cat.jpg').convert('L'),'f')
print im.shape,im.dtype
'''
################################################################################

#3.2
'''

x=np.arange(0,505,0.1)
y1=255-x
y2=(100.0/255)*x+100
y3=255.0*(x/255.0)**2
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.show()
im=array(Image.open('cat.jpg').convert('L'))
figure()
gray()
imshow(im)
im1=255-im
figure()
gray()
imshow(im1)
im2=(100.0/255)*im+100
figure()
gray()
imshow(im2)
im3=255.0*(im/255.0)**2
figure()
gray()
imshow(im3)
show()
figure()
hist(im.flatten(),128)
figure()
hist(im1.flatten(),128)
figure()
hist(im2.flatten(),128)
figure()
hist(im3.flatten(),128)
show()
'''
################################################################################
'''

#3.3


im=array(Image.open('cat.jpg').convert('L'))
height,width=im.shape[:2]
figure()
gray()
imshow(im)

size=(int(width*0.5),int(height*0.5))
im1=imresize(im,size)
figure()
gray()
imshow(im1)
print im1.shape,im.shape
show()

'''
################################################################################
#3.4
'''
def histeq(im,nbr_bins=256):
    imhist,bins=histogram(im.flatten(),nbr_bins,normed=True)
    cdf=imhist.cumsum()
    cdf=255*cdf/cdf[-1]
    im2=interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape),cdf
im=array(Image.open('cat.jpg').convert('L'))
figure()
gray()
imshow(im)
im2,cdf=histeq(im)
figure()
gray()
imshow(im2)
figure()
hist(im.flatten(),128)
figure()
hist(im2.flatten(),128)
show()

'''
################################################################################
'''
#3.5

def get_fileNames(rootdir):
    fs=[]
    for root,dirs,files in os.walk(rootdir,topdown=True):
        for name in files:
            _,ending=os.path.splitext(name)
            if ending==".jpg":
                fs.append(os.path.join(root,name))
    return fs


path = r'D:/209/new'
pp=get_fileNames(path)
print pp

def compute_average(imlist):
    averageim=array(Image.open(imlist[0]),'f')
    for imname in imlist[1:]:
        try:
            averageim+=array(Image.open(imname))
        except:
            print imname +'...skipped'
        averageim/=len(imlist)
    return array(averageim,'uint8')

cc=compute_average(pp)
print cc



'''

##################################################################
#not work
'''

def pca(X):
    num_data,dim=X.shape
    mean_X=X.mean(axis=0)
    X=X-mean_X
    if dim>num_data:
        M=dot(X,X.T)
        e,EV=linalg.eigh(M)
        tmp=dot(X.T,EV).T
        V=tmp[::-1]
        S=sqrt(e)[::-1]
        for i in range(V.shape[1]):
            V[:,1]/=S
    else:
        U,S,V=linalg.svd(X)
        V=V[:num_data]
    return V,S,mean_X



def get_fileNames(rootdir):
    fs=[]
    for root,dirs,files in os.walk(rootdir,topdown=True):
        for name in files:
            _,ending=os.path.splitext(name)
            if ending==".jpg":
                fs.append(os.path.join(root,name))
    return fs


path = r'D:/209/new'
imlist=get_fileNames(path)


im=array(Image.open(imlist[0]))
m,n=im.shape[0:2]
imnbr=len(imlist)
immatrix=array([array(Image.open(im).convert('L')).flatten()
                for im in imlist],'f')
V,S,immean=pca(immatrix)
figure()
gray()
subplot(2,4,1)
imshow(immean.reshape(m,n))
for i in range (1):
    subplot(2,4,i+2)
    imshow(V[i].reshape(m,n))
show()



'''
##################################################################

'''
#3.7
immean=[1,2,3]
f=open('font_pca_modes.pkl','wb')
pickle.dump(immean,f)
f.close



f=open('font_pca_modes.pkl','rb')
V=pickle.load(f)
f.close()
print V

'''
##################################################################
'''
#4.1
im=array(Image.open('cat.jpg').convert('L'))
im2=filters.gaussian_filter(im,2)
im3=filters.gaussian_filter(im,5)
figure()
gray()
imshow(im2)
figure()
gray()
imshow(im3)
show()
'''
##################################################################
#4.2
'''
im=array(Image.open('cat.jpg').convert('L'))
imx=zeros(im.shape)
filters.sobel(im,1,imx)
imy=zeros(im.shape)
filters.sobel(im,0,imy)
magnitude=sqrt(imx**2+imy**2)
sigma=5
imx1=zeros(im.shape)
filters.gaussian_filter(im,(sigma,sigma),(0,1),imx1)
imy1=zeros(im.shape)
filters.gaussian_filter(im,(sigma,sigma),(1,0),imy1)
figure()
gray()
imshow(im)
figure()
gray()
imshow(imx)
figure()
gray()
imshow(imy)
figure()
gray()
imshow(imx1)
figure()
gray()
imshow(imy1)
show()


'''
##################################################################
'''
#4.3
im=array(Image.open('cat.jpg').convert('L'))
im1=1*(im<128)
labels,nbr_objects=measurements.label(im1)
print "number of objects",nbr_objects
im_open=morphology.binary_opening(im1,ones((9,5)),iterations=2)
labels_open,nbr_objects_open=measurements.label(im_open)
print "number of objects",nbr_objects_open
figure()
gray()
imshow(im1)
figure()
gray()
imshow(labels)
figure()
gray()
imshow(im_open)
figure()
gray()
imshow(labels_open)
show()


'''
##################################################################

#5
def denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
    m,n=im.shape
    U=U_init
    Px=im
    Py=im
    error=1
    while(error>tolerance):
        Uold=U
        GradUx=roll(U,-1,axis=1)-U
        GradUy=roll(U,-1,axis=0)-U
        PxNew=Px+(tau/tv_weight)*GradUx
        PyNew=Py+(tau/tv_weight)*GradUy
        NormNew=maximum(1,sqrt(PxNew**2+PyNew**2))
        Px=PxNew/NormNew
        Py=PyNew/NormNew
        RxPx=roll(Px,1,axis=1)
        RyPy=roll(Py,1,axis=0)
        DivP=(Px-RxPx)+(Py-RyPy)
        U=im+tv_weight*DivP
        error=linalg.norm(U-Uold)/sqrt(n*m)
    return U,im-U
im=zeros((500,500))
im[100:400,100:400]=128
im[200:300,200:300]=255
im=im+30*standard_normal((500,500))
U,T=denoise(im,im)
G=filters.gaussian_filter(im,10)
from scipy.misc import imsave
imsave('original_rof.pdf',im)
imsave('synth_rof.pdf',U)
imsave('synth_gaussian.pdf',G)

im1=array(Image.open('cat.jpg').convert('L'))
U1,T1=denoise(im1,im1)
G1=filters.gaussian_filter(im1,10)
figure()
gray()
axis('equal')
axis('off')
imshow(im1)
figure()
gray()
axis('equal')
axis('off')
imshow(U1)
figure()
gray()
imshow(G1)
axis('equal')
axis('off')
show()


