from PIL import Image
from numpy import *
from pylab import *
from scipy.ndimage import filters
import os
import pickle
import urllib,urlparse
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import filters
def compute_harris_response(im,sigma=3):
    imx=zeros(im.shape)
    filters.gaussian_filter(im,(sigma,sigma),(0,1),imx)
    imy=zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    Wxx=filters.gaussian_filter(imx*imx,sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    Wdet=Wxx*Wyy-Wxy**2
    Wtr=Wxx+Wyy
    return  Wdet/Wtr

def get_harris_points(harrisim,min_dist=10,threshold=0.1):
    corner_threshold=harrisim.max()*threshold
    harrisim_t=(harrisim>corner_threshold)*1
    coords=array(harrisim_t.nonzero()).T
    candidata_values=[harrisim[c[0],c[1]] for c in coords]
    index=argsort(candidata_values)
    allowed_locations=zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist]=1

    filtered_coords=[]
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]]==1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),(coords[i,1]-min_dist):(coords[i,1]+min_dist)]=0
    return filtered_coords


def plot_harris_points(image, filtered_coords):
     figure()
     gray()
     imshow(image)
     plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
     axis('off')
     show()
#################################################################
'''
im=array(Image.open('cat.jpg').convert('L'))
harrisim = compute_harris_response(im)
filtered_coords = get_harris_points(harrisim, 6,0.01)
filtered_coords1 = get_harris_points(harrisim, 6,0.05)
filtered_coords2 = get_harris_points(harrisim, 6)
plot_harris_points(im, filtered_coords)
plot_harris_points(im, filtered_coords1)
figure()
gray()
plot([p[1] for p in filtered_coords2],[p[0] for p in filtered_coords2],'*')
axis('off')
show()
plot_harris_points(im, filtered_coords2)
'''
#################################################################

def get_descriptors(image,filtered_coords,wid=5):
    desc=[]
    for coords in filtered_coords:
        patch=image[coords[0]-wid:coords[0]+wid+1,coords[1]-wid:coords[1]+wid+1].flatten()
        desc.append(patch)

    return desc
def match(desc1,desc2,threshold=0.5):
    n=len(desc1[0])
    d=-ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1=(desc1[i]-mean(desc1[i]))/std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc_value=sum(d1*d2)/(n-1)
            if ncc_value>threshold:
                d[i,j]=ncc_value
    ndx=argsort(-d)
    matchscores=ndx[:,0]
    return matchscores
def match_twosided(desc1,desc2,threshold=0.5):
    matches_12=match(desc1,desc2,threshold)
    matches_21=match(desc2,desc1,threshold)
    ndx_12=where(matches_12>=0)[0]
    for n in ndx_12:
        if matches_21[matches_12[n]]!=n:
            matches_12[n]=-1
    return matches_12
def appendimages(im1,im2):
    rows1=im1.shape[0]
    rows2=im2.shape[0]
    if rows1<rows2:
        im1=concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1>rows2:
        im2=concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
    return concatenate((im1,im2),axis=1)
def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
    im3=appendimages(im1,im2)
    if show_below:
        im3=vstack((im3,im3))
    imshow(im3)
    cols1=im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
    axis('off')


#############################################################
'''
wid=5
im1=array(Image.open('cat.jpg').convert('L'))
im2=array(Image.open('dog.jpg').convert('L'))
harrisim=compute_harris_response(im1,5)
filtered_coords1=get_harris_points(harrisim,wid+1)
d1=get_descriptors(im1,filtered_coords1,wid)

harrisim=compute_harris_response(im2,5)
filtered_coords2=get_harris_points(harrisim,wid+1)
d2=get_descriptors(im2,filtered_coords2,wid)
print "starting matching"
matches=match_twosided(d1,d2)
figure()
gray()
plot_matches(im1,im2,filtered_coords1,filtered_coords2,matches)
show()
'''

#############################################################

def process_image(imagename,resultname,params="--edge-thresh 10 --peak-thresh 5"):
    if imagename[-3:]!='pgm':
        im=Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename='tmp.pgm'
    cmmd=str("sift"+imagename+"--output="+resultname+" "+params)
    os.system(cmmd)
    print 'processed',imagename,'to',resultname
def read_features_from_file(filename):
    f=loadtxt(filename)
    return f[:,:4],f[:,4:]
def write_features_to_file(filename,locs,desc):
    savetxt(filename,hstack((locs,desc)))
def plot_features(im,locs,circle=False):
    def draw_circle(c,r):
        t=arange(0,1.01,.01)*2*pi
        x=r*cos(t)+c[0]
        y=r*sin(t)+c[1]
        plot(x,y,'b',linewideth=2)
        imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2],p[2])
        else:
            plot(locs[:,0],locs[:,1],'ob')
        axis('off')

'''

imname='cat.jpg'
im1=array(Image.open(imname).convert('L'))
process_image(imname,'cat.sift')
l1,d1=read_features_from_file('cat.sift')
figure()
gray()
plot_features(im1,l1,circle=True)
show()
'''
#############################################################
'''

def match1(desc1,desc2):
    desc1=array([d/linalg.norm(d) for d in desc1])
    desc2=array([d/linalg.norm(d) for d in desc2])
    dist_ratio=0.6
    desc1_size=desc1.shape
    matchscores=zeros((desc1_size[0],1),'int')
    desc2t=desc2.T
    for i in range(desc1_size[0]):
        dotprods=dot(desc1[i,:],desc2t)
        dotprods=0.9999*dotprods
        indx=argsort(arccos(dotprods))
        if arccos(dotprods)[indx[0]]<dist_ratio*arccos(dotprods)[indx[1]]:
            matchscores[i]=int(indx[0])
    return matchscores
def match_twosided1(desc1,desc2):
    matches_12=match1(desc1,desc2)
    matches_21 = match1(desc2, desc1)
    ndx_12=matches_12.nonzero()[0]

    for n in ndx_12:
        if matches_21[int(matches_12[n])]!=n:
            matches_12[n]=0
    return matches_12
def plot_matches1(im1,im2,locs1,locs2,matchscores,show_below=True):
    im3=appendimages(im1,im2)
    if show_below:
        im3=vstack((im3,im3))
    imshow(im3)
    cols1=im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
    axis('off')
    

wid=5
im1=array(Image.open('cat.jpg').convert('L'))
im2=array(Image.open('dog.jpg').convert('L'))
harrisim=compute_harris_response(im1,5)
filtered_coords1=get_harris_points(harrisim,wid+1)
d1=get_descriptors(im1,filtered_coords1,wid)
harrisim=compute_harris_response(im2,5)
filtered_coords2=get_harris_points(harrisim,wid+1)
d2=get_descriptors(im2,filtered_coords2,wid)
print "starting matching"
matches=match_twosided(d1,d2)
matches1=match_twosided1(d1,d2)
print(matches1.shape)
print(matches.shape)
'''

url='www.panoramio.com/map\get_panoramas.php?order=popularity&\set=public&from=0&to=20&minx=-77.037564&miny=38.896662&\ maxx=-77.035564&maxy=38.898662&size=medium'
c=urllib.urlopen(url)
j=json.loads(c.read())
imurls=[]
for im in j['photos']:
    imurls.append(im['photo_file_url'])
for url in imurls:
    iamge=urllib.URLopener()
    iamge.retrueve(url,os.path.basename(urlparse.urlparse(url).path))
    print 'downloading:',url
