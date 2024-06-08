
from skimage.color import rgb2lab as rgb2lab_lib, deltaE_ciede2000
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import BallTree
import cv2
import numpy as np
import random
import time
import functools
from collections import Counter
from Pylette import extract_colors


rgb_map = {}
lab_map = {}

dmetric = DistanceMetric.get_metric('euclidean')

@functools.cache
def rgb2lab(c):
    return rgb2lab_lib([[c]])

for row in open('floss_dmc.txt','r').readlines():
    code,name,hexs,r,g,b, = row.strip().split('\t')
    r,g,b = int(r),int(g),int(b)



    rgb_map[(r,g,b)]  = code,r,g,b,name
    l,a,bl = rgb2lab((r/255.0, g/255.0, b/255.0))[0][0]
    lab_map[l,a,bl] = r,g,b

floss_colours = list(lab_map.keys())
tree = BallTree(floss_colours, metric=dmetric)


@functools.cache
def rgb2closest(c):
    l,a,bl = rgb2lab(c)[0][0]

    _, result = tree.query( np.array([[l,a,bl]]), k=1 )

    out = []

    for c in result[0]:

        l2,a2,bl2  = floss_colours[c]

        r2,g2,b2 = lab_map[(l2,a2,bl2)]
        out.append( (r2,g2,b2) )
    return out


import tkinter as tk
root = tk.Tk()
from tkinter.filedialog import askopenfilename

screen_width = root.winfo_screenwidth()-100
screen_height = root.winfo_screenheight()-100

max_fit_dimensions = lambda w1, h1, w2, h2: ((min(w2, int(w1 * min(w2/w1, h2/h1))), min(h2, int(h1 * min(w2/w1, h2/h1)))) if w1 > w2 or h1 > h2 else (w1, h1))

root.withdraw()

fn_source =  askopenfilename()

imsrc = cv2.imread(fn_source)
floss_colours2 = []
addedcolours = {}
addedFinalColours = {}
im = imsrc.copy()

previewwidth, previewheight = max_fit_dimensions(im.shape[1],im.shape[0],screen_width,screen_height)


roi = cv2.selectROI("impreview", cv2.resize(im,(previewwidth,previewheight)))

df = im.shape[0]/previewheight

roi = [roi[0]*df,roi[1]*df,roi[2]*df,roi[3]*df]

targetwidth=200

floss_colours_temp = floss_colours



while 1:

    cv2.destroyAllWindows()

    floss_colours = floss_colours_temp
    tree = BallTree(floss_colours, metric=dmetric)

    im = imsrc.copy()
    

    if sum(roi) > 0:
        im = im[ int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2]),]

    widthorig,heightorig = im.shape[1], im.shape[0]

    targetheight = int((im.shape[0]/im.shape[1])*targetwidth)



    previewwidth, previewheight = max_fit_dimensions(widthorig,heightorig,screen_width,screen_height)

    im0 = np.zeros_like(im)

    if targetwidth > widthorig:
        targetwidth = widthorig
        targetheight = heightorig

    im0 = cv2.resize(im0,(previewwidth+80,previewheight))

    im0[0:previewheight,0:previewwidth] = cv2.resize(im,(previewwidth,previewheight))

    def click_and_crop(event, x, y, flags, param):
        if event == 1:


            if x > previewwidth:
                b,g,r = im0[y,x]
                print(b,g,r)
                print(addedFinalColours)
                if (b,g,r) in addedFinalColours:
                    l2,a2,bl2 = addedFinalColours[(b,g,r)]
                    print(l2,a2,bl2)
                    print(floss_colours2)
                    while (l2,a2,bl2) in floss_colours2:
                        floss_colours2.remove((l2,a2,bl2))
                    print(floss_colours2)
                    del addedFinalColours[(b,g,r)]

                if (b,g,r) in addedcolours:
                    del addedcolours[(b,g,r)]

            else:

                cl = im0[y-2:y+2,x-2:x+2]

                for b,g,r in cl.reshape(-1, 3):
                    
                    if (b,g,r) in addedcolours:
                        floss_colours2.append( addedcolours.get((b,g,r)) )
                        continue
                    

                    l,a,bl = rgb2lab((r/255.0,g/255.0,b/255.0))[0][0]

                    _, result = tree.query( np.array([[l,a,bl]]) )
                    print(result)
                    l2,a2,bl2  = floss_colours[result[0][0]]

                    print((l2,a2,bl2))
                    
                    floss_colours2.append((l2,a2,bl2))
                    addedcolours[(b,g,r)] = (l2,a2,bl2)

                    r,g,b = lab_map[(l2,a2,bl2)]

                    addedFinalColours[(b,g,r)] = (l2,a2,bl2)
                    print(rgb_map.get((r,g,b)))

    cv2.namedWindow("impreview")
    cv2.setMouseCallback("impreview", click_and_crop)

    matchsize = 25

    if len(addedFinalColours) == 0:
        palette = extract_colors(image=fn_source, palette_size=matchsize, resize=True)

        floss_colours2.clear()
        addedcolours.clear()
        addedFinalColours.clear()
        
        for p in palette:
            
            r,g,b = p.rgb

            if (b,g,r) in addedcolours:
                floss_colours2.append( addedcolours.get((b,g,r)) )
                continue
            
            l,a,bl = rgb2lab((r/255.0,g/255.0,b/255.0))[0][0]

            _, result = tree.query( np.array([[l,a,bl]]) )
            print(result)
            l2,a2,bl2  = floss_colours[result[0][0]]

            print((l2,a2,bl2))
            
            floss_colours2.append((l2,a2,bl2))
            addedcolours[(b,g,r)] = (l2,a2,bl2)

            r,g,b = lab_map[(l2,a2,bl2)]

            addedFinalColours[(b,g,r)] = (l2,a2,bl2)
            print(rgb_map.get((r,g,b)))

    while 1:
        cv2.imshow('impreview',im0)
        im0[:,im.shape[1]:] = 0,0,0

        imax = 0


        for i,x in enumerate(sorted(addedFinalColours.keys(),key=lambda x: x[0]*0.0722+x[1]*0.7152+x[2]*0.2126 )):
            s,e = i*((previewheight-10)//len(addedFinalColours.keys())), (i+1)*((previewheight-10)//len(addedFinalColours.keys()))
            im0[s:e,previewwidth:] = x[0],x[1],x[2]
            imax=i
            mid = int((s+e)/2)
            
            colour = (255,255,255)
            l = (.299 * x[2]) + (.587 * x[1]) + (.114 * x[0])  
            if l > 127:
                colour = (0,0,0)


            im0 = cv2.putText(im0, str(i),(previewwidth+5,mid), cv2.FONT_HERSHEY_SIMPLEX,  
                    0.5, colour, 1, cv2.LINE_AA) 

        k = cv2.waitKey(1)

        if k == ord('q') or k == ord('c'):
            break

        if k == ord('w'):
            matchsize+=1
            k = ord('e')
        if k == ord('s'):
            matchsize-=1
            k = ord('e')
        matchsize = max(0,matchsize)

        if k == ord('e'):

            palette = extract_colors(image=fn_source, palette_size=matchsize, resize=True)

            floss_colours2.clear()
            addedcolours.clear()
            addedFinalColours.clear()
            
            for p in palette:
                
                r,g,b = p.rgb

                if (b,g,r) in addedcolours:
                    floss_colours2.append( addedcolours.get((b,g,r)) )
                    continue
                
                l,a,bl = rgb2lab((r/255.0,g/255.0,b/255.0))[0][0]

                _, result = tree.query( np.array([[l,a,bl]]) )
                print(result)
                l2,a2,bl2  = floss_colours[result[0][0]]

                print((l2,a2,bl2))
                
                floss_colours2.append((l2,a2,bl2))
                addedcolours[(b,g,r)] = (l2,a2,bl2)

                r,g,b = lab_map[(l2,a2,bl2)]

                addedFinalColours[(b,g,r)] = (l2,a2,bl2)
                print(rgb_map.get((r,g,b)))
            
    cv2.destroyAllWindows()

    if len(floss_colours2) > 1:
        floss_colours = list(set(floss_colours2))
        tree = BallTree(floss_colours, metric=dmetric)

    rgb2closest.cache_clear()
    rgb2lab.cache_clear()

    im = (cv2.resize(im, ( targetwidth,targetheight) ,cv2.INTER_AREA   )).astype(np.uint8)
    im0 = im.copy()

    coords = []

    ditherFactor=1.0

    cache = {}

    ditherimg = np.zeros(im.shape).astype(np.double)

    usedcodes = set()

    for x in range(0,im.shape[0]):
        for y in range(0,im.shape[1]):
            coords.append((x,y))


    start = 0
    errtype = ['per-channel','luminance'][0]

    ditherPattern = [
        [0,     0,      0,   2/14,      1/14],
        [0,     2/14,   2/14,   2/14,   0, 0],
        [1/14,  0,      1/14,   0,      1/14],

    ]


    ditherx,dithery = 0,0

    for dx, row in enumerate(ditherPattern):
        for dy, val in enumerate(row):
            if val == 0:
                ditherx,dithery = dx,dy
            else:
                break
        if val != 0:
            break


    for i,(x,y) in enumerate(coords):

        b0,g0,r0 = im[x,y]
        r,g,b = r0,g0,b0

        cr,cg,cb = r,g,b

        r += ditherimg[x,y][0]
        g += ditherimg[x,y][1]
        b += ditherimg[x,y][2]


        cr,cg,cb = round(min(max(r, 0), 255)), round(min(max(g, 0), 255)), round(min(max(b, 0), 255))

        er,eg,eb = 0.0, 0.0, 0.0

        r, g, b = cr, cg, cb

        r2, g2, b2 = rgb2closest((r/255.0, g/255.0, b/255.0))[0]


        if errtype == 'luminance':
            sumerr = ((r0-r2)*0.2126)+((g0-g2)*0.7152)+((b0-b2)*0.0722)
            er += sumerr
            eg += sumerr
            eb += sumerr
        else:
            er += (r0-r2)*ditherFactor
            eg += (g0-g2)*ditherFactor
            eb += (b0-b2)*ditherFactor

        usedcodes.add(rgb_map.get((r2,g2,b2))[-1])

        for dx, row in enumerate(ditherPattern):
            for dy, val in enumerate(row):
                xpos = x+(dx-ditherx)
                ypos = y+(dy-dithery)
                if 0<=xpos<im0.shape[0] and 0<=ypos<im0.shape[1]:
                    ditherimg[xpos,ypos] += [er*val,eg*val,eb*val]


        #exit()

        im0[x,y] = [b2,g2,r2]

        if abs(start-time.time()) > 0.5:
            start = time.time()
            cv2.imshow('im',  (cv2.resize(im0, ( previewwidth, previewheight  ), 0, 0, interpolation = cv2.INTER_NEAREST   )) )
            #cv2.imshow('im',  im0)
            k = cv2.waitKey(1)
            if k == ord('q') or k == ord('c'):
                break
            print(len(usedcodes))
            print(','.join(usedcodes))

    cv2.imshow('im',  (cv2.resize(im0, ( previewwidth, previewheight  ), 0, 0, interpolation = cv2.INTER_NEAREST   )) )
    #cv2.imshow('im',  im0)

    t = time.time()
    cv2.imwrite(f'out{t}.png', im0) 
    k = cv2.waitKey(0)
    if k == ord('y'):
        break

codes = []
for c,n in Counter([tuple(x) for x in im0.reshape(-1, 3)]).items():
    b,g,r = c
    code,r,g,b,name = rgb_map.get((r,g,b),(000,0,0,0,'UNDFINED'))
    codes.append((n,code,r,g,b,name))

symbols = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789abcdefghijklmnopqrstuvwxyz")
symbols = symbols+symbols+symbols+symbols

symbolcodes = {}
boxsize=20

for s,(n,code,r,g,b,name) in zip(symbols,codes):
    symbolcodes[(r,g,b)] = s,n,code,r,g,b,name

runs = {}

for x in range(0,im0.shape[1]):
    for y in range(0,im0.shape[0]):
        b,g,r = im0[y,x]
        runs.setdefault((r,g,b),[]).append((x,y))

chart = cv2.resize(im0,(im0.shape[1]*boxsize, im0.shape[0]*boxsize))
chart[:,:,:]=255

start = 0
for (r,g,b),coords in  runs.items():

    s,n,code,r,g,b,name = symbolcodes.get((r,g,b))

    for i,(x,y) in enumerate(coords):
        chart = cv2.putText(chart, s ,(x*boxsize,y*boxsize), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0), 1, cv2.LINE_AA) 
        if abs(start-time.time()) > 0.5:
            start = time.time()
            cv2.imshow('chart',cv2.resize(chart, ( widthorig,heightorig) ) )
            k = cv2.waitKey(1)

cv2.imshow('chart',chart) 
k = cv2.waitKey(0)