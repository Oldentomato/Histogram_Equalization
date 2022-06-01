import cv2
import numpy as np
import matplotlib.pyplot as plt

def HistoEqual(m_image):
    k = 0
    Sum = 0
    total_pixels = 0
    hist = list(0 for i in range(256))
    sum_of_hist = list(0 for i in range(256))
    for z in range(0, 256): #초기화 단계
        hist[z] = 0
        sum_of_hist[z] = 0
    
    for i in range(0,256): 
        for j in range(0,256):
            k = int(m_image[i][j])
            hist[k] = hist[k] + 1
    for i in range(0,256):
        Sum = Sum + hist[i]
        sum_of_hist[i] = Sum

    total_pixels = 256 * 256

    for i in range(0,256):
        for j in range(0,256):
            k = int(m_image[i][j])
            m_image[i][j] = sum_of_hist[k]*(255.0/total_pixels)
            #sum_of_hist 누적분포함수 * 255/total_pixels 최대 명암도
    return m_image

def rgb2ycbcr(img):
    height, width, channel = img.shape
    b = img[...,0]
    g = img[...,1]
    r = img[...,2]

    y = np.zeros((height, width), dtype=np.float)
    cr = np.zeros((height,width),dtype=np.float)
    cb = np.zeros((height,width), dtype=np.float)
    for i in range(height):
        for j in range(width):
            y[i][j] = 0.299 * r[i][j] + 0.587 * g[i][j] + 0.114 * b[i][j]
            cr[i][j] = (r[i][j] - y[i][j]) * 0.713 + 128
            cb[i][j] = (b[i][j] - y[i][j]) * 0.564 + 128
    return y.astype(np.uint8),cr.astype(np.uint8),cb.astype(np.uint8)

def ycbcr2rgb(img):
    xform = np.array([[1,0,1.402],[1,-0.34414, -.71414],[1,1.772,0]])
    rgb = img.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(matrixMult(xform))


    np.putmask(rgb,rgb>255,255)
    np.putmask(rgb,rgb<0,0)
    return np.uint8(rgb)



def matrixMult(A): # 전치행렬
    row = len(A)
    col = len(A[0])
    B=[[0 for row in range(row)] for col in range(col)]

    for i in range(row):
        for j in range(col):
            B[j][i]=A[i][j]
    return B



img = cv2.imread("./image.jpeg", cv2.IMREAD_UNCHANGED)
r = img[...,0]
g = img[...,1]
b = img[...,2]

y,cr,cb = rgb2ycbcr(img)

new_y = HistoEqual(y)
ycrcb = (np.dstack((new_y, cr, cb))).astype(np.uint8)

new_rgb = ycbcr2rgb(ycrcb)

nr = new_rgb[...,0]
ng = new_rgb[...,1]
nb = new_rgb[...,2]



f,axes = plt.subplots(2,3)
f.set_size_inches((10,5))
plt.subplots_adjust(hspace=0.5)

axes[0,0].hist(r)
axes[0,0].set_title('Before Red')
axes[0,1].hist(g)
axes[0,1].set_title('Before Green')
axes[0,2].hist(b)
axes[0,2].set_title('Before Blue')

axes[1,0].hist(nr)
axes[1,0].set_title('After Red')
axes[1,1].hist(ng)
axes[1,1].set_title('After Green')
axes[1,2].hist(nb)
axes[1,2].set_title('After Blue')


cv2.imshow("original",img)
cv2.imshow("image",new_rgb)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()