from PIL import Image
from pylab import *
from skimage import filters
import numpy as np
import matplotlib.pyplot as plt
import cv2
from multiprocessing import Process

t1=0
t2=0

def otsuSegmentacija(img):
    global t1, t2
    maxVarianca = 0

    #ker imamo 2 t meji imamo tudi 2 for zanki, ce bi imeli 2 pragova se pravi bi imeli samo 1 t in sao eno for zanko
    for t1 in range(0, 256):
        for t2 in range(0, 256):
            interval1 = img <= t1
            interval2 = (img > t1) & (img <= t2)
            interval3 = img > t2

            komulativniInterval1 = np.sum(interval1)
            komulativniInterval2 = np.sum(interval2)
            komulativniInterval3 = np.sum(interval3)


            if komulativniInterval1 == 0:
                break
            if komulativniInterval2 == 0:
                continue
            if komulativniInterval3 == 0:
                continue


            w0 = float(komulativniInterval1) / img.size
            w1 = float(komulativniInterval2) / img.size
            w2 = float(komulativniInterval3) / img.size

            u0= float(np.sum(img * interval1)) / komulativniInterval1
            u1 = float(np.sum(img * interval2)) / komulativniInterval2
            u2 = float(np.sum(img * interval3)) / komulativniInterval3

            Ut = w1 * u1 + w2 * u2 + w0 * u0

            #min varianca
            minVarianca = w0 * pow((u0 - Ut), 2) + w1 * pow((u1 - Ut), 2) + w2 * pow((u2 - Ut), 2)

            if minVarianca > maxVarianca:
                maxVarianca = minVarianca
                koncni_T1 = t1
                koncni_T2 = t2


    print("t1=" , koncni_T1 , " t2=" , koncni_T2)

    koncnaSlika = np.zeros((img.shape[0], img.shape[1]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < koncni_T1:
                koncnaSlika[i][j] = 0
            elif (img[i][j] >= koncni_T1) & (img[i][j] < koncni_T2):
                koncnaSlika[i][j] = 127
            else:
                koncnaSlika[i][j] = 255

    return koncnaSlika

def sirjenje(img):
    #v to sliko bomo zapisovali rezultate
    novaSlika = np.zeros((img.shape[0], img.shape[1]))
    # 3x3 matrika
    matrika = np.ones((3, 3))

    '''matrika[0][0] = 0
    matrika[0][2] = 0
    matrika[2][0] = 0
    matrika[2][2] = 0

    [[0 1 0]
     [1 1 1 ]
     [0 1 0 ]]
        ne dela ce uporabim taksno matriko kot je na sistemu za vaje, tudi na wikipediji je matrika samih enk
    '''

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):

            #ce se vrednosti ujemajo napisemo 1 (matrika samih enk se mora ujemati ko polozimo na sliko, da zapisemo 1)
            novaSlika[i][j] = matrika[0][1] * img[i - 1][j] or matrika[1][0] * img[i][j - 1] or matrika[1][1] * img[i][j] or \
                            matrika[1][2] * img[i][j + 1] or matrika[2][1] * img[i + 1][j]

    novaSlika = np.asarray(novaSlika)
    return novaSlika


def ozanje(img):
    #v to sliko bomo zapisovali rezultate
    novaSlika = np.zeros((img.shape[0], img.shape[1]))
    # 3x3 matrika
    matrika = np.ones((3, 3))


    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):

            #ce se vrednosti ujemajo napisemo 1 (matrika samih enk se mora ujemati ko polozimo na sliko, da zapisemo 1)
            novaSlika[i][j] = matrika[0][1] * img[i - 1][j] and matrika[1][0] * img[i][j - 1] and matrika[1][1] * img[i][j] and \
                            matrika[1][2] * img[i][j + 1] and matrika[2][1] * img[i + 1][j]

    novaSlika = np.asarray(novaSlika)
    return novaSlika


def zapiranje(img):
    sImg = sirjenje(img)
    oImg = ozanje(sImg)

    return oImg

def odpiranje(img):
    oImg = ozanje(img)
    sImg = sirjenje(oImg)

    return sImg

def gradient(sImg, oImg):
    rezImg = sImg - oImg

    return rezImg

def topHat(imgOrg, oImg):
    novaSlika = imgOrg - oImg

    return novaSlika

def backHat(imgOrg, zImg):
    novaSlika = zImg - imgOrg

    return novaSlika

#im_gray = cv2.imread('slika.png',0)
img = array(Image.open("slika.png").convert('L'))

#edges = cv2.GaussianBlur(im_gray,(5,5),0)

imageGaussian = filters.gaussian(img, sigma=1)
#ker dobimo vse stevila 0.3432 moramo pomnoziti z 255
imageGaussian = imageGaussian * 255
#print(edges)


'''p1 = Process(target=otsuSegmentacija, args=(imageGaussian,))
p1.start()
p1.join()'''

otsuImg = otsuSegmentacija(img)

rev, otsu = cv2.threshold(img,127,255,cv2.THRESH_OTSU)


sirjenjeImg = sirjenje(otsu)
ozanjeImg = ozanje(otsu)

zapiranjeImg = zapiranje(otsu)
odpiranjeImg = odpiranje(otsu)
gradientImg = gradient(sirjenjeImg, ozanjeImg)
topHatImg = topHat(img, odpiranjeImg)
blackHatImg = topHat(img, zapiranjeImg)

#cv2.imshow("Sirjenje", sirjenjeImg)
#cv2.waitKey(0)



ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Slika original','otsu2prag', 'Otsu3prag', 'BINARY','BINARY_INV','TRUNC', 'TOZERO','TOZERO_INV', "Sirjenje", "Ozanje", "Zapiranje", "Odpiranje", "Gradient", "Top Hat", "Black Hat"]
images = [img, otsu, otsuImg, thresh1, thresh2, thresh3, thresh4, thresh5, sirjenjeImg, ozanjeImg, zapiranjeImg, odpiranjeImg, gradientImg, topHatImg, blackHatImg]
for i in range(15):
    plt.subplot(4,4,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

'''
figure('otsu')

subplot(121)
gray()
imshow(img)
axis('off')
subplot(122)
gray()
imshow(koncnaSlika)
axis('off')

plt.show()



cv2.imshow("xx", im_bin)

cv2.imshow("xss", im_gray)

cv2.waitKey(0)
'''





