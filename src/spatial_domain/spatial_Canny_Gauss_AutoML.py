import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d
from tqdm import tqdm
import cv2.cv2 as cv2
from sklearn.cluster import KMeans
import PIL.Image as Image
np.random.seed(1)
"""avg_change_rate=0.1122 avg_distortion=1.5647   test_accuracy = 60.90"""
def ternary_entropyf(pP1, pM1):
    p0 = 1.0 - pP1 - pM1
    P = np.array([p0, pP1, pM1])
    eps = 2.2204e-16
    P[P <= 0] = eps
    H = np.array(-1 * (np.multiply(P, np.log2(P))))
    H[(P < eps) | (P > 1 - eps)] = 0
    Ht = np.sum(H)
    return Ht

# print(ternary_entropyf(rhoP1,rhoM1))

def calc_lambda(rhoP1, rhoM1, message_length, n):
    lambda_value = 0
    l3 = 1e+3
    m3 = float(message_length + 1)
    iterations = 0
    while m3 > message_length:
        l3 = l3 * 2
        pP1 = np.divide(np.exp(np.multiply(-l3, rhoP1)),
                        1 + np.exp(np.multiply(-l3, rhoP1)) + np.exp(np.multiply(-l3, rhoM1)))
        pM1 = np.divide(np.exp(np.multiply(-l3, rhoM1)),
                        1 + np.exp(np.multiply(-l3, rhoP1)) + np.exp(np.multiply(-l3, rhoM1)))
        m3 = ternary_entropyf(pP1, pM1)
        iterations = iterations + 1
        if iterations > 10:
            lambda_value = l3
            return lambda_value

    l1 = 0
    m1 = float(n)

    alpha = float(message_length) / n
    """limit search to 30 iterations and require
    that relative payload embedded is roughly within 1/1000 of the required relative payload"""
    while (float(m1 - m3) / n > alpha / 1000.0) and (iterations < 30):
        lambda_value = l1 + (l3 - l1) / 2
        pP1 = np.divide(np.exp(np.multiply(-lambda_value, rhoP1)),
                        1 + np.exp(np.multiply(-lambda_value, rhoP1)) + np.exp(np.multiply(-lambda_value, rhoM1)))
        pM1 = np.divide(np.exp(np.multiply(-lambda_value, rhoM1)),
                        1 + np.exp(np.multiply(-lambda_value, rhoP1)) + np.exp(np.multiply(-lambda_value, rhoM1)))
        m2 = ternary_entropyf(pP1, pM1)
        if m2 < message_length:
            l3 = lambda_value
            m3 = m2
        else:
            l1 = lambda_value
            m1 = m2

        iterations = iterations + 1

    return lambda_value

def EmbeddingSimulator(x, rhoP1, rhoM1, m):
    x = np.array(x)
    n = x.size
    lambda_value = calc_lambda(rhoP1, rhoM1, m, n)
    pChangeP1 = np.divide(np.exp(np.multiply(-lambda_value, rhoP1)),
                          1 + np.exp(np.multiply(-lambda_value, rhoP1)) + np.exp(np.multiply(-lambda_value, rhoM1)))
    pChangeM1 = np.divide(np.exp(np.multiply(-lambda_value, rhoM1)),
                          1 + np.exp(np.multiply(-lambda_value, rhoP1)) + np.exp(np.multiply(-lambda_value, rhoM1)))
    randChange = np.random.rand(x.shape[0], x.shape[1])
    y = x
    y[randChange < pChangeP1] = y[randChange < pChangeP1] + 1
    y[(randChange >= pChangeP1) & (randChange < pChangeP1 + pChangeM1)] = \
        y[(randChange >= pChangeP1) & (randChange < pChangeP1 + pChangeM1)] - 1
    return y



def P_map(img, canny_img, gauss_k_size=7, iter=100):
    img = np.array(img, dtype=np.uint8)
    row, col = img.shape[0], img.shape[1]
    gauss_img = cv2.GaussianBlur(src=img, ksize=[gauss_k_size,gauss_k_size], sigmaX = 0)
    canny = canny_img.copy()
    canny[canny != 0] = 1
    img_data = gauss_img.flatten().reshape([row * col, 1])
    km = KMeans(2, max_iter=iter)
    label = km.fit_predict(img_data)
    label = label.reshape([row, col])
    label = (label - np.mean(label)) / np.std(label)
    label[label > 0] = 1
    label[label <= 0] = 0
    label_canny = np.count_nonzero(label * canny)
    non_label_canny = np.count_nonzero((1-label) * canny)
    if len(str(label_canny)) <= len(str(non_label_canny)):
        if label_canny < non_label_canny:
            label = 1 - label
    # print(np.sum(label),np.count_nonzero(label * canny), np.sum(1-label),np.count_nonzero((1-label) * canny))
    # plt.imshow(label,cmap="gray")
    # plt.show()
    return label


eps = 2.2204e-16

sgm=1
wetCost = 10 ** 10
distortions=[]
change_rates=[]

"""Compute Variance"""
"""Warning: Png images created by plt with pixels from (0,1),
but Pgm images pixels can be read from (0,255)"""

payload =0.4
img = cv2.imread("1.png")
"""Get Simple Dry Cost"""

img_data = np.array(img[:, :, 0])
l_img = np.zeros(shape=[img.shape[0], img.shape[1] + 1], dtype=np.int)
l_img[:, :-1] = img_data
l_img = l_img[:, 1:] - img_data

u_img = np.zeros(shape=[img.shape[0]+1, img.shape[1]], dtype=np.int)
u_img[:-1, :] = img_data
u_img = u_img[1:, :] - img_data

l_u_img = np.zeros(shape=[img.shape[0]+1, img.shape[1] + 1], dtype=np.int)
l_u_img[:-1, :-1] = img_data
l_u_img = l_u_img[1:, 1:] - img_data

r_u_img = np.zeros(shape=[img.shape[0]+1, img.shape[1] + 1], dtype=np.int)
r_u_img[:-1, 1:] = img_data
r_u_img = r_u_img[1:, :-1] - img_data


dry_img = l_img*u_img*l_u_img*r_u_img
dry_img[dry_img != 0] = 255

"""Canny Compute"""
cover = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dtype=np.float)
threshold1 = np.median(list(cover[cover <= 127]) + [50])
threshold2 = np.median(list(cover[cover > 127]) + [150])
canny_img = np.array(cv2.Canny(img, threshold1, threshold2), dtype=np.float)


"""Compute probability map by k-means"""
#{"c_k_size": 1, "d_k_size": 9, "c_multiplier": 5.323643680100246, "d_multiplier": 7.416052845349256}

params = {'c_k_size': 1,'d_k_size': 7,'c_multiplier': 3.747144584574637,'d_multiplier': 6.563151201999116}
# params ={'c_k_size': 1, 'd_k_size': 7, 'c_multiplier': 8.247207160154943, 'd_multiplier': 3.9021337848160913}

canny_gauss_label = P_map(canny_img, canny_img,gauss_k_size=params["c_k_size"])*10
dry_label = P_map(dry_img, canny_img,params["d_k_size"])*10
Variance = canny_gauss_label*params["c_multiplier"]+dry_label/params["d_multiplier"]
Variance[Variance < 0.01] = 0.01

"""Compute rho"""
rho =1.0/Variance

"""adjust embedding costs"""
rho[rho > wetCost] = wetCost  # threshold on the costs
rho[np.isnan(rho)] = wetCost  # if all xi{} are zero threshold the cost
rhoP1 = rho
rhoM1 = rho
rhoP1[cover == 255] = wetCost  # do not embed +1 if the pixel has max value
rhoM1[cover == 0] = wetCost  # do not embed -1 if the pixel has min value
y = EmbeddingSimulator(cover, rhoP1, rhoM1, payload * cover.size)
stego = np.array(y,dtype=np.uint8)


distortion = sum(rho[cover != stego]) / cover.size
change_rate = np.count_nonzero(cover!=stego) / cover.size
print("change_rate=%.4f distortion=%.4f" % (change_rate, distortion))
plt.figure(num="canny_gauss no."+str(1),figsize=(6, 7), dpi=100)
plt.subplot(2, 2, 1)
plt.imshow(cover, cmap='gray')
plt.title("cover")

plt.subplot(2, 2, 2)
plt.imshow(stego, cmap='gray')
plt.title("stego")

plt.subplot(2, 2, 3)
plt.imshow(Variance, cmap='gray')
plt.title("rho")

diff = cover - stego
plt.subplot(2, 2, 4)
plt.imshow(diff, cmap='gray')
plt.title("disturb")

plt.show()