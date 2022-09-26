import os

import PIL.Image as Image
import cv2.cv2 as cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
np.random.seed(1)
"""avg_change_rate=0.1026 avg_distortion=1.3727   test_accuracy = 60.90"""
eps = 2.2204e-16

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


def P_map(img, canny_img, gauss_k_size=(7, 7), iter=100):
    img = np.array(img, dtype=np.uint8)
    row, col = img.shape[0], img.shape[1]
    gauss_img = cv2.GaussianBlur(img, gauss_k_size,3)
    canny = canny_img.copy()
    canny[canny != 0] = 255
    img_data = gauss_img.flatten().reshape([row * col, 1])
    if img_data.min() == img_data.max():
        km = KMeans(1, max_iter=iter)
    else:
        km = KMeans(2, max_iter=iter)
    label = km.fit_predict(img_data)
    label = label.reshape([row, col])
    label = (label - np.mean(label)) / (np.std(label)+eps)
    label[label > 0] = 255
    label[label <= 0] = 0
    label_canny = np.count_nonzero(label * canny)
    non_label_canny = np.count_nonzero((255 - label) * canny)
    if len(str(label_canny)) <= len(str(non_label_canny)):
        if label_canny < non_label_canny:
            label = 255 - label
    # print(np.sum(label),np.count_nonzero(label * canny), np.sum(255-label),np.count_nonzero((255-label) * canny))
    # plt.imshow(label,cmap="gray")
    # plt.show()
    return label



payload = 0.4
sgm = 1
wetCost = 10 ** 10
distortions = []
change_rates = []

"""Compute Variance"""
"""Warning: Png images created by plt with pixels from (0,1),
but Pgm images pixels can be read from (0,255)"""

img = cv2.imread("./1.png")
"""Get Simple Dry Cost"""
# def conv2(x, y, mode='same'):
#     return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

img_data = np.array(img[:, :, 0])
C_row, C_col = img_data.shape
beta = 1
l_img = np.zeros(shape=[C_row, C_col + beta], dtype=np.int)
l_img[:, :-beta] = img_data
l_img = l_img[:, beta:] - img_data

u_img = np.zeros(shape=[C_row + beta, C_col], dtype=np.int)
u_img[:-beta, :] = img_data
u_img = u_img[beta:, :] - img_data

l_u_img = np.zeros(shape=[C_row + beta, C_col + beta], dtype=np.int)
l_u_img[:-beta, :-beta] = img_data
l_u_img = l_u_img[beta:, beta:] - img_data

r_u_img = np.zeros(shape=[C_row + beta, C_col + beta], dtype=np.int)
r_u_img[:-beta, beta:] = img_data
r_u_img = r_u_img[beta:, :-beta] - img_data

dry_img = l_img * u_img * l_u_img * r_u_img
dry_img[dry_img != 0] = 1  # 0 black 1 white
# plt.imshow(dry_img,cmap="gray")
# plt.show()
"""Canny Compute"""
cover = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
threshold1 = np.median(list(cover[cover <= 127]) + [50])
threshold2 = np.median(list(cover[cover > 127]) + [150])
# print(threshold1,threshold2)
canny_img = np.array(cv2.Canny(cover, threshold1, threshold2), dtype=np.float)

cover = cover.astype(np.float)
"""Compute probability map by k-means"""
density_canny_img = canny_img
delta = np.count_nonzero(density_canny_img) / (density_canny_img.size - np.count_nonzero(density_canny_img))
delta_info = int(np.ceil(9 + np.log2(delta)))
if delta_info % 2 == 0:
    delta_info += 1
dry_label = P_map(dry_img, canny_img, gauss_k_size=(delta_info, delta_info))
canny_img_blur = cv2.GaussianBlur(canny_img, (delta_info, delta_info), 3) # threshold
gamma = np.median(np.histogram(canny_img_blur)[-1])
canny_img_blur[canny_img_blur >= gamma] *= 2.0-canny_img_blur[canny_img_blur >= gamma]/255.0
canny_img_blur[canny_img_blur < gamma] = 0
dry_label = np.clip(dry_label,0,1)
canny_gauss_label = np.clip(canny_img_blur,0,1)
# plt.imshow(canny_gauss_label,cmap="gray")
# plt.show()

Variance = canny_gauss_label * 2 + dry_label / 1 # threshold
Variance[Variance < 0.01] = 0.01

"""Compute rho"""
rho = 1. / Variance ** 2

"""adjust embedding costs"""
rho[rho > wetCost] = wetCost  # threshold on the costs
rho[np.isnan(rho)] = wetCost  # if all xi{} are zero threshold the cost
rhoP1 = rho
rhoM1 = rho
# print(rho)

rhoP1[cover == 255] = wetCost  # do not embed +1 if the pixel has max value
rhoM1[cover == 0] = wetCost  # do not embed -1 if the pixel has min value
y = EmbeddingSimulator(cover, rhoP1, rhoM1, payload * cover.size)
y[y > 255] = 253  # Taking care of boundary cases
y[y < 0] = 2
stego = np.array(y, dtype=np.uint8)

distortion = sum(rho[cover != stego]) / cover.size
change_rate = np.count_nonzero(cover != stego) / cover.size
print("change_rate=%.4f distortion=%.4f" % (change_rate, distortion))
plt.figure(num="canny_gauss", figsize=(6, 7), dpi=100)
plt.subplot(2, 2, 1)
plt.imshow(cover, cmap='gray')
plt.title("cover")

plt.subplot(2, 2, 2)
plt.imshow(stego, cmap='gray')
#
plt.title("stego")

plt.subplot(2, 2, 3)
plt.imshow(Variance, cmap='gray')
plt.title("rho")

diff = cover - stego

plt.subplot(2, 2, 4)
plt.imshow(diff, cmap='gray')
plt.title("disturb")
plt.show()

diff = cover-stego
idx = np.where((diff>0)|(diff<0))
r_diff,g_diff,b_diff= diff.copy(),diff.copy(),diff.copy()
r_diff[idx]=5
g_diff[idx]=0
b_diff[idx]=255
diff = np.array([r_diff,g_diff,b_diff],dtype=np.uint8)
diff = np.transpose(diff,axes=[1,2,0])

cover = np.array([cover,cover,cover],dtype=np.uint8)
cover = np.transpose(cover,axes=[1,2,0])
out= cover/255+diff
out = np.clip(out,0,1)
plt.imshow(out)
plt.show()

