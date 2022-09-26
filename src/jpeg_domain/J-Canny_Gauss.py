import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2.cv2 as cv2
from sklearn.cluster import KMeans
import PIL.Image as Image
import collections
import jpegio as jio


def ternary_entropyf(pP1, pM1):
    p0 = 1.0 - pP1 - pM1
    P = np.array([p0, pP1, pM1])
    eps = 2.2204e-16
    P[P <= 0] = eps
    H = np.array(-1 * (np.multiply(P, np.log2(P))))
    H[(P < eps) | (P > 1 - eps)] = 0
    Ht = np.sum(H)
    return Ht


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
    gauss_img = cv2.GaussianBlur(img, gauss_k_size, 0)
    canny = canny_img.copy()
    canny[canny != 0] = 1
    img_data = gauss_img.flatten().reshape([row * col, 1])
    if np.max(img_data)==np.min(img_data):
        km = KMeans(1, max_iter=iter)
    else:
        km = KMeans(2, max_iter=iter)
    label = km.fit_predict(img_data)
    label = label.reshape([row, col])
    label = (label - np.mean(label)) / (np.std(label)+2**(-6))
    label[label > 0] = 1
    label[label <= 0] = 0
    label_canny = np.count_nonzero(label * canny)
    non_label_canny = np.count_nonzero((1 - label) * canny)
    if len(str(label_canny)) <= len(str(non_label_canny)):
        if label_canny < non_label_canny:
            label = 1 - label
    # print(np.sum(label),np.count_nonzero(label * canny), np.sum(1-label),np.count_nonzero((1-label) * canny))
    # plt.imshow(label,cmap="gray")
    # plt.show()
    return label


"""Get Simple Dry Cost"""
def drt_cost(img):
    img_data = np.array(img[:, :])
    l_img = np.zeros(shape=[img.shape[0], img.shape[1] + 1], dtype=np.int)
    l_img[:, :-1] = img_data
    l_img = l_img[:, 1:] - img_data

    u_img = np.zeros(shape=[img.shape[0] + 1, img.shape[1]], dtype=np.int)
    u_img[:-1, :] = img_data
    u_img = u_img[1:, :] - img_data

    l_u_img = np.zeros(shape=[img.shape[0] + 1, img.shape[1] + 1], dtype=np.int)
    l_u_img[:-1, :-1] = img_data
    l_u_img = l_u_img[1:, 1:] - img_data

    r_u_img = np.zeros(shape=[img.shape[0] + 1, img.shape[1] + 1], dtype=np.int)
    r_u_img[:-1, 1:] = img_data
    r_u_img = r_u_img[1:, :-1] - img_data

    dry_img = l_img * u_img * l_u_img * r_u_img
    dry_img[dry_img != 0] = 255
    return dry_img


"""Canny Compute"""
def canny_cost(img, payload=0.4):
    canny_img = np.array(cv2.Canny(img, 50, 150), dtype=np.float)
    j = 0.1
    while np.sum(canny_img) < payload * img.size and j < 50:
        threshold = 50 - j
        canny_img = np.array(cv2.Canny(img, threshold, threshold * 3), dtype=np.float)
        j += 0.1
    return canny_img


"""Compute cost in spatial_domain domain"""
def get_RC(C_SPATIAL, payload):
    dry_img = drt_cost(C_SPATIAL)
    threshold1 = np.median(list(C_SPATIAL[C_SPATIAL <= 127]) + [50])
    threshold2 = np.median(list(C_SPATIAL[C_SPATIAL > 127]) + [150])
    # print(threshold1,threshold2)
    canny_img = np.array(cv2.Canny(C_SPATIAL, threshold1, threshold2), dtype=np.float)
    """Compute probability map by GaussianBlur"""
    density_canny_img = canny_img
    delta = np.count_nonzero(density_canny_img) / (density_canny_img.size - np.count_nonzero(density_canny_img))
    delta_info = int(np.ceil(9 + np.log2(delta)))
    if delta_info % 2 == 0:
        delta_info += 1
    dry_label = P_map(dry_img, canny_img, gauss_k_size=(delta_info, delta_info))
    canny_img = cv2.GaussianBlur(canny_img, (delta_info, delta_info), delta_info)
    gamma = np.median(np.histogram(canny_img)[-1])
    canny_img[canny_img >= gamma] = 255
    canny_img[canny_img < gamma] = 0
    canny_gauss_label = canny_img
    cost = canny_gauss_label * 1 + dry_label * 1
    cost[cost < 0.01] = 0.01
    RC = 1. / cost ** 2
    return RC


"""jpeg"""
payload = 0.4
C_SPATIAL = np.array(cv2.cvtColor(cv2.imread("cover.jpg"), cv2.COLOR_BGR2GRAY), dtype=np.uint8)
jpeg = jio.read("cover.jpg")
C_COEFFS = np.array(jpeg.coef_arrays[0], dtype=np.float)
C_QUANT = np.array(jpeg.quant_tables[0], dtype=np.float)
wetConst = 10 ** 13
sgm = 2**(-6)

"""Compute nzAC """
(k, l) = C_COEFFS.shape
DC_nonzero_counts = 0  # DC coefficient count nonzeros
for i in range(0, k, 8):
    for j in range(0, l, 8):
        if C_COEFFS[i][j] != 0:
            DC_nonzero_counts += 1

nzAC = np.count_nonzero(C_COEFFS) - DC_nonzero_counts

""" Pre-compute impact in spatial_domain domain when a jpeg coefficient is changed by 1"""
spatialImpact = np.zeros(shape=[8,8,8,8],dtype=np.float)

for bcoord_i in range(8):
    for bcoord_j in range(8):
        testCoeffs = np.zeros(shape=[8,8],dtype=np.float)
        testCoeffs[bcoord_i,bcoord_j] = 1
        spatialImpact[bcoord_i,bcoord_j] = cv2.idct(testCoeffs) * C_QUANT[bcoord_i, bcoord_j]

RC = get_RC(C_SPATIAL,payload)

rho = np.zeros(shape=[k,l],dtype=np.float)
"""Computation of costs"""
for row in range(k):
    for col in range(l):
        modRow = np.mod(row,8)
        modCol = np.mod(col,8)
        CoverStegoDiff = spatialImpact[modRow, modCol]
        RC_sub = RC[(row-modRow):(row-modRow+8),(col-modCol):(col-modCol+8)]
        rhoTemp = np.multiply(np.abs(CoverStegoDiff),RC_sub)
        rho[row, col] = np.sum(rhoTemp)
"""jpeg end"""
# print(np.max(rho),np.min(rho))
rhoM1 = rho
rhoP1 = rho

rhoP1[rhoP1 > wetConst] = wetConst
rhoP1[np.isnan(rhoP1)] = wetConst
rhoP1[C_COEFFS > 1023] = wetConst

rhoM1[rhoM1 > wetConst] = wetConst
rhoM1[np.isnan(rhoM1)] = wetConst
rhoM1[C_COEFFS < -1023] = wetConst

S_COEFFS = EmbeddingSimulator(C_COEFFS, rhoP1, rhoM1, round(payload * nzAC))
# print(S_COEFFS.shape)
# print(np.count_nonzero(C_COEFFS-S_COEFFS))
jpeg.coef_arrays[0] -= np.array(C_COEFFS - S_COEFFS, dtype=np.int32)
# print(jpeg.coef_arrays)
jio.write(jpeg, "./stego.jpg")

print("change rate per nzAC = %.4f , nzAC = %d" % (np.count_nonzero(C_COEFFS - S_COEFFS) / nzAC, nzAC))

cover = jio.read("cover.jpg")
cover_coef_array = cover.coef_arrays[0]
cover_quant_tbl = cover.quant_tables[0]
# print(cover.coef_arrays)
stego = jio.read("stego.jpg")
stego_coef_array = stego.coef_arrays[0]
stego_quant_tbl = stego.quant_tables[0]

spatial_cover = plt.imread("cover.jpg")
spatial_stego = plt.imread("stego.jpg")

plt.figure(figsize=(14, 8), dpi=100)

plt.subplot(2, 3, 1)
plt.title("cover")
plt.imshow(spatial_cover, cmap="gray")

plt.subplot(2, 3, 2)
plt.title("stego")
plt.imshow(spatial_stego, cmap="gray")

plt.subplot(2, 3, 3)
plt.title("spatial_distortion")
plt.imshow(spatial_stego - spatial_cover, cmap="gray")

plt.subplot(2, 3, 4)
plt.title("DCT_coefficients_distortion")
plt.imshow(cover_coef_array - stego_coef_array, cmap="gray")

plt.subplot(2, 3, 5)
plt.title("rho")
plt.imshow(rho, cmap="gray")

changeCoef = cover_coef_array[cover_coef_array != stego_coef_array]
Counters = collections.Counter(changeCoef.flatten())
changeNum = []
for i in range(-50, 50, 1):
    changeNum.append(Counters[i])
plt.subplot(2, 3, 6)
plt.title("Number of changed DCT coefficients")
plt.ylabel("Number")
plt.xlabel("DCT_coef")
plt.plot(range(-50, 50, 1), changeNum)
plt.show()

diff = cover_coef_array-stego_coef_array
idx = np.where((diff>0)|(diff<0))
r_diff,g_diff,b_diff= diff.copy(),diff.copy(),diff.copy()
r_diff[idx]=0
g_diff[idx]=255
b_diff[idx]=0
diff = np.array([r_diff,g_diff,b_diff],dtype=np.uint8)
diff = np.transpose(diff,axes=[1,2,0])
img = np.array([C_SPATIAL,C_SPATIAL,C_SPATIAL],dtype=np.uint8)
img = np.transpose(img,axes=[1,2,0])
out= img/255+diff
out = np.clip(out,0,1)
plt.imshow(out)
plt.show()