import collections

import cv2.cv2 as cv2
import jpegio as jio
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve2d
from sklearn.cluster import KMeans


def Qmatrix(quality):
    if quality < 1:
        return
    Q50 = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])  # Quantization matrix for 50% quality

    if quality > 50:
        Q = np.round(2 * Q50 * (1 - quality / 100))
        Q[Q < 1] = 1

    else:
        Q = np.round(Q50 * 50 / quality)
        Q[Q > 255] = 255

    Q = Q.astype(np.int)
    return Q

# np.set_printoptions(formatter={'float': '{:.4f}'.format})
payload = 0.4
quality_factor = 75
quality_mat = loadmat("default_gray_jpeg_obj/"+str(quality_factor)+".mat")
C_quant_tables = quality_mat["quant_tables"]
C_quant_arrays = quality_mat["coef_arrays"]


PC_SPATIAL = np.array(cv2.cvtColor(cv2.imread("images_precover/1.pgm"), cv2.COLOR_BGR2GRAY), dtype=np.float)
C_row,C_col = PC_SPATIAL.shape

DCT_real = np.zeros_like(PC_SPATIAL)
for i in range(C_row//8):
    for j in range(C_col//8):
            DCT_real[i*8:(i+1)*8,j*8:(j+1)*8]=cv2.dct(PC_SPATIAL[i*8:(i+1)*8,j*8:(j+1)*8]-128)/C_quant_tables

DCT_rounded = np.round(DCT_real)
# print(DCT_rounded)

mat_jpeg = jio.read("default_gray_jpeg_obj/"+str(quality_factor)+".jpg")

"""Save cover"""
mat_jpeg.coef_arrays[0] -= np.array(mat_jpeg.coef_arrays[0]-DCT_rounded,dtype=np.int8)
jio.write(mat_jpeg,"images_cover/1.jpg")

e = DCT_rounded - DCT_real    # Compute rounding error
direction = np.sign(e)

"""   ***********   """
"""   BEGIN costs   """
"""   ***********   """
wetConst = 10**13
sgm = 2**(-6)

def P_map(img, canny_img, gauss_k_size=(7, 7), iter=100):
    img = np.array(img, dtype=np.uint8)
    row, col = img.shape[0], img.shape[1]
    gauss_img = cv2.GaussianBlur(img, gauss_k_size, 0)
    canny = canny_img.copy()
    canny[canny != 0] = 1
    img_data = gauss_img.flatten().reshape([row * col, 1])
    if np.max(img_data) == np.min(img_data):
        km = KMeans(1, max_iter=iter)
    else:
        km = KMeans(2, max_iter=iter)
    label = km.fit_predict(img_data)
    label = label.reshape([row, col])
    label = (label - np.mean(label)) / (np.std(label) + 2 ** (-6))
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


def dry_cost(img):
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

def canny_cost(img):
    canny_img = np.array(cv2.Canny(img, 50, 150), dtype=np.float)
    return canny_img

"""Compute cost in spatial_domain domain"""

def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def get_RC(C_SPATIAL, payload):
    dry_img = dry_cost(C_SPATIAL)

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


""" Pre-compute impact in spatial_domain domain when a jpeg coefficient is changed by 1"""
spatialImpact = np.zeros(shape=[8, 8, 8, 8], dtype=np.float)
for bcoord_i in range(8):
    for bcoord_j in range(8):
        testCoeffs = np.zeros(shape=[8, 8], dtype=np.float)
        testCoeffs[bcoord_i, bcoord_j] = 1
        spatialImpact[bcoord_i, bcoord_j] = cv2.idct(testCoeffs) * C_quant_tables[bcoord_i, bcoord_j]

"""Compute embedding cost in spatial domain"""
RC = get_RC(np.array(PC_SPATIAL, dtype=np.uint8), payload)

"""Computation of costs"""
rho = np.zeros(shape=[C_row, C_col], dtype=np.float)
for row in range(C_row):
    for col in range(C_col):
        sub_e = e[row, col]
        modRow = np.mod(row, 8)
        modCol = np.mod(col, 8)
        CoverStegoDiff = spatialImpact[modRow, modCol]
        R_PC_sub = RC[(row - modRow):(row - modRow + 8), (col - modCol):(col - modCol + 8)]
        rhoTemp = np.abs(CoverStegoDiff ) * R_PC_sub
        rho[row, col] = np.sum(rhoTemp)

# print(np.max(rho),np.min(rho))
rho = rho + 10**(-4)
rho[rho > wetConst] = wetConst
rho[np.isnan(rho)] = wetConst
plot_rho = rho.copy()
"""Avoid 04 coefficients with e=0.5"""
maxCostMat = np.zeros_like(rho)
maxCostMat[::8,::8] = 1
maxCostMat[5::8, 1::8] = 1
maxCostMat[1::8, 5::8] = 1
maxCostMat[5::8, 5::8] = 1

pre_quant_tables = Qmatrix(100)

g= C_quant_tables/np.gcd(C_quant_tables,pre_quant_tables)
contrib_ind = 2*((g%2==0)|(g==1))-1


for i in range(C_row//8):
    for j in range(C_col//8):
        rho[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] =rho[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]*contrib_ind

rho[rho<0]=wetConst
rhoP1 = rho
rhoM1 = rho
weight_rho = (1-2*np.abs(e))*rho
rhoP1[direction==1]=weight_rho[direction==1]
rhoM1[direction==-1]=weight_rho[direction==-1]

rhoP1[DCT_rounded > 1022] = wetConst
rhoM1[DCT_rounded < -1022] = wetConst

rhoP1[(maxCostMat==1) & (abs(e)>0.4999)] = wetConst
rhoM1[(maxCostMat==1) & (abs(e)>0.4999)] = wetConst



# print(rho)
"""   ***********   """
"""   END costs   """
"""   ***********   """

"""Embedding simulation"""
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


"""Embedding simulator simulates the embedding made by the best possible ternary coding method 
(it embeds on the entropy bound). 
This can be achieved in practice using "Multi-layered  syndrome-trellis codes" (ML STC) 
that are asymptotically aproaching the bound."""
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


"""Compute message lenght for each run"""
DC_nonzero_counts = 0 # DC coefficient count nonzeros
for i in range(0,C_row,8):
    for j in range(0,C_col, 8):
        if DCT_rounded[i][j] != 0:
            DC_nonzero_counts+=1
nzAC = np.count_nonzero(DCT_rounded) - DC_nonzero_counts
totalMessageLength = np.round(payload*nzAC)

"""Embedding"""
S_COEFFS = EmbeddingSimulator(DCT_rounded,rhoP1,rhoM1,totalMessageLength)


"""Save stego"""
mat_jpeg.coef_arrays[0] -= np.array(mat_jpeg.coef_arrays[0]-S_COEFFS,dtype=np.int8)
jio.write(mat_jpeg,"images_stego/1.jpg")



print("change rate per nzAC = %.4f , nzAC = %d"%(np.count_nonzero(DCT_rounded-S_COEFFS)/nzAC,nzAC))

cover = jio.read("images_cover/1.jpg")
cover_coef_array = cover.coef_arrays[0]
cover_quant_tbl = cover.quant_tables[0]
# print(cover.coef_arrays)
stego = jio.read("images_stego/1.jpg")
stego_coef_array = stego.coef_arrays[0]
stego_quant_tbl = stego.quant_tables[0]

spatial_cover = plt.imread("images_cover/1.jpg")
spatial_stego = plt.imread("images_stego/1.jpg")

plt.figure(num="si-Canny_Gauss-newPQ",figsize=(14,8),dpi=100)

plt.subplot(2,3,1)
plt.title("cover")
plt.imshow(spatial_cover,cmap="gray")

plt.subplot(2,3,2)
plt.title("stego")
plt.imshow(spatial_stego,cmap="gray")

plt.subplot(2,3,3)
plt.title("spatial_distortion")
plt.imshow(spatial_stego-spatial_cover,cmap="gray")

plt.subplot(2,3,4)
plt.title("DCT_coefficients_distortion")
plt.imshow(cover_coef_array-stego_coef_array,cmap="gray")

plt.subplot(2,3,5)
plt.title("rho")
plt.imshow(plot_rho,cmap="gray")

changeCoef = cover_coef_array[cover_coef_array!=stego_coef_array]
Counters = collections.Counter(changeCoef.flatten())
changeNum=[]
for i in range(-50,50,1):
    changeNum.append(Counters[i])
plt.subplot(2,3,6)
plt.title("Number of changed DCT coefficients")
plt.ylabel("Number")
plt.xlabel("DCT_coef")
plt.plot(range(-50,50,1),changeNum)
plt.show()

diff = cover_coef_array-stego_coef_array
idx = np.where((diff>0)|(diff<0))
r_diff,g_diff,b_diff= diff.copy(),diff.copy(),diff.copy()
r_diff[idx]=0
g_diff[idx]=255
b_diff[idx]=0
diff = np.array([r_diff,g_diff,b_diff],dtype=np.uint8)
diff = np.transpose(diff,axes=[1,2,0])
img = np.array([PC_SPATIAL,PC_SPATIAL,PC_SPATIAL],dtype=np.uint8)
img = np.transpose(img,axes=[1,2,0])
out= img/255+diff
out = np.clip(out,0,1)
plt.imshow(out)
plt.show()






