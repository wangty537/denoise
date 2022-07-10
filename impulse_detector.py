import numpy as np

import cv2
import matplotlib.pyplot as plt


def show_img4(im1, im2, im3, im4):
    plt.figure()
    plt.subplot(221)
    plt.imshow(im1)

    plt.subplot(222)
    plt.imshow(im2)

    plt.subplot(223)
    plt.imshow(im3)

    plt.subplot(224)
    plt.imshow(im4)
    plt.show()


def add_impulse_noise(im, p):
    im2 = im.copy()
    n = np.random.randint(0, 256, im.shape)
    sel = np.random.rand(*im.shape) < p  # uniform distribution , [0,1) 或者用uniform函数
    im2[sel] = n[sel]
    return im2


def add_salt_and_pepper_noise(im, p):
    im2 = im.copy()
    n = np.random.randint(0, 2, im.shape) * 255
    # print(n[:10,:10,0])
    sel = np.random.rand(*im.shape) < p  # uniform distribution , [0,1)
    im2[sel] = n[sel]
    return im2


def add_gaussian_noise(im, sigma):
    im2 = im.copy().astype(np.int32)
    im2 += np.random.normal(0.0, sigma, im2.shape).astype(np.int32)
    return np.clip(im2, 0, 255).astype(np.uint8)


def cal_road(im_gray, radiu=3):
    h, w = im_gray.shape
    if radiu == 1:
        m = 5
    else:
        m = radiu * radiu // 3
    road = np.zeros(im_gray.shape, np.int32)
    for i in np.arange(radiu, h-radiu):
        for j in np.arange(radiu, w-radiu):
            c0 = im_gray[i, j]
            ro = []
            for ii in np.arange(-radiu, radiu+1):
                for jj in np.arange(-radiu, radiu+1):
                    it = i + ii
                    jt = j + jj
                    c1 = im_gray[it, jt]
                    ro.append(abs(int(c1)-c0))
            ro = np.sort(np.array(ro))
            road[i, j] = int(np.sum(ro[:m]))
            # if i < 10 and j < 10:
            #     print(m, ro, sum(ro[:m]), road[i, j])
    return road

def trilateral_filter(im_gray, road, radiu=1, sigma_s=5, sigma_r=20, sigma_i=40, sigma_j=50):
    im2 = im_gray.copy()
    h, w = im2.shape

    for i in np.arange(radiu, h-radiu):
        for j in np.arange(radiu, w-radiu):
            c0 = im2[i, j]
            sum_v = 0
            w_v = 0
            for ii in np.arange(-radiu, radiu+1):
                for jj in np.arange(-radiu, radiu+1):
                    it = i + ii
                    jt = j + jj
                    c1 = im2[it, jt]

                    a = (it-i)*(it-i) + (jt-j)*(jt-j)
                    b = (int(c1) - c0)*(int(c1)-c0)
                    ws = np.exp(-a / 2 / (sigma_s * sigma_s))
                    wr = np.exp(-b / 2 / (sigma_r * sigma_r))

                    c = road[it, jt] * road[it, jt]
                    d = (road[it, jt] + road[i, j])**2
                    wi = np.exp(-c / 2 / (sigma_i * sigma_i))
                    J = 1 - np.exp(-d/8/(sigma_j * sigma_j))
                    weight = ws * (wr**(1-J)) * (wi**J)
                    sum_v += weight * c1
                    w_v += weight

            im2[i, j] = np.round(sum_v / w_v).astype(np.int32)
            # if i < 10 and j < 10:
            #     print(i, j, sum_v, w_v, im2[i, j], im_gray[i, j])
    im2 = np.clip(im2, 0, 255).astype(np.uint8)
    return im2


if __name__ == "__main__":
    file = r'G:\dataset\kodak\kodim03.png'
    im = cv2.imread(file)
    im = cv2.resize(im, [200, 200])
    im = im[..., ::-1]

    im_impulse_noise = add_impulse_noise(im, 0.2)
    im_impulse_noise2 = add_impulse_noise(im, 0.4)
    im_impulse_noise3 = add_impulse_noise(im, 0.7)
    #show_img4(im, im_impulse_noise, im_impulse_noise2, im_impulse_noise3)

    im_salt_and_pepper = add_salt_and_pepper_noise(im, 0.2)
    im_salt_and_pepper2 = add_salt_and_pepper_noise(im, 0.4)
    im_salt_and_pepper3 = add_salt_and_pepper_noise(im, 0.7)
    #show_img4(im, im_salt_and_pepper, im_salt_and_pepper2, im_salt_and_pepper3)

    im_mix_noise = add_gaussian_noise(im_salt_and_pepper, 10)

    r, g, b = cv2.split(im_mix_noise)

    radiu = 1
    road = cal_road(r, radiu)
    r_denoised = trilateral_filter(r, road, radiu,
                                    sigma_s=5, sigma_r=20, sigma_i=40, sigma_j=50)

    road = cal_road(g, radiu)
    g_denoised = trilateral_filter(g, road, radiu,
                                   sigma_s=5, sigma_r=20, sigma_i=40, sigma_j=50)
    road = cal_road(b, radiu)
    b_denoised = trilateral_filter(b, road, radiu,
                                   sigma_s=5, sigma_r=20, sigma_i=40, sigma_j=50)

    im_res = cv2.merge([r_denoised, g_denoised, b_denoised])

    im_bi = cv2.bilateralFilter(im_mix_noise, 3, 10, 5)
    show_img4(im, im_mix_noise, im_bi, im_res)