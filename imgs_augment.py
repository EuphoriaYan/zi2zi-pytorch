import os
import sys

import random
import warnings
import math

import numpy as np
import pylab
import scipy.ndimage as ndi
from PIL import Image
from tqdm import tqdm


def autoinvert(image):
    assert np.amin(image) >= 0
    assert np.amax(image) <= 1
    if np.sum(image > 0.9) > np.sum(image < 0.1):
        return 1 - image
    else:
        return image


def zerooneimshow(img):
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).show()
    return


#
# random geometric transformations
#

def random_transform(translation=(-0.05, 0.05), rotation=(-2, 2), scale=(-0.1, 0.1), aniso=(-0.1, 0.1)):
    dx = random.uniform(*translation)
    dy = random.uniform(*translation)
    angle = random.uniform(*rotation)
    angle = angle * np.pi / 180.0
    scale = 10 ** random.uniform(*scale)
    aniso = 10 ** random.uniform(*aniso)
    return dict(angle=angle, scale=scale, aniso=aniso, translation=(dx, dy))


def transform_image(image, angle=0.0, scale=1.0, aniso=1.0, translation=(0, 0), order=1):
    dx, dy = translation
    scale = 1.0 / scale
    c = np.cos(angle)
    s = np.sin(angle)
    sm = np.array([[scale / aniso, 0], [0, scale * aniso]], 'f')
    m = np.array([[c, -s], [s, c]], 'f')
    m = np.dot(sm, m)
    w, h = image.shape
    c = np.array([w, h]) / 2.0
    d = c - np.dot(m, c) + np.array([dx * w, dy * h])
    return ndi.affine_transform(image, m, offset=d, order=order, mode="nearest", output=np.dtype("f"))


#
# random distortions
#

def bounded_gaussian_noise(shape, sigma, maxdelta):
    n, m = shape
    deltas = pylab.rand(2, n, m)
    deltas = ndi.gaussian_filter(deltas, (0, sigma, sigma))
    deltas -= np.amin(deltas)
    deltas /= np.amax(deltas)
    deltas = (2 * deltas - 1) * maxdelta
    return deltas


def distort_with_noise(image, deltas, order=1):
    assert deltas.shape[0] == 2
    assert image.shape == deltas.shape[1:], (image.shape, deltas.shape)
    n, m = image.shape
    xy = np.transpose(np.array(np.meshgrid(
        range(n), range(m))), axes=[0, 2, 1])
    deltas += xy
    return ndi.map_coordinates(image, deltas, order=order, mode="reflect")


def noise_distort1d(shape, sigma=100.0, magnitude=100.0):
    h, w = shape
    noise = ndi.gaussian_filter(pylab.randn(w), sigma)
    noise *= magnitude / np.amax(abs(noise))
    dys = np.array([noise] * h)
    deltas = np.array([dys, np.zeros((h, w))])
    return deltas


#
# mass preserving blur
#

def percent_black(image):
    n = np.prod(image.shape)
    k = np.sum(image < 0.5)
    return k * 100.0 / n


def binary_blur(image, sigma, noise=0.0):
    p = percent_black(image)
    blurred = ndi.gaussian_filter(image, sigma)
    if noise > 0:
        blurred += pylab.randn(*blurred.shape) * noise
    t = np.percentile(blurred, p)
    return np.array(blurred > t, 'f')


#
# multiscale noise
#

def make_noise_at_scale(shape, scale):
    h, w = shape
    h0, w0 = int(h / scale + 1), int(w / scale + 1)
    data = pylab.rand(h0, w0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ndi.zoom(data, scale)
    return result[:h, :w]


def make_multiscale_noise(shape, scales, weights=None, limits=(0.0, 1.0)):
    if weights is None:
        weights = [1.0] * len(scales)
    result = make_noise_at_scale(shape, scales[0]) * weights[0]
    for s, w in zip(scales, weights):
        result += make_noise_at_scale(shape, s) * w
    lo, hi = limits
    result -= np.amin(result)
    result /= np.amax(result)
    result *= (hi - lo)
    result += lo
    return result


def make_multiscale_noise_uniform(shape, srange=(1.0, 100.0), nscales=4, limits=(0.0, 1.0)):
    lo, hi = np.log10(srange[0]), np.log10(srange[1])
    scales = np.random.uniform(size=nscales)
    scales = np.add.accumulate(scales)
    scales -= np.amin(scales)
    scales /= np.amax(scales)
    scales *= hi - lo
    scales += lo
    scales = 10 ** scales
    weights = 2.0 * np.random.uniform(size=nscales)
    return make_multiscale_noise(shape, scales, weights=weights, limits=limits)


#
# random blobs
#

def random_blobs(shape, blobdensity, size, roughness=2.0):
    from random import randint
    from builtins import range  # python2 compatible
    h, w = shape
    numblobs = int(blobdensity * w * h)
    mask = np.zeros((h, w), 'i')
    for i in range(numblobs):
        mask[randint(0, h - 1), randint(0, w - 1)] = 1
    dt = ndi.distance_transform_edt(1 - mask)
    mask = np.array(dt < size, 'f')
    mask = ndi.gaussian_filter(mask, size / (2 * roughness))
    mask -= np.amin(mask)
    mask /= np.amax(mask)
    noise = pylab.rand(h, w)
    noise = ndi.gaussian_filter(noise, size / (2 * roughness))
    noise -= np.amin(noise)
    noise /= np.amax(noise)
    return np.array(mask * noise > 0.5, 'f')


def random_blotches(image, fgblobs, bgblobs, fgscale=10, bgscale=10):
    fg = random_blobs(image.shape, fgblobs, fgscale)
    bg = random_blobs(image.shape, bgblobs, bgscale)
    return np.minimum(np.maximum(image, fg), 1 - bg)


#
# random fibers
#

def make_fiber(l, a, stepsize=0.5):
    angles = np.random.standard_cauchy(l) * a
    angles[0] += 2 * np.pi * pylab.rand()
    angles = np.add.accumulate(angles)
    coss = np.add.accumulate(np.cos(angles) * stepsize)
    sins = np.add.accumulate(np.sin(angles) * stepsize)
    return np.array([coss, sins]).transpose((1, 0))


def make_fibrous_image(shape, nfibers=300, l=300, a=0.2, stepsize=0.5, limits=(0.1, 1.0), blur=1.0):
    h, w = shape
    lo, hi = limits
    result = np.zeros(shape)
    for i in range(nfibers):
        v = pylab.rand() * (hi - lo) + lo
        fiber = make_fiber(l, a, stepsize=stepsize)
        y, x = random.randint(0, h - 1), random.randint(0, w - 1)
        fiber[:, 0] += y
        fiber[:, 0] = np.clip(fiber[:, 0], 0, h - .1)
        fiber[:, 1] += x
        fiber[:, 1] = np.clip(fiber[:, 1], 0, w - .1)
        for y, x in fiber:
            result[int(y), int(x)] = v
    result = ndi.gaussian_filter(result, blur)
    result -= np.amin(result)
    result /= np.amax(result)
    result *= (hi - lo)
    result += lo
    return result


#
# print-like degradation with multiscale noise
#

def printlike_multiscale(image, blur=0.5, blotches=5e-5, paper_range=(0.8, 1.0), ink_range=(0.0, 0.2)):
    selector = autoinvert(image)
    # selector = random_blotches(selector, 3 * blotches, blotches)
    selector = random_blotches(selector, 2 * blotches, blotches)
    paper = make_multiscale_noise_uniform(image.shape, limits=paper_range)
    ink = make_multiscale_noise_uniform(image.shape, limits=ink_range)
    blurred = ndi.gaussian_filter(selector, blur)
    printed = blurred * ink + (1 - blurred) * paper
    return printed


def printlike_fibrous(image, blur=0.5, blotches=5e-5, paper_range=(0.8, 1.0), ink_range=(0.0, 0.2)):
    selector = autoinvert(image)
    selector = random_blotches(selector, 2 * blotches, blotches)
    paper = make_multiscale_noise(image.shape, [1.0, 5.0, 10.0, 50.0], weights=[1.0, 0.3, 0.5, 0.3], limits=paper_range)
    paper -= make_fibrous_image(image.shape, 300, 500, 0.01, limits=(0.0, 0.25), blur=0.5)
    ink = make_multiscale_noise(image.shape, [1.0, 5.0, 10.0, 50.0], limits=ink_range)
    blurred = ndi.gaussian_filter(selector, blur)
    printed = blurred * ink + (1 - blurred) * paper
    return printed


def test():

    img = np.array(Image.open('data/book_pages/imgs_vertical/book_page_0.jpg'))
    # img = distort_with_noise(img, bounded_gaussian_noise(img.shape, 15.0, 5.0))
    # img = ndi.gaussian_filter(img, 0.5)
    img = (binary_blur(img / 255, 0.7, noise=0.1) * 255).astype(np.uint8)
    # img = (printlike_fibrous(img / 255) * 255).astype(np.uint8)
    # img = (printlike_multiscale(img / 255, blur=0.5) * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.show()


def ocrodeg_augment(img):
    img = np.array(img)
    # 50% use distort, 50% use raw
    flag = 0
    if random.random() < 0.5:
        img = distort_with_noise(
            img,
            deltas=bounded_gaussian_noise(
                shape=img.shape,
                sigma=random.uniform(12.0, 20.0),
                maxdelta=random.uniform(3.0, 5.0)
            )
        )
        flag += 1

    img = img / 255

    # 50% use binary blur, 50% use raw
    if random.random() < 0.5:
        img = binary_blur(
            img,
            sigma=random.uniform(0.5, 0.7),
            noise=random.uniform(0.05, 0.1)
        )
        flag += 1

    img = np.clip(img, 0.0, 1.0)

    # raw - 50% use multiscale, 50% use fibrous, 0% use raw
    # flag=1 - 35% use multiscale, 35% use fibrous, 30% use raw
    # flag=2 - 20% use multiscale, 20% use fibrous, 60% use raw
    rnd = random.random()
    if rnd < 0.5 - flag * 0.15:
        img = printlike_multiscale(img, blur=0.5)
    elif rnd < 1 - flag * 0.15:
        img = printlike_fibrous(img)

    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img


def add_noise(img, generate_ratio=0.003, generate_size=0.006):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    h, w = img.shape
    R_max = max(3, int(min(h, w) * generate_size))
    threshold = int(h * w * generate_ratio)

    random_choice_list = []
    for i in range(1, R_max + 1):
        random_choice_list.extend([i] * (R_max - i + 1))

    def cal_dis(pA, pB):
        return math.sqrt((pA[0] - pB[0]) ** 2 + (pA[1] - pB[1]) ** 2)

    cnt = 0
    while True:
        R = random.choice(random_choice_list)
        P_noise_x = random.randint(R, w - 1 - R)
        P_noise_y = random.randint(R, h - 1 - R)
        for i in range(P_noise_x - R, P_noise_x + R):
            for j in range(P_noise_y - R, P_noise_y + R):
                if cal_dis((i, j), (P_noise_x, P_noise_y)) < R:
                    if random.random() < 0.6:
                        img[j][i] = random.randint(0, 255)
        cnt += 2 * R
        if cnt >= threshold:
            break

    R_max *= 2
    random_choice_list = []
    for i in range(1, R_max + 1):
        random_choice_list.extend([i] * (R_max - i + 1))
    cnt = 0
    while True:
        R = random.choice(random_choice_list)
        P_noise_x = random.randint(0, w - 1 - R)
        P_noise_y = random.randint(0, h - 1 - R)
        for i in range(P_noise_x + 1, P_noise_x + R):
            for j in range(P_noise_y + 1, P_noise_y + R):
                if random.random() < 0.6:
                    img[j][i] = random.randint(0, 255)
        cnt += R
        if cnt >= threshold:
            break

    img = Image.fromarray(img)
    return img


if __name__ == '__main__':
    # test()
    # augment()
    pass