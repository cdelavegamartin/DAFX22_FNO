import torch
import numpy as np
import matplotlib.pyplot as plt


def fft():
    x = np.linspace(0, 2 * np.pi, 101, dtype=np.float32)
    y = np.ones_like(x)
    omega = 5
    sinusoid = np.sin(omega * x)
    sinfft = np.fft.rfft(sinusoid)
    mag = np.abs(sinfft)
    print(mag.max())
    print(mag.min())
    print(mag.argmax())
    print(sinfft.shape)
    fig, ax = plt.subplots()
    ax.plot(sinusoid)
    ax.plot(y)
    deltay = 1e-1
    ax.set_ylim(1 - deltay, 1 + deltay)
    fig, ax = plt.subplots()
    deltafreq = 2
    ax.plot(mag)
    # ax.set_ylim(0, 1e-1)
    ax.set_xlim(omega - deltafreq, omega + deltafreq)
    plt.show()
    return


def fft2():
    nx = 101
    ny = 201
    xmax = 1.0
    ymax = 1.0
    x = np.linspace(0, xmax, nx, dtype=np.float32)
    y = np.linspace(0, ymax, ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    print(f"X.shape: {X.shape}")
    # print(X)
    print(f"Y.shape: {Y.shape}")
    # print(Y)
    xfreqs = np.fft.fftfreq(nx, d=xmax / nx)
    yfreqs = np.fft.fftfreq(ny, d=ymax / ny)
    print(f"xfreqs.shape: {xfreqs.shape}")
    print(f"yfreqs.shape: {yfreqs.shape}")
    fig, ax = plt.subplots()
    plt.plot(xfreqs)
    plt.plot(yfreqs)
    # print(xfreqs)
    # print(yfreqs)
    omegax = 10
    omegay = 5
    sinusoid = np.sin(2 * np.pi * omegax * X) * np.sin(2 * np.pi * omegay * Y)
    sinfft2 = np.fft.fft2(sinusoid)
    mag = np.abs(sinfft2)
    print(f"mag.shape: {mag.shape}")
    print(mag.max())
    print(mag.min())
    print(np.unravel_index(mag.argmax(), mag.shape))
    peaks = np.nonzero(mag > 1000)
    print(f"peaks: \n{peaks}")
    print(peaks[0])
    print(peaks[1])
    print(f"xpeaks freq: {xfreqs[peaks[0]]}")
    print(f"ypeaks freq: {yfreqs[peaks[1]]}")

    print(f"sinfft2.shape: {sinfft2.shape}")
    fig, ax = plt.subplots()
    ax.imshow(sinusoid.T)
    fig, ax = plt.subplots()
    ax.imshow(mag.T)
    plt.show()

    return


if __name__ == "__main__":
    # fft2()
    ftrans = np.fft.rfft(np.ones((2, 10), dtype=np.float32))
    freqs = np.fft.rfftfreq(10, d=1.0 / 10)
    print(f"freqs: {freqs}")
    print(f"ftrans.shape: {ftrans.shape}")
    print(f"ftrans: {ftrans}")
    print(f"ftrans[0][0]: {ftrans[0][0]}")
    print(f"ftrans[0][1]: {ftrans[0][1]}")
