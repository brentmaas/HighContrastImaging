import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits

def create_cube(name, folder, size, apsize, apfile, star, planet, noise, noisemap, radius, phase, angles, anglesfile, diagrams):
    valid = True
    if not apfile is None:
        if not os.path.exists(apfile):
            print("The specified aperture file could not be found.")
            valid = False
        else:
            aperture = fits.getdata(apfile)
    elif not size is None and not apsize is None:
        y, x = np.mgrid[0:size,0:size]
        x -= size // 2
        y -= size // 2
        aperture = (np.sqrt(x * x + y * y) <= apsize).astype(np.float32)
    else:
        print("Please specify a valid positive image size and a valid positive aperture size, or an aperture file.")
        valid = False
    
    if star is None or star < 0:
        print("Please specify a valid positive star flux.")
        valid = False
    
    if planet is None or not (type(planet) == list and all([p >= 0 for p in planet])):
        print("Please specify valid positive planet fluxes.")
        valid = False
    
    if not noisemap is None:
        if not os.path.exists(noisemap):
            print("The specified noise map could not be found.")
            valid = False
        else:
            noiseimg = fits.getdata(noisemap)
    
    if radius is None or not (type(radius) == list and all([r >= 0 for r in radius]) and len(radius) == len(planet)):
        print("Please specify valid positive radii.")
        valid = False
    
    if phase is None or not (type(phase) == list and len(phase) == len(planet)):
        print("Please specify valid phases.")
        valid = False
    
    if angles is None or not type(angles) == list:
        print("Please specify valid angles.")
        valid = False
    
    if not valid:
        exit()
    
    os.makedirs(folder, exist_ok=True)
    
    data = np.zeros((len(angles), aperture.shape[0], aperture.shape[1]))
    
    data[:,aperture.shape[0]//2,aperture.shape[1]//2] = star
    for i in range(len(angles)):
        for j in range(len(planet)):
            data[i,int(aperture.shape[0]//2+radius[j]*np.sin(np.deg2rad(phase[j]+angles[i]))),int(aperture.shape[1]//2+radius[j]*np.cos(np.deg2rad(phase[j]+angles[i])))] += planet[j]
    
    data = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.sqrt(data), axes=(1, 2)), axes=(1, 2)), axes=(1, 2))
    data *= aperture * size / np.sqrt(np.sum(aperture))
    data = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(data, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))
    
    data = np.random.poisson(np.abs(data) ** 2 + noise)
    if not noisemap is None and os.file.exists(noisemap):
        data += noiseimg
    
    fits.writeto(f"{folder}/{name}.fits", data, overwrite=True)
    if anglesfile:
        fits.writeto(f"{folder}/{name}.angles.fits", np.array(angles), overwrite=True)
    if diagrams:
        numwidth = int(np.ceil(np.sqrt(len(angles))))
        numheight = int(np.ceil(len(angles) / numwidth))
        plt.figure(figsize=(numwidth, numheight))
        plt.style.use("dark_background")
        for i in range(len(angles)):
            plt.subplot(numheight, numwidth, i + 1)
            plt.scatter([aperture.shape[1]//2], [aperture.shape[0]//2], marker="*", color="yellow")
            plt.scatter([aperture.shape[1]//2+radius[j]*np.cos(np.deg2rad(phase[j]+angles[i])) for j in range(len(planet))], [aperture.shape[0]//2+radius[j]*np.sin(np.deg2rad(phase[j]+angles[i])) for j in range(len(planet))], marker="o", color="red")
            plt.xlim(0, aperture.shape[1])
            plt.ylim(0, aperture.shape[0])
            plt.axis('equal')
            plt.axis("off")
            plt.plot([0, 0, aperture.shape[1], aperture.shape[1], 0], [0, aperture.shape[0], aperture.shape[0], 0, 0], color="white", linewidth=1)
            plt.title(f"{i+1}", pad=1, fontdict={"fontsize": "small"})
        plt.savefig(f"{folder}/{name}.png", bbox_inches='tight', dpi=200)
        plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="""Command line utility to create a datacube of synthetic data of pointlike star and planets to use for high contrast imaging testing.
                                                    The produced images simulate observations during a single night while rotating the telescope along the line of sight.
                                                    These images can be reduced with, for example, Angular Differential Imaging.
                                                    The number of values for each planet parameter must be equal to 1 or the desired number of planets.""")
    parser.add_argument("name", help="Name of the cube, which will be used for the filename.", type=str)
    parser.add_argument("--folder", help="Name of the target folder. './cubes' by default.", default="./cubes")
    parser.add_argument("--size", help="Dimension of the images in the cube in pixels. Also requires --apsize.", type=int, default=None)
    parser.add_argument("--apsize", help="Radius of the aperture. Also requires --size.", type=float, default=None)
    parser.add_argument("--apfile", help="Path to a FITS file describing the aperture. Overrides --size and --apsize.", type=str, default=None)
    parser.add_argument("--star", help="Total flux of the central star in counts.", type=float, default=None)
    parser.add_argument("--planet", help="Total flux of (each of) the planet(s) in counts.", type=float, nargs="+", default=None)
    parser.add_argument("--noise", help="Noise intensity in counts per square pixel.", type=float, default=0)
    parser.add_argument("--noisemap", help="Path to a FITS file containing a noise pattern (speckles, for example).", type=str, default=None)
    parser.add_argument("--radius", help="Radius of the planet(s) around the star in pixels.", type=float, nargs="+", default=None)
    parser.add_argument("--phase", help="Angular offset of the planet(s) around the star in degrees.", type=float, nargs="+", default=None)
    parser.add_argument("--angles", help="Rotation angles for each images in degrees.", type=float, nargs="+", default=None)
    parser.add_argument("--anglesfile", help="Also generate a FITS file containing the specified rotation angles.", type=bool, nargs="?", const=True, default=False)
    parser.add_argument("--diagrams", help="Also produce diagrams of each of the images.", type=bool, nargs="?", const=True, default=False)
    args = parser.parse_args()
    
    create_cube(**vars(args))