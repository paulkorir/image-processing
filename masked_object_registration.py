import numpy as np
import matplotlib.pyplot as plt


from skimage import data, draw, io
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi


# image = data.camera()
image = io.imread("/Users/pkorir/Downloads/image01.jpg")
shift = (-22, 13)

rng = np.random.default_rng()
# the mask
corrupted_pixels = rng.choice([False, True], size=image.shape, p=[0.25, 0.75])
# print(f"{corrupted_pixels=}")

# The shift corresponds to the pixel offset relative to the reference image
# offset_image = ndi.shift(image, shift)
# offset_image *= corrupted_pixels
offset_image = io.imread("/Users/pkorir/Downloads/image02.jpg")
print(f"Known offset (row, col): {shift}")

# Determine what the mask is based on which pixels are invalid
# In this case, we know what the mask should be since we corrupted
# the pixels ourselves
mask = corrupted_pixels

# detected_shift, _, _ = phase_cross_correlation(image, offset_image, reference_mask=mask)
detected_shift, error, phasediff = phase_cross_correlation(image, offset_image)
print(f"Detected pixel offset (row, col): {detected_shift}")
print(f"Error: {error}")
print(f"Phase difference: {phasediff}")


# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8, 3))
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3))

ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Corrupted offset image')

# ax3.imshow(mask, cmap='gray')
# ax3.set_axis_off()
# ax3.set_title('Masked pixels')

plt.show()
