from aicsimageio import AICSImage
img = AICSImage('test.czi')
print("Physical pixel sizes:", img.physical_pixel_sizes)
