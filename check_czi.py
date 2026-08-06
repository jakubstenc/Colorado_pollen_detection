from aicsimageio import AICSImage
img = AICSImage('test.czi')
print("Shape:", img.shape)
print("Dims:", img.dims.order)
print("Channel names:", img.channel_names)
