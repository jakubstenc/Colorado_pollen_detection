from aicsimageio import AICSImage
import xml.etree.ElementTree as ET

img = AICSImage('test.czi')
print("Channel names:", img.channel_names)

# Let's extract the raw OME-XML metadata
ome_xml = img.ome_metadata
print(ome_xml.channels)
