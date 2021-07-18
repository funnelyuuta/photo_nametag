import pyexiv2

def exif(image_path, memberlist):
    exifnames = []
    img = pyexiv2.Image(str(image_path))
    metadata = img.read_exif()
    for name in memberlist:
        exifnames.append(name)
    exifstr = ";".join(exifnames)
    exifnames2 = exifstr + " "
    metadata["Exif.Image.XPKeywords"] = exifnames2
    img.modify_exif(metadata)
    img.close()
