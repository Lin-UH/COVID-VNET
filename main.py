import pydicom
#进行绘图
from matplotlib import pylab
from matplotlib import pyplot as plt

filePath='F:/Edge_download/data/dicom/new/dicom_archive_v2.tar/dicom_clean/5032497707401895.dcm'
ds=pydicom.read_file(filePath)
print(ds)
print(ds.dir("pat"))
pix = ds.pixel_array
print(pix.shape)
plt.imshow(pix,cmap='gray')
plt.show()
# pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
# pylab.show()



