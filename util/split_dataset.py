import os,shutil
def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(srcfile,dstfile)
        print ("move %s -> %s"%( srcfile,dstfile))
filePath_normal='F:/Edge_download/data/dicom/new/dicom_archive_v2.tar/dicom_clean/'
k=1
for root, dirs, files in os.walk(filePath_normal):
    for f in sorted(files):
        if k<=939:
            k+=1
            eachpath = os.path.join(root, f)
            mymovefile(eachpath,'F:/Edge_download/data/dicom/new/dicom_archive_v2.tar/val/')