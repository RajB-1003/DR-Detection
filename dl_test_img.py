import urllib.request
url = 'https://raw.githubusercontent.com/mdraihanujjaman/Diabetic-Retinopathy-Detection/master/dataset/train/No_DR/009b03ae2852.png'
urllib.request.urlretrieve(url, 'test_normal.png')
print("Downloaded test_normal.png")
