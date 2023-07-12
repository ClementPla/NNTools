import nntools.dataset as D

img_root = "/home/clement/Documents/phd/DR/MessidorAnnotation/img/images/"
label = "/home/clement/Documents/phd/DR/MessidorAnnotation/labelId/"
dataset = D.ImageDataset(img_root, shape=(1024, 512), keep_size_ratio=True)


print(dataset[0]["image"].shape)
