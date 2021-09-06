import nntools.dataset as D



img_root = '/home/clement/Documents/phd/DR/MessidorAnnotation/img/images/'
label = '/home/clement/Documents/phd/DR/MessidorAnnotation/labelId/'
dataset = D.SegmentationDataset(img_root, label, shape=(1024, 1024))

img = dataset.get_mosaic(8, show=True, fig_size=1, add_labels=True, save='test.png')
