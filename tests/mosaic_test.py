import nntools.dataset as D

img_root = "/home/clement/Documents/phd/DR/MessidorAnnotation/img/images/"
label = "/home/clement/Documents/phd/DR/MessidorAnnotation/labelId/"
grade = "/home/clement/Documents/phd/DR/MessidorAnnotation/diagnostic.xls"
dataset = D.ClassificationDataset(img_root, label_filepath=grade, gt_column='retinopathy', file_column='name')

dataset.remap('retinopathy', 'label')

print(dataset[0]["label"])
dataset = D.SegmentationDataset(img_root=img_root, shape=(512, 512), mask_root=label)

print("Segmentation", len(dataset))

imgDataset = D.ImageDataset(img_root=img_root)
print("Image", len(imgDataset))

multiImgDataset = D.MultiImageDataset(img_root={'foo':img_root, 'bar':label})

print("Multi Image", len(multiImgDataset))
