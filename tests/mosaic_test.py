import nntools.dataset as D

img_root = "/home/clement/Documents/phd/DR/MessidorAnnotation/img/images/"
label = "/home/clement/Documents/phd/DR/MessidorAnnotation/labelId/"
grade = "/home/clement/Documents/phd/DR/MessidorAnnotation/diagnostic.xls"
dataset = D.ClassificationDataset(img_root, label_filepath=grade, gt_column='retinopathy', file_column='name')

dataset.remap('retinopathy', 'label')

print(dataset[0]["label"])
