# Faster RCNN for strawberry disease detection in keras

### The data that I used here is not currently available in the internet. But you can make your own dataset for specific task for object detection.

## For labelling your own data in Pascal-VOC format you can install the popular labelling tool https://github.com/tzutalin/labelImg in your windows machine.

### For data augmentation in Pascal-VOC format I strongly recommend https://medium.com/@bhuwanbhattarai/image-data-augmentation-and-parsing-into-an-xml-file-in-pascal-voc-format-for-object-detection-4cca3d24b33b

## Faster-RCNN architecture

![architecture](https://user-images.githubusercontent.com/26374302/62862751-c52b8b80-bd41-11e9-8078-7b76fbc4773d.JPG)

## some examples of my strawberry data used during training. 

### 1) angular leafspot

![angular_leafspot3](https://user-images.githubusercontent.com/26374302/62831186-3261f280-bc56-11e9-9c56-0cc6e82d78a2.jpg)

### 2) anthracnose fruitrot

![anthrancnose_fruit_rot30](https://user-images.githubusercontent.com/26374302/62831197-5291b180-bc56-11e9-9cea-71a943dd6539.jpg)

### 3) gray mold

![gray_mold89](https://user-images.githubusercontent.com/26374302/62831209-79e87e80-bc56-11e9-90c0-275c40062530.jpg)

### 4) leaf blight

![leaf_blight49](https://user-images.githubusercontent.com/26374302/62831223-a7352c80-bc56-11e9-92e0-51de2367ae00.jpg)

### 5) leaf scorch

![leaf_scorch42](https://user-images.githubusercontent.com/26374302/62831173-00e92700-bc56-11e9-8e9a-d0af1cc174e1.jpg)

### 6) leaf spot

![leaf_spot151](https://user-images.githubusercontent.com/26374302/62831235-cfbd2680-bc56-11e9-9335-d7bec1b917c6.jpg)

### 7) powdery mildew fruit

![powdery_mildew_fruit103](https://user-images.githubusercontent.com/26374302/62831244-f11e1280-bc56-11e9-9d6e-02b99df126ce.jpg)

### 8) powdery mildew leaf 

![powdery_mildew_leaf253](https://user-images.githubusercontent.com/26374302/62831259-190d7600-bc57-11e9-98ae-c1bf1af38bf2.jpg)


hyper parameter used in faster rcnn

![hyper_parameter](https://user-images.githubusercontent.com/26374302/62831269-4b1ed800-bc57-11e9-81c9-bbf89fe6be3e.JPG)

## Training results using Resnet50 as a base network

![Capture](https://user-images.githubusercontent.com/26374302/62830949-47d51d80-bc52-11e9-907c-50af4895beec.JPG)
![Capture1](https://user-images.githubusercontent.com/26374302/62830961-88349b80-bc52-11e9-987c-69d4fbbf8481.JPG)
![Capture3](https://user-images.githubusercontent.com/26374302/62830969-ab5f4b00-bc52-11e9-8ead-914fdb057c52.JPG)
![Capture4](https://user-images.githubusercontent.com/26374302/62830976-c0d47500-bc52-11e9-826c-010fdfcf237b.JPG)

## Detection result 
### for graymold with 9 anchor 3 anchor scale and 3 anchor ratios

![graymold](https://user-images.githubusercontent.com/26374302/62831332-3db61d80-bc58-11e9-8932-09cc242a4477.JPG)

### for mildew with 9 anchor 3 anchor scale and 3 anchor ratios

![mildew](https://user-images.githubusercontent.com/26374302/62831347-91286b80-bc58-11e9-8d1f-ef60a8b6081b.JPG)

