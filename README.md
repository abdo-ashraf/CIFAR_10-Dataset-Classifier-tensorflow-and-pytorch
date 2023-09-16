# CIFAR_10-Dataset-Classifier
- CIFAR-10 is a dataset that consists of several images divided into the following 10 classes:
    0. Airplanes
    1. Cars
    2. Birds
    3. Cats
    4. Deer
    5. Dogs
    6. Frogs
    7. Horses
    8. Ships
    9. Trucks

- The dataset stands for the Canadian Institute For Advanced Research (CIFAR)
- CIFAR-10 is widely used for machine learning and computer vision applications.
- The dataset consists of 60,000 32x32 color images and 6,000 images of each class.
- Images have low resolution (32x32).

In this notebook uses two models
frist one is CNN model with training accuracy: 87.7%
                         and test accuracy: 77.3%
second model is Upsampling layer ==> ResNet-50 ==> two Dense layers then one last Dense layer for output
training accuracy: 97.8%
test accuracy: 88.4%

each model runed for 10 epochs without data augmentation
