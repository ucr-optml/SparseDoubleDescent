# Sparse Double Descent

### All the trained models will be stored in a 'result' folder. If it's not existed, please create one!!!
### ResNet_CIFAR100.ipynb is the main experiment code.
### Utils.py contains data loading,  
pruning, training, and all the possible calculations related to neural network training & testing pipeline. Note: in addition to CIFAR-100, you can also load CIFAR-10 or MNIST with the same code.
### DataAug.py contains data augmentation code. It's imported by Utils.py, so it's not necessary to import it separately.
### ResNet.py provides other ResNet models. You can delete the code for ResNet20 in ResNet_CIFAR10.ipynb and import this .py file.
