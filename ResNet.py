
# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------


import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation,AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomUniform
class ResNet(Model):
    def __init__(self,version,net_n,input_shape,num_classes,reduceTop=0,activation='linear',bn=False):
        super(ResNet,self).__init__()
        self.reduceTop=reduceTop
        self.version=version
        self.n=net_n
        self.num_classes=num_classes
        if self.version == 1:
            depth = self.n * 6 + 2
        elif self.version == 2:
            depth = self.n * 9 + 2
        self.batch_normalization=bn
        self.model_type = 'ResNet%dv%d' % (depth, self.version)
        self.depth=depth
        print(self.model_type)
        
        if self.version != 1:
            print("[ERROR] Do not support ResNet-V2")
            return

        
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
            
        self.first_res_layer=self.resnet_layer(
                                      batch_normalization=self.batch_normalization
                              )
        
        self.dim_adjust_layers=[]
        self.residual_blocks=[]
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        # Instantiate the stack of residual units
        for stack in range(3):
            #print('stack: ',stack)
            self.dim_adjust_layers.append([])
            self.residual_blocks.append([])
            
            for res_block in range(num_res_blocks):
                self.residual_blocks[stack].append([])
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample

                self.residual_blocks[stack][res_block].append(self.resnet_layer(
                                 num_filters=num_filters,
                                 strides=strides,
                                    batch_normalization=self.batch_normalization
                                      ))
                self.residual_blocks[stack][res_block].append(self.resnet_layer(
                                 num_filters=num_filters,
                                 activation=None,
                                     batch_normalization=self.batch_normalization
                                      ))
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    self.dim_adjust_layers[stack].append(
                        self.resnet_layer(
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False))
            num_filters *= 2

        #self.last=Dense(self.num_classes,activation=activation,bias_initializer='zeros',kernel_initializer='he_normal')#RandomUniform(minval=-0.5,maxval=0.5))
        self.last = Dense(self.num_classes, activation='linear',
                          kernel_initializer='he_normal')
        self.last_act=Activation(activation)
#         for stack in range(3):
#             print()
#             for res_block in range(num_res_blocks):
#                 print('    ',self.residual_blocks[stack][res_block])
                
#         print(self.residual_blocks,self.dim_adjust_layers)

        

    def handel_res_layer(self,x,res_layer):
        for layer in res_layer:
            x=layer(x)
        return x
    
    def call(self,x):
        x=self.handel_res_layer(x,self.first_res_layer)
        
        num_filters = 16
        num_res_blocks = int((self.depth - 2) / 6)
        # Instantiate the stack of residual units
        for stack in range(3):       
            for res_block in range(num_res_blocks):
                y=self.handel_res_layer(x,self.residual_blocks[stack][res_block][0])
                y=self.handel_res_layer(y,self.residual_blocks[stack][res_block][1])
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x=self.handel_res_layer(x,self.dim_adjust_layers[stack][0])
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
        

        x=AveragePooling2D(pool_size=8)(x)
        x=Flatten()(x)
        return self.last_act(self.last(x))

    def logits(self,x):
        x=self.handel_res_layer(x,self.first_res_layer)
        
        num_filters = 16
        num_res_blocks = int((self.depth - 2) / 6)
        # Instantiate the stack of residual units
        for stack in range(3):       
            for res_block in range(num_res_blocks):
                y=self.handel_res_layer(x,self.residual_blocks[stack][res_block][0])
                y=self.handel_res_layer(y,self.residual_blocks[stack][res_block][1])
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x=self.handel_res_layer(x,self.dim_adjust_layers[stack][0])
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)

            # Add classifier on top.
            # v1 does not use BN after last shortcut connection-ReLU
        x=AveragePooling2D(pool_size=8)(x)
        x=Flatten()(x)
        return self.last(x)
                   
        
            
    def resnet_layer(self,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder
        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)
        # Returns
            x (tensor): tensor as input to the next layer
        """
        ret=[]
        conv=Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4)
                      )
        if conv_first:
            ret.append(conv)

            if batch_normalization:
                ret.append(BatchNormalization())
            if activation is not None:
                ret.append(Activation(activation))
        else:
            if batch_normalization:
                ret.append(BatchNormalization())
            if activation is not None:
                ret.append(Activation(activation))
            ret.append(conv)
        return ret

