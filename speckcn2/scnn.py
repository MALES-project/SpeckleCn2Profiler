import torch
from escnn import gspaces
from escnn import nn


class C8SteerableCNN(torch.nn.Module):

    def __init__(self, nscreens, in_image_res):

        super(C8SteerableCNN, self).__init__()

        # Compute the size of the feature map
        self.nfeatures = in_image_res

        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.rot2dOnR2(N=8)

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # This mask is to remove the 'corner' pixels in the input image, that would get outside of the square while rotating
        self.mask = nn.MaskModule(in_type, in_image_res, margin=0),

        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = nn.FieldType(self.r2_act, 24 * [self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.InnerBatchNorm(out_type), nn.ReLU(out_type, inplace=True))
        # **Out dim of a convolution is:
        # (in_image_res - kernel_size + 2*pad)/stride + 1
        self.nfeatures = int((self.nfeatures - 7 + 2 * 1) / 1 + 1)

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type), nn.ReLU(out_type, inplace=True))
        self.nfeatures = int((self.nfeatures - 5 + 2 * 2) / 1 + 1)

        # pool 1
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))
        # **Out dim of pooling is:
        # (in + 2*pad - (kernel-1) - 1) / stride + 1
        # where: kernel = 2*int(round(3*sigma))+1
        # and  : pad    = int((kernel-1)//2)
        k = 2 * int(round(3 * 0.66)) + 1
        p = int((k - 1) // 2)
        self.nfeatures = int((self.nfeatures + 2 * p - (k - 1) - 1) / 2 + 1)

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type), nn.ReLU(out_type, inplace=True))
        self.nfeatures = int((self.nfeatures - 5 + 2 * 2) / 1 + 1)

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96 * [self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type), nn.ReLU(out_type, inplace=True))
        self.nfeatures = int((self.nfeatures - 5 + 2 * 2) / 1 + 1)

        # pool 2
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))
        self.nfeatures = int((self.nfeatures + 2 * p - (k - 1) - 1) / 2 + 1)

        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96 * [self.r2_act.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type), nn.ReLU(out_type, inplace=True))
        self.nfeatures = int((self.nfeatures - 5 + 2 * 2) / 1 + 1)

        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr])
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.InnerBatchNorm(out_type), nn.ReLU(out_type, inplace=True))
        self.nfeatures = int((self.nfeatures - 5 + 2 * 1) / 1 + 1)

        # pool 3
        #self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        self.pool3 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type,
                                           sigma=0.66,
                                           stride=1,
                                           padding=0))
        self.nfeatures = int((self.nfeatures + 2 * 0 - (k - 1) - 1) / 1 + 1)

        self.gpool = nn.GroupPooling(out_type)

        # number of output channels
        c = self.gpool.out_type.size * self.nfeatures * self.nfeatures

        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, nscreens),
            torch.nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        x = self.block5(x)
        x = self.block6(x)

        # pool over the spatial dimensions
        x = self.pool3(x)

        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x


class C16SteerableCNN(torch.nn.Module):

    def __init__(self, nscreens, in_image_res):

        super(C16SteerableCNN, self).__init__()

        # Compute the size of the feature map
        self.nfeatures = in_image_res

        # the model is equivariant under rotations by 22.5 degrees, modelled by C16
        self.r2_act = gspaces.rot2dOnR2(N=16)

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # This mask is to remove the 'corner' pixels in the input image, that would get outside of the square while rotating
        self.mask = nn.MaskModule(in_type, in_image_res, margin=0),

        # convolution 1
        # first specify the output type of the convolutional layer
        out_type = nn.FieldType(self.r2_act, 24 * [self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.InnerBatchNorm(out_type), nn.ReLU(out_type, inplace=True))
        # **Out dim of a convolution is:
        # (in_image_res - kernel_size + 2*pad)/stride + 1
        self.nfeatures = int((self.nfeatures - 7 + 2 * 1) / 1 + 1)

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type,
                      out_type,
                      kernel_size=7,
                      stride=2,
                      padding=2,
                      bias=False), nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True))
        self.nfeatures = int((self.nfeatures - 7 + 2 * 2) / 2 + 1)

        # pool 1
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))
        # **Out dim of pooling is:
        # (in + 2*pad - (kernel-1) - 1) / stride + 1
        # where: kernel = 2*int(round(3*sigma))+1
        # and  : pad    = int((kernel-1)//2)
        k = 2 * int(round(3 * 0.66)) + 1
        p = int((k - 1) // 2)
        self.nfeatures = int((self.nfeatures + 2 * p - (k - 1) - 1) / 2 + 1)

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type,
                      out_type,
                      kernel_size=7,
                      stride=2,
                      padding=2,
                      bias=False), nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True))
        self.nfeatures = int((self.nfeatures - 7 + 2 * 2) / 2 + 1)

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96 * [self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type,
                      out_type,
                      kernel_size=7,
                      stride=2,
                      padding=2,
                      bias=False), nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True))
        self.nfeatures = int((self.nfeatures - 7 + 2 * 2) / 2 + 1)

        # pool 2
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))
        self.nfeatures = int((self.nfeatures + 2 * p - (k - 1) - 1) / 2 + 1)

        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96 * [self.r2_act.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type), nn.ReLU(out_type, inplace=True))
        self.nfeatures = int((self.nfeatures - 5 + 2 * 2) / 1 + 1)

        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr])
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.InnerBatchNorm(out_type), nn.ReLU(out_type, inplace=True))
        self.nfeatures = int((self.nfeatures - 5 + 2 * 1) / 1 + 1)

        # pool 3
        #self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        self.pool3 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type,
                                           sigma=0.66,
                                           stride=1,
                                           padding=0))
        self.nfeatures = int((self.nfeatures + 2 * 0 - (k - 1) - 1) / 1 + 1)

        self.gpool = nn.GroupPooling(out_type)

        # number of output channels
        c = self.gpool.out_type.size * self.nfeatures * self.nfeatures

        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, nscreens),
            torch.nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        x = self.block5(x)
        x = self.block6(x)

        # pool over the spatial dimensions
        x = self.pool3(x)

        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x


class small_C16SteerableCNN(torch.nn.Module):

    def __init__(self, nscreens, in_image_res):

        super(small_C16SteerableCNN, self).__init__()

        # Compute the size of the feature map
        self.nfeatures = in_image_res

        # the model is equivariant under rotations by 22.5 degrees, modelled by C16
        self.r2_act = gspaces.rot2dOnR2(N=16)

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # This mask is to remove the 'corner' pixels in the input image, that would get outside of the square while rotating
        self.mask = nn.MaskModule(in_type, in_image_res, margin=0),

        # convolution 1
        # first specify the output type of the convolutional layer
        out_type = nn.FieldType(self.r2_act, 24 * [self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type,
                      out_type,
                      kernel_size=7,
                      stride=2,
                      padding=1,
                      bias=False), nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True))
        # **Out dim of a convolution is:
        # (in_image_res - kernel_size + 2*pad)/stride + 1
        self.nfeatures = int((self.nfeatures - 7 + 2 * 1) / 2 + 1)

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type,
                      out_type,
                      kernel_size=7,
                      stride=2,
                      padding=2,
                      bias=False), nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True))
        self.nfeatures = int((self.nfeatures - 7 + 2 * 2) / 2 + 1)

        # pool 1
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))
        # **Out dim of pooling is:
        # (in + 2*pad - (kernel-1) - 1) / stride + 1
        # where: kernel = 2*int(round(3*sigma))+1
        # and  : pad    = int((kernel-1)//2)
        k = 2 * int(round(3 * 0.66)) + 1
        p = int((k - 1) // 2)
        self.nfeatures = int((self.nfeatures + 2 * p - (k - 1) - 1) / 2 + 1)

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type,
                      out_type,
                      kernel_size=7,
                      stride=2,
                      padding=2,
                      bias=False), nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True))
        self.nfeatures = int((self.nfeatures - 7 + 2 * 2) / 2 + 1)

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96 * [self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type,
                      out_type,
                      kernel_size=7,
                      stride=2,
                      padding=2,
                      bias=False), nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True))
        self.nfeatures = int((self.nfeatures - 7 + 2 * 2) / 2 + 1)

        # pool 2
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type,
                                           sigma=0.66,
                                           stride=2,
                                           padding=0))
        self.nfeatures = int((self.nfeatures + 2 * 0 - (k - 1) - 1) / 2 + 1)

        self.gpool = nn.GroupPooling(out_type)

        # number of output channels
        c = self.gpool.out_type.size * self.nfeatures * self.nfeatures

        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, nscreens),
            torch.nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x
