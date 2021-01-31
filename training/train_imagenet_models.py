import numpy as np
import glob
import os
import sys
import json

import argparse

from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50, ResNet101, ResNet152

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers
from keras.initializers import glorot_uniform

from keras.utils.np_utils import to_categorical   
from keras.preprocessing.image import load_img, img_to_array


def get_args_parser():
    '''
      Command-line parsing for training parameters.
    '''
    parser = argparse.ArgumentParser('Set training parameters for classification model', add_help=False)
    
    # Model parameters
    
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--band', default=2, type=int, help="Choose from [2-4,8].")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size for training.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--learning_rate', default=0.0003, type=float)

    parser.add_argument('--model', type=str, default="inceptionv3",
                        help="Choose models from [inceptionV3, resnet50, resnet101, resnet152].")
    
    parser.add_argument('--datapath', type=str, help="Source path of BigEarth data.")
    parser.add_argument('--pretrained', action="store_true")
    
    parser.add_argument('--save_path', type=str, default=None,
                        help="Filepath for saved model location.")
    # Return to caller
    return parser

def parse_jsonLabels(file) :

    with open(file) as f:
        data = json.load(f)

    for label in data['labels'] :
        if 'irrigated' in label.split(" ") :
            return 1

    return 0

def batch_generator(args) :
    # glob all folders in datapath
    data_folders = glob.glob(os.path.join(args.datapath,"*")+"/")

    # # load bands
    # for folder in data_folders[:10] :
    #     print( glob.glob( folder+"*0"+str(args.band)+".tif"))

    while True:
        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a = data_folders, size = args.batch_size)
        images, classes = [],[]

        for folder in batch_paths :
            image = load_img((glob.glob( folder+"*0"+str(args.band)+".tif")[0]))
            clss = to_categorical( parse_jsonLabels(glob.glob( folder+"*.json")[0]),
                                    num_classes=2)

            images += [image]
            classes += [clss]

        yield (images, classes)


def create_Classifier(args) :
    '''
        - Create the pre-trained model based on InceptionV3
        - Want weights? Include: weights='imagenet')
    '''
    weights = 'imagenet' if args.pretrained else None

    if args.model == "resnet50" :
        base_model = ResNet50(weights=weights, include_top=False)
    elif args.model == "resnet101"  :
        base_model = ResNet101(weights=weights, include_top=False)
    elif args.model == "resnet152"  :
        base_model = ResNet152(weights=weights, include_top=False)
    else :
        base_model = InceptionV3(weights=weights, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(1024, activation='relu',kernel_initializer = glorot_uniform(seed=args.seed))(x)

    # and a logistic layer for our num_classes classes
    predictions = Dense(2, activation='softmax', kernel_initializer=glorot_uniform(seed=args.seed))(x)

    # prep model with new layers and compile
    model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = optimizers.Adam(lr=args.learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['mse', 'accuracy'])

    return model


def main(args):
    '''
      Our training function. Loads our dataset, requests model with appropriate parameters and begins training.
      Models get saved to save_path location.
    '''

    data_generator = batch_generator(args)
    model = create_Classifier(args)

    # for epoch in range(args.epochs) :
    #     next(data_generator)[1]

    model.fit(data_generator, steps_per_epoch=256, epochs=args.epochs)

    if args.save_path :
        model.save(args.save_path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('CNN training and evaluation script', 
    								parents=[get_args_parser()])
    args = parser.parse_args()
    
    # TODO: check parameters!
    if args.datapath :
	    main(args)
    else :
    	parser.print_help(sys.stderr)
    	sys.exit(1)