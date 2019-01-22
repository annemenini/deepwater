import json
import os

import tensorflow as tf


flags = tf.flags

# Mode

flags.DEFINE_string(name='mode',
                    default='train',
                    help='Mode among: "train", "test" and "predict"')


# Inputs


flags.DEFINE_bool(name='use_tfrecords',
                  default=True,
                  help='If true, use tfrecords inputs, if false, use images')
flags.DEFINE_string(name='input_path',
                    default='input',
                    help='Location of the inputs (e.g. config files)')
flags.DEFINE_string(name='config_file',
                    default='configuration.json',
                    help='Name of the config file to be found in the input folder')

# Data base

flags.DEFINE_string(name='database_source_path',
                    default='database\\source',
                    help='Location of the source images (e.g. .png, .jpg) to create the tfrecords')
flags.DEFINE_string(name='database_path',
                    default='\\Users\\stere\\Documents\\deepwater\\database\\tfrecords',
                    help='Location of the database (.tfrecords or .jpeg, .jpg, .png, .webp)')

# Outputs

flags.DEFINE_string(name='output_path',
                    default='output',
                    help='Location of the outputs (e.g. models, logs)')

# Database loading

flags.DEFINE_bool(name='shuffle',
                  default=True,
                  help='Shuffle the database loading order')
flags.DEFINE_integer(name='num_threads',
                     default=1,
                     help='Number of threads used to load the data')

# Training hyper-parameters

flags.DEFINE_integer(name='batch_size',
                     default=1,
                     help='Batch size')
flags.DEFINE_integer(name='num_epochs',
                     default=1024,
                     help='Number of epochs')
flags.DEFINE_float(name='learning_rate',
                   default=0.0001,
                   help='Learning rate')


# Data augmentation

flags.DEFINE_float(name='brightness_offset_min',
                   default=-16,
                   help='Minimum value for brightness offset')
flags.DEFINE_float(name='brightness_offset_max',
                   default=16,
                   help='Maximum value for brightness offset')

flags.DEFINE_float(name='desaturate_red_min',
                   default=1,
                   help='Minimum value to divide red channel')
flags.DEFINE_float(name='desaturate_red_max',
                   default=4,
                   help='Maximum value to divide red channel')

flags.DEFINE_float(name='contrast_min',
                   default=1,
                   help='Minimum value of contrast reduction')
flags.DEFINE_float(name='contrast_max',
                   default=2.2,
                   help='Maximum value of contrast reduction')

flags.DEFINE_bool(name='resize',
                  default=True,
                  help='Resize images before processing')
flags.DEFINE_bool(name='resize_with_fixed_size',
                  default=False,
                  help='Use a fixed resizing size (of size resize_max)')
flags.DEFINE_integer(name='resize_min',
                     default=256,
                     help='Minimum value of contrast reduction')
flags.DEFINE_integer(name='resize_max',
                     default=1024,
                     help='Maximum value of contrast reduction')

FLAGS = flags.FLAGS


def customize_configuration():
    input_path = FLAGS.input_path
    config_file = FLAGS.config_file
    config_path = os.path.join(input_path, config_file)

    if not os.path.exists(config_path):
        print('Warning, no configuration file found.')
        print('The CLI/default configurations will be used.')
    else:
        with open(config_path, 'r') as json_file:
            data = json.load(json_file)
            for key in FLAGS.flag_values_dict():
                if key in data:
                    FLAGS.key = data[key]
                else:
                    print('Key ' + key + ' not found. Using default value ' + FLAGS.key)
