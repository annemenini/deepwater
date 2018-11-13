import json
import os

import tensorflow as tf


flags = tf.flags

# Mode

flags.DEFINE_string(name='mode',
                    default='train',
                    help='Mode among: "train", "test" and "predict"')


# Inputs

flags.DEFINE_string(name='input_path',
                    default='input',
                    help='Location of the inputs (e.g. config files)')
flags.DEFINE_string(name='config_file',
                    default='configuration.json',
                    help='Name of the config file to be found in the input folder')

# Data base

flags.DEFINE_string(name='database_source_path',
                    default='database\\source',
                    help='Location of the source images (e.g. .png, .jpg)')
flags.DEFINE_string(name='database_path',
                    default='database\\tfrecords',
                    help='Location of the database (.tfrecords)')

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

flags.DEFINE_integer(name='batchsize',
                     default=1,
                     help='Batch size')
flags.DEFINE_integer(name='num_epochs',
                     default=16,
                     help='Number of epochs')


FLAGS = flags.FLAGS


def customize_configuration():
    input_path = FLAGS.input_path
    config_file = FLAGS.config_file
    config_path = os.path.join(input_path, config_file)

    try:
        with open(config_path, 'r') as json_file:
            data = json.load(json_file)
            for key in FLAGS.flag_values_dict():
                if key in data:
                    FLAGS.key = data[key]
                else:
                    print('Key ' + key + ' not found. Using default value ' + FLAGS.key)
    except Exception as exc:
        print('Warning, the configuration file provided is invalid:')
        print('\t' + str(exc))
        print('The CLI/default configurations will be used.')

    output_path = FLAGS.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
