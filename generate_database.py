import json
import os

import scipy
import tensorflow as tf


flags = tf.flags

# Inputs

flags.DEFINE_string(name='input_path',
                    default='input',
                    help='Location of the inputs (e.g. config files)')
flags.DEFINE_string(name='config_file',
                    default='configuration.json',
                    help='Name of the config file to be found in the input folder')

# Data base

flags.DEFINE_string(name='database_source_path',
                    default='source',
                    help='Location of the source images (e.g. .png, .jpg)')
flags.DEFINE_string(name='database_tfrecords_path',
                    default='tfrecords',
                    help='Location where to store the generated .tfrecords')


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
    except:
        print('Warning, the configuration file provided is invalid: ' + config_path)
        print(ValueError)
        print('The CLI/default configurations will be used.')


def parse_database():
    source_path = FLAGS.database_source_path
    tfrecords_path = FLAGS.database_tfrecords_path

    expected_types = ['train', 'validation', 'test']
    supported_extensions = ['.jpeg', '.jpg', '.png']

    case_list = []

    for type in os.listdir(source_path):
        type_input_path = os.path.join(source_path, type)
        is_dir = os.path.isdir(type_input_path)
        is_expected_type = (type in expected_types)
        if is_dir and is_expected_type:
            type_output_path = os.path.join(tfrecords_path, type)
            if not os.path.exists(type_output_path):
                os.makedirs(type_output_path)

            for elem in os.listdir(type_input_path):
                filename, extension = os.path.splitext(elem)
                if extension in supported_extensions:
                    elem_input_path = os.path.join(type_input_path, elem)
                    elem_output_path = os.path.join(type_output_path, filename + '.tfrecords')
                    case = (elem_input_path, elem_output_path)
                    case_list.append(case)

    return case_list


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def generate_tfrecords(case_list):
    for case in case_list:
        source_path = case[0]
        tfrecords_path = case[1]
        image = scipy.misc.imread(source_path)

        tf_record = tf.train.Example(features=tf.train.Features(feature={'reference': _bytes_feature(image.tostring())}
                                                                ))
        tf_writer = tf.python_io.TFRecordWriter(tfrecords_path)
        tf_writer.write(tf_record.SerializeToString())
        tf_writer.close()


def main(argv=None):
    customize_configuration()
    case_list = parse_database()
    generate_tfrecords(case_list)


if __name__ == '__main__':
    tf.app.run()
