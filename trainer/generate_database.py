import os

import imageio
import numpy as np
import tensorflow as tf

from trainer import configuration

FLAGS = tf.flags.FLAGS


def parse_database():
    source_path = FLAGS.database_source_path
    tfrecords_path = FLAGS.database_path
    if not os.path.exists(tfrecords_path):
        os.makedirs(tfrecords_path)

    expected_types = ['train', 'validation', 'test']
    supported_extensions = ['.jpeg', '.jpg', '.png', '.webp']

    case_list = []

    for set_type in os.listdir(source_path):
        type_input_path = os.path.join(source_path, set_type)
        is_dir = os.path.isdir(type_input_path)
        is_expected_type = (set_type in expected_types)
        if is_dir and is_expected_type:
            type_output_path = os.path.join(tfrecords_path, set_type)
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


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def generate_tfrecords(case_list):
    for case in case_list:
        source_path = case[0]
        tfrecords_path = case[1]
        image = imageio.imread(source_path)
        image = image.astype(np.float32)
        print('maxi:')
        print(np.max(image))

        shape = image.shape

        tf_record = tf.train.Example(features=tf.train.Features(feature={
            'dim_x': _int64_feature(shape[0]),
            'dim_y': _int64_feature(shape[1]),
            'input': _bytes_feature(image.tostring()),
            'reference': _bytes_feature(image.tostring())}))
        tf_writer = tf.python_io.TFRecordWriter(tfrecords_path)
        tf_writer.write(tf_record.SerializeToString())
        tf_writer.close()


def main(_):
    configuration.customize_configuration()
    case_list = parse_database()
    generate_tfrecords(case_list)


if __name__ == '__main__':
    tf.app.run()
