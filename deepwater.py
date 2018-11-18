import os

import tensorflow as tf

import configuration


FLAGS = tf.flags.FLAGS


def reduce_contrast(image):
    ratio = tf.random_uniform([], FLAGS.contrast_min, FLAGS.contrast_max, dtype=tf.float32)
    image = 128.0 + (1.0 / ratio) * (image - 128.0)
    return image


def change_brightness(image):
    offset = tf.random_uniform([], FLAGS.brightness_offset_min, FLAGS.brightness_offset_max, dtype=tf.float32)
    image += offset
    image = tf.minimum(image, 255.0)
    image = tf.maximum(image, 0.0)
    return image


def desaturate_red(image):
    shape = tf.shape(image)
    red = tf.slice(image, begin=[0, 0, 0], size=[shape[0], shape[1], 1])
    green = tf.slice(image, begin=[0, 0, 1], size=[shape[0], shape[1], 1])
    blue = tf.slice(image, begin=[0, 0, 2], size=[shape[0], shape[1], 1])
    ratio = tf.random_uniform([], FLAGS.desaturate_red_min, FLAGS.desaturate_red_max, dtype=tf.float32)
    red /= ratio
    image = tf.concat([red, green, blue], axis=2)
    return image


def augment(image):
    image = reduce_contrast(image)
    image = change_brightness(image)
    image = desaturate_red(image)
    return image


def process_tfrecords(case):
    features = tf.parse_single_example(case,
                                       features={
                                           'dim_x': tf.FixedLenFeature([], tf.int64),
                                           'dim_y': tf.FixedLenFeature([], tf.int64),
                                           'input': tf.FixedLenFeature([], tf.string),
                                           'reference': tf.FixedLenFeature([], tf.string)})

    dim_x = tf.cast(features['dim_x'], dtype=tf.int32)
    dim_y = tf.cast(features['dim_y'], dtype=tf.int32)
    shape = [dim_x, dim_y, 3]

    record_bytes = tf.decode_raw(features['input'], tf.float32)
    image = tf.reshape(record_bytes, shape)
    record_bytes = tf.decode_raw(features['reference'], tf.float32)
    reference = tf.reshape(record_bytes, shape)

    image = augment(image)

    return image, reference


def get_dataset(mode='train'):
    mode_path = os.path.join(FLAGS.database_path, mode)
    pattern = os.path.join(mode_path, '*.tfrecords')
    case_list = tf.data.Dataset.list_files(pattern, shuffle=FLAGS.shuffle)
    dataset = tf.data.TFRecordDataset(case_list)
    dataset = dataset.map(process_tfrecords, num_parallel_calls=FLAGS.num_threads)
    dataset.batch(FLAGS.batchsize)
    dataset = dataset.repeat(FLAGS.num_epochs)
    return dataset


def input_fn(dataset):
    iterator = dataset.make_one_shot_iterator()
    image, reference = iterator.get_next()
    feature = {"input": image}
    label = reference
    return feature, label


def cnn(input_image):
    # Input Layer
    input_layer = tf.expand_dims(input_image, axis=0)

    conv = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    conv = tf.layers.conv2d(inputs=conv, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    conv = tf.layers.conv2d(inputs=conv, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    conv = tf.layers.conv2d(inputs=conv, filters=3, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    return conv


def loss_fn(image, reference):
    mse = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(reference, image)), axis=2))
    return mse


def model_fn(features, labels, mode, params=None, config=None):
    input_image = features['input']

    output_image = cnn(input_image)

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'output': output_image}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = loss_fn(output_image, labels)

    # Compute evaluation metrics.
    concat = tf.concat([tf.expand_dims(input_image, axis=0), output_image, tf.expand_dims(labels, axis=0)], axis=1)
    tf.summary.image('concat', concat)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def train(enhancer):
    # Train
    train_dataset = get_dataset(mode='train')
    enhancer.train(input_fn=lambda: input_fn(train_dataset))

    # Evaluate the model.
    validation_dataset = get_dataset(mode='validation')
    eval_result = enhancer.evaluate(input_fn=lambda: input_fn(validation_dataset))

    return eval_result


def test(enhancer):

    # Test
    test_dataset = get_dataset(mode='test')
    test_result = enhancer.evaluate(input_fn=lambda: input_fn(test_dataset),
                                    checkpoint_path=None)  # TODO

    return test_result


def predict(enhancer):
    # Test
    prediction_dataset = get_dataset(mode='prediction')
    output = enhancer.predict(input_fn=lambda: input_fn(prediction_dataset),
                              checkpoint_path=None)  # TODO

    #TODO save output


def main(argv=None):
    configuration.customize_configuration()

    my_feature_columns = [tf.feature_column.numeric_column(key='input')]

    # Build model
    enhancer = tf.estimator.Estimator(
        model_fn=model_fn,
        params={'feature_columns': my_feature_columns},
        model_dir=FLAGS.output_path
    )

    if FLAGS.mode == 'train':
        train(enhancer)
    elif FLAGS.mode == 'test':
        test(enhancer)
    elif FLAGS.mode == 'predict':
        predict(enhancer)
    else:
        raise ValueError('Unrecognized mode: ' + FLAGS.mode)


if __name__ == '__main__':
    tf.app.run()
