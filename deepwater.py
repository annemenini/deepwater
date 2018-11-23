import os

import tensorflow as tf

import unet.unet_nn

import configuration


FLAGS = tf.flags.FLAGS


def resize(image, reference):
    if FLAGS.resize_with_fixed_size:
        dim_x1 = FLAGS.resize_max
        dim_y1 = FLAGS.resize_max
    else:
        shape = tf.shape(image)
        dim_x1 = tf.cast(8 * tf.round(tf.random_uniform([], FLAGS.resize_min, FLAGS.resize_max) / 8), dtype=tf.int32)
        dim_y1 = tf.cast(8 * tf.round((shape[1] * dim_x1 / shape[0]) / 8), dtype=tf.int32)
    image = tf.image.resize_images(image, [dim_x1, dim_y1])
    reference = tf.image.resize_images(reference, [dim_x1, dim_y1])
    return image, reference


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


feature_spec = {'dim_x': tf.FixedLenFeature([], tf.int64),
                'dim_y': tf.FixedLenFeature([], tf.int64),
                'input': tf.FixedLenFeature([], tf.string),
                'reference': tf.FixedLenFeature([], tf.string)}


def process_tfrecords(case):
    features = tf.parse_single_example(case, features=feature_spec)

    dim_x = tf.cast(features['dim_x'], dtype=tf.int32)
    dim_y = tf.cast(features['dim_y'], dtype=tf.int32)
    shape = [dim_x, dim_y, 3]

    record_bytes = tf.decode_raw(features['input'], tf.float32)
    image = tf.reshape(record_bytes, shape)
    record_bytes = tf.decode_raw(features['reference'], tf.float32)
    reference = tf.reshape(record_bytes, shape)

    image, reference = resize(image, reference)

    image = augment(image)

    feature = {"degraded": image}
    label = reference

    return feature, label


def get_dataset(mode='train'):
    mode_path = os.path.join(FLAGS.database_path, mode)
    pattern = os.path.join(mode_path, '*.tfrecords')
    case_list = tf.data.Dataset.list_files(pattern, shuffle=FLAGS.shuffle)
    dataset = tf.data.TFRecordDataset(case_list)
    dataset = dataset.map(process_tfrecords, num_parallel_calls=FLAGS.num_threads)
    dataset.batch(FLAGS.batchsize)
    if mode == 'train':
        repeat = 1
    else:
        # TODO: Understand why the eval only appear towards the end only every 3000 subrun
        repeat = int(39 / 9)
    dataset = dataset.repeat(FLAGS.num_epochs * repeat)
    return dataset


def input_fn(dataset):
    iterator = dataset.make_one_shot_iterator()
    feature, label = iterator.get_next()
    return feature, label


def fcnn(input_image):
    """
    Fully Connected Neural Network
    Assumes an input image of shape [N, 64, 64, 3]
    """

    input_image1 = tf.image.resize_images(input_image, [64, 64])

    shape = tf.shape(input_image1)
    units0 = shape[1] * shape[2] * shape[3]  # 64 * 64 * 3 = 12288
    input_image1 = tf.reshape(input_image1, [shape[0], units0])

    layer = tf.layers.dense(inputs=input_image1, units=1536, activation=tf.nn.relu)
    layer = tf.layers.dense(inputs=layer, units=768, activation=tf.nn.relu)
    layer = tf.layers.dense(inputs=layer, units=192, activation=tf.nn.relu)
    layer = tf.layers.dense(inputs=layer, units=96, activation=tf.nn.relu)
    layer = tf.layers.dense(inputs=layer, units=24, activation=tf.nn.relu)
    layer = tf.layers.dense(inputs=layer, units=12, activation=tf.nn.relu)
    # 6 final outputs for the 3 affine transformations on the 3 channels (RGB)
    layer = tf.layers.dense(inputs=layer, units=3, activation=None)

    tf.summary.scalar('red', layer[0, 0])
    tf.summary.scalar('brightness', layer[0, 1])
    tf.summary.scalar('contrast', layer[0, 2])

    # Red saturation
    output_image_R = (layer[:, 0]) * input_image[:, :, :, 0]
    output_image_G = input_image[:, :, :, 1]
    output_image_B = input_image[:, :, :, 2]
    output_image_R = tf.expand_dims(output_image_R, axis=3)
    output_image_G = tf.expand_dims(output_image_G, axis=3)
    output_image_B = tf.expand_dims(output_image_B, axis=3)
    output_image = tf.concat([output_image_R, output_image_G, output_image_B], axis=3)

    # Brightness
    output_image = output_image - (layer[:, 1])

    # Contrast
    output_image = 0.5 + (layer[:, 2]) * (output_image - 0.5)

    return output_image


def cnn(input_image):

    conv = tf.layers.conv2d(inputs=input_image, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    conv = tf.layers.conv2d(inputs=conv, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    conv = tf.layers.conv2d(inputs=conv, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    conv = tf.layers.conv2d(inputs=conv, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    conv = tf.layers.conv2d(inputs=conv, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    conv = tf.layers.conv2d(inputs=conv, filters=3, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    return conv


def loss_fn(image, reference):
    mse = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(reference, image)), axis=2))
    return mse


def model_fn(features, labels, mode, params=None, config=None):

    # TODO: Support batch size > 1

    input_image = features['degraded']

    input_image1 = tf.divide(input_image, 255.0)
    # input_image1 = input_image * 1.0e5

    input_image1 = tf.expand_dims(input_image1, axis=0)

    output_image = fcnn(input_image1)
    # output_image = cnn(input_image1)
    # output_image, _, _ = unet.unet_nn.create_conv_net(input_image, 1, 3, 3, layers=3, features_root=16, filter_size=3, pool_size=2,
    #                 summaries=True)

    output_image = tf.multiply(output_image, 255.0)
    # input_image = input_image / 1.0e5

    output_image = tf.minimum(output_image, 255)
    output_image = tf.maximum(output_image, 0, name="final_output")

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

    optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    export_outputs={'output': output_image}
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, export_outputs=export_outputs)


def serving_input_receiver_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=None,
                                           name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    features, _ = process_tfrecords(serialized_tf_example)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def train_and_evaluate(enhancer):
    train_dataset = get_dataset(mode='train')
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_dataset))
    validation_dataset = get_dataset(mode='validation')
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(validation_dataset), steps=10)

    eval_result = tf.estimator.train_and_evaluate(enhancer, train_spec, eval_spec)

    saved_model_path = os.path.join(FLAGS.output_path, 'saved_model')
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)
    enhancer.export_savedmodel(saved_model_path, serving_input_receiver_fn)

    return eval_result


def train(enhancer):
    # Train
    train_dataset = get_dataset(mode='train')
    enhancer.train(input_fn=lambda: input_fn(train_dataset))

    # Evaluate the model.
    validation_dataset = get_dataset(mode='validation')
    eval_result = enhancer.evaluate(input_fn=lambda: input_fn(validation_dataset))

    saved_model_path = os.path.join(FLAGS.output_path, 'saved_model')
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)
    enhancer.export_savedmodel(saved_model_path, serving_input_receiver_fn)

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
        train_and_evaluate(enhancer)
    elif FLAGS.mode == 'test':
        test(enhancer)
    elif FLAGS.mode == 'predict':
        predict(enhancer)
    else:
        raise ValueError('Unrecognized mode: ' + FLAGS.mode)


if __name__ == '__main__':
    tf.app.run()
