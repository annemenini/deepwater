import os

import imageio
import numpy as np
import tensorflow as tf

from trainer import configuration

FLAGS = tf.flags.FLAGS


def resize(image, reference):
    if not FLAGS.resize:
        shape = tf.shape(image)
        dim_x1 = shape[1]
        dim_y1 = shape[2]
    elif FLAGS.resize_with_fixed_size:
        dim_x1 = FLAGS.resize_max
        dim_y1 = FLAGS.resize_max
    else:
        shape = tf.shape(image)
        dim_x1 = tf.cast(8 * tf.round(tf.random_uniform([], FLAGS.resize_min, FLAGS.resize_max) / 8), dtype=tf.int32)
        dim_y1 = tf.cast(8 * tf.round((shape[2] * dim_x1 / shape[1]) / 8), dtype=tf.int32)
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
    red = tf.slice(image, begin=[0, 0, 0, 0], size=[shape[0], shape[1], shape[2], 1])
    green = tf.slice(image, begin=[0, 0, 0, 1], size=[shape[0], shape[1], shape[2], 1])
    blue = tf.slice(image, begin=[0, 0, 0, 2], size=[shape[0], shape[1], shape[2], 1])
    ratio = tf.random_uniform([], FLAGS.desaturate_red_min, FLAGS.desaturate_red_max, dtype=tf.float32)
    red /= ratio
    image = tf.concat([red, green, blue], axis=3)
    return image


def augment(image):
    # TODO: should the data augmentation be different for each element of the batch? It is right the same now.
    image = reduce_contrast(image)
    image = change_brightness(image)
    image = desaturate_red(image)
    return image


feature_spec = {'dim_x': tf.FixedLenFeature([], tf.int64),
                'dim_y': tf.FixedLenFeature([], tf.int64),
                'input': tf.FixedLenFeature([], tf.string),
                'reference': tf.FixedLenFeature([], tf.string)}


def process_tfrecords(case, mode='train'):
    features = tf.parse_single_example(case, features=feature_spec)
    dim_x = tf.cast(features['dim_x'], dtype=tf.int32)
    dim_y = tf.cast(features['dim_y'], dtype=tf.int32)

    # case = tf.expand_dims(case, axis=0)
    # features = tf.parse_example(case, features=feature_spec)
    #
    # # The different elements of the batch must have consistent size
    # dim_x = tf.cast(features['dim_x'], dtype=tf.int32)
    # dim_x = tf.reshape(dim_x, [-1])[0]
    # dim_y = tf.cast(features['dim_y'], dtype=tf.int32)
    # dim_y = tf.reshape(dim_y, [-1])[0]

    shape = [-1, dim_x, dim_y, 3]
    record_bytes = tf.decode_raw(features['input'], tf.float32)
    image = tf.reshape(record_bytes, shape)
    record_bytes = tf.decode_raw(features['reference'], tf.float32)
    reference = tf.reshape(record_bytes, shape)

    image, reference = resize(image, reference)

    if not mode == 'predict':
        image = augment(image)

    feature = {"degraded": image}
    label = reference

    return feature, label


def process_images(case, mode):
    image_string = tf.read_file(case)
    image = tf.image.decode_jpeg(image_string, channels=3)
    shape = tf.shape(image)
    image = tf.reshape(image, [-1, shape[0], shape[1], shape[2]])

    reference = image
    image, reference = resize(image, reference)
    if not mode == 'predict':
        image = augment(image)
    feature = {"degraded": image}
    label = reference
    return feature, label


def process_placeholder(placeholder):
    feature = {"degraded": placeholder}
    return feature


def get_dataset(mode='train'):
    mode_path = os.path.join(FLAGS.database_path, mode)
    if FLAGS.use_tfrecords:
        pattern = os.path.join(mode_path, '*.tfrecords')
        case_list = tf.data.Dataset.list_files(pattern, shuffle=FLAGS.shuffle)
        dataset = tf.data.TFRecordDataset(case_list)
        dataset = dataset.map(lambda case: process_tfrecords(case, mode), num_parallel_calls=FLAGS.num_threads)
    else:
        pattern = os.path.join(mode_path, '*.*')
        dataset = tf.data.Dataset.list_files(pattern, shuffle=FLAGS.shuffle)
        dataset = dataset.map(lambda case: process_images(case, mode), num_parallel_calls=FLAGS.num_threads)
    dataset.batch(FLAGS.batch_size)
    if mode == 'train':
        repeat = 1
    else:
        # TODO: Understand why the eval only appears towards the end only every 3000 subrun
        repeat = int(39 / 9)
    if FLAGS.mode == 'train':
        dataset = dataset.repeat(FLAGS.num_epochs * repeat)
    else:
        dataset = dataset.repeat(1)
    return dataset


def input_fn(dataset):
    iterator = dataset.make_one_shot_iterator()
    feature, label = iterator.get_next()
    return feature, label


def fcnn(input_image):
    """Fully Connected Neural Network

    Assumes an input image of shape [N, 64, 64, 3]
    """

    input_image1 = tf.image.resize_images(input_image, [64, 64])

    input_image1 = tf.reshape(input_image1, [-1, 64 * 64 * 3])

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
    output_image_r = (1 + layer[:, 0]) * input_image[:, :, :, 0]
    output_image_g = input_image[:, :, :, 1]
    output_image_b = input_image[:, :, :, 2]
    output_image_r = tf.expand_dims(output_image_r, axis=3)
    output_image_g = tf.expand_dims(output_image_g, axis=3)
    output_image_b = tf.expand_dims(output_image_b, axis=3)
    output_image = tf.concat([output_image_r, output_image_g, output_image_b], axis=3)

    # Brightness
    output_image = output_image - layer[:, 1]

    # Contrast
    output_image = 0.5 + (1 + layer[:, 2]) * (output_image - 0.5)

    return output_image


def cnn(input_image):

    layer = tf.layers.conv2d(inputs=input_image, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    layer = tf.layers.conv2d(inputs=layer, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    layer = tf.layers.conv2d(inputs=layer, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    layer = tf.layers.conv2d(inputs=layer, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    layer = tf.layers.conv2d(inputs=layer, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    layer = tf.layers.conv2d(inputs=layer, filters=3, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    return layer


def loss_fn(image, reference):
    mse = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(reference, image)), axis=2))
    return mse


def model_fn(features, labels, mode, params=None, config=None):

    input_image = features['degraded']

    input_image1 = tf.divide(input_image, 255.0)

    output_image = fcnn(input_image1)
    # output_image = cnn(input_image1)
    # output_image, _, _ = unet.unet_nn.create_conv_net(input_image, 1, 3, 3, layers=3, features_root=16, filter_size=3,
    # pool_size=2,
    #                 summaries=True)

    output_image = tf.multiply(output_image, 255.0)

    output_image = tf.minimum(output_image, 255)
    output_image = tf.maximum(output_image, 0, name="final_output")

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'output': output_image}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = loss_fn(output_image, labels)

    # Compute evaluation metrics.
    concat = tf.concat([input_image, output_image, labels], axis=1)
    tf.summary.image('concat', concat)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    export_outputs = {'output': output_image}
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, export_outputs=export_outputs)


def serving_input_receiver_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=None,
                                           name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    if FLAGS.use_tfrecords:
        features, _ = process_tfrecords(serialized_tf_example, 'predict')
    else:
        features, _ = process_images(serialized_tf_example, 'predict')
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
    # enhancer.export_savedmodel(saved_model_path, serving_input_receiver_fn)
    enhancer.export_saved_model(saved_model_path, tf.estimator.export.build_raw_serving_input_receiver_fn(
        features={'degraded': tf.placeholder(dtype=tf.float32,
                                             shape=[None, None, None, 3],
                                             name='degraded')}))

    return eval_result


def train(enhancer):
    # Train
    train_dataset = get_dataset(mode='train')
    enhancer.train(input_fn=lambda: input_fn(train_dataset))

    # Evaluate the model on validation dataset
    validation_dataset = get_dataset(mode='validation')
    eval_result = enhancer.evaluate(input_fn=lambda: input_fn(validation_dataset))

    enhancer.export_saved_model(FLAGS.output_path, serving_input_receiver_fn)

    return eval_result


def test(enhancer):

    # Test, i.e. evaluate model on test dataset
    test_dataset = get_dataset(mode='test')
    test_result = enhancer.evaluate(input_fn=lambda: input_fn(test_dataset))

    return test_result


def save_prediction(predictions):
    prediction_path = os.path.join(FLAGS.output_path, 'predict')
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)
    # TODO: Change so that the prediction are done progressively and not stored together
    outputs = [p["output"] for p in predictions]
    for index, output in enumerate(outputs):
        # TODO: Not clear why the batch dimension disappear in predict mode??
        item_path = os.path.join(prediction_path, str(index) + '.png')
        imageio.imwrite(item_path, np.squeeze(output).astype(np.uint8))


def predict(enhancer):
    # Predict
    prediction_dataset = get_dataset(mode='predict')
    predictions = enhancer.predict(input_fn=lambda: input_fn(prediction_dataset))
    save_prediction(predictions)


def main(_):
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
