import csv
import os

import tensorflow as tf
from skimage import io

from pushin_matyshin import PushinMatyshin

tf.flags.DEFINE_string(
    'input_dir', 'data/input_dir', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', 'data/output_dir', 'Output directory with images.')

tf.flags.DEFINE_integer(
    'max_epsilon', 15, 'How strong app can change the image')

tf.flags.DEFINE_integer(
    'batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_string(
    'checkpoint_path', 'weights/adv_inception_v3.ckpt', 'Path to checkpoint for inception network.')

FLAGS = tf.flags.FLAGS


def get_batches(folder_path, batch_size):
    images_batch = []
    targets_batch = []
    names_batch = []

    with open(os.path.join(folder_path, 'target_class.csv'), 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 0:
                break

            file_name, target = row[0], int(row[1])

            file_path = os.path.join(folder_path, file_name)

            names_batch.append(file_name)
            images_batch.append(io.imread(file_path))
            targets_batch.append(target)

            if len(images_batch) == batch_size:
                yield names_batch, images_batch, targets_batch, batch_size
                names_batch, images_batch, targets_batch = [], [], []

        if len(images_batch) > 0:
            last_name = names_batch[-1]
            last_image = images_batch[-1]
            last_target = targets_batch[-1]

            enter_len = len(names_batch)
            extra_len = batch_size - enter_len

            names_batch.extend([last_name] * extra_len)
            images_batch.extend([last_image] * extra_len)
            targets_batch.extend([last_target] * extra_len)

            yield names_batch, images_batch, targets_batch, enter_len


def save_images(folder_path, names, images, real_len):
    for n, (image, name) in enumerate(zip(images, names)):
        if n == real_len:
            break
        img_path = os.path.join(folder_path, name)
        io.imsave(img_path, image)


def main(_):
    pm = PushinMatyshin(FLAGS.checkpoint_path, FLAGS.batch_size)

    for names_batch, images_batch, targets_batch, real_len in get_batches(FLAGS.input_dir, FLAGS.batch_size):
        images = pm.inference(
            images_batch, targets_batch,
            max_perturbation=FLAGS.max_epsilon, alpha=1, start_lr=0.05, end_lr=0.001, n=10
        )
        save_images(FLAGS.output_dir, names_batch, images, real_len)


if __name__ == '__main__':
    tf.app.run()
