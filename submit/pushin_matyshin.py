import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception


class PushinMatyshin:
    def __init__(self, checkpoint_path, batch_size):
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

        self.graph = tf.Graph()
        self.image_shape = (self.batch_size, 299, 299, 3)
        self.num_classes = 1001

        with self.graph.as_default():
            self.real_image = tf.placeholder(tf.float32, self.image_shape, name='real_image')
            self.max_perturbation = tf.placeholder(tf.float32, (), name='max_perturbation')

            self.fake_image_subst = tf.Variable(
                np.zeros(self.image_shape, dtype=np.float32), name='fake_image_subst'
            )

            self.min_real_image = tf.maximum(0., self.real_image - self.max_perturbation)
            self.max_real_image = tf.minimum(1., self.real_image + self.max_perturbation)

            self.fake_image = self.min_real_image + \
                              (self.max_real_image - self.min_real_image) * (tf.tanh(self.fake_image_subst) + 1) / 2

            with slim.arg_scope(inception.inception_v3_arg_scope()):
                self.fake_logits, self.inception_end_points = inception.inception_v3(
                    self.fake_image, num_classes=self.num_classes, is_training=False,
                )

            self.softmaxed_fake = tf.nn.softmax(self.fake_logits)
            self.top_classes = tf.nn.top_k(self.softmaxed_fake, 3)

            self.assign_fake_image_subs_by_real = self.fake_image_subst.assign(
                tf.atanh(2 * (self.real_image - self.min_real_image) / (self.max_real_image - self.min_real_image) - 1)
            )

            self.target_probs = tf.placeholder(
                tf.float32, [self.batch_size, self.num_classes], name='target_probs'
            )

            self.alpha = tf.placeholder(tf.float32, (), name='alpha')
            self.learning_rate = tf.placeholder(tf.float32, (), name='learning_rate')

            #             self.assign_fake_image_subs_by_clipped_fake = self.fake_image_subst.assign(
            #                 tf.atanh(
            #                     2 * tf.clip_by_value(
            #                         self.fake_image, self.min_real_image, self.max_real_image
            #                     ) - 1
            #                 )
            #             )

            self.main_loss = tf.losses.softmax_cross_entropy(
                self.target_probs,
                self.fake_logits,
                label_smoothing=0.1,
                weights=1.0
            )

            #             self.abs_img_diff = tf.abs(self.fake_image - self.real_image)
            #             self.clipped_img_diff = tf.where(
            #                 self.abs_img_diff > self.max_perturbation,
            #                 self.abs_img_diff, tf.zeros(self.image_shape)
            #             )
            #             self.clipped_diff_sum = tf.reduce_sum(self.clipped_img_diff)
            #             self.reg_loss = self.alpha * self.clipped_diff_sum

            self.loss = self.main_loss  # + self.reg_loss

            start_vars = set(x.name for x in tf.global_variables())
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

            self.train = self.optimizer.minimize(self.loss, var_list=[self.fake_image_subst])

            end_vars = tf.global_variables()
            new_vars = [x for x in end_vars if x.name not in start_vars]
            self.init = tf.variables_initializer(var_list=[self.fake_image_subst] + new_vars)

            self.sess = tf.Session()
            saver = tf.train.Saver(slim.get_model_variables())
            saver.restore(self.sess, self.checkpoint_path)

    def inference(self, images, targets, max_perturbation, alpha=1, start_lr=0.1, end_lr=0.01, n=10):
        def clip_fake(real_image, max_perturbation):
            self.sess.run(
                [self.assign_fake_image_subs_by_clipped_fake], feed_dict={
                    self.max_perturbation: max_perturbation,
                    self.real_image: real_image,
                }
            )

        target_probs = np.zeros((len(targets), self.num_classes))
        max_perturbation /= 255
        images = np.array(images) / 255

        learning_rate = start_lr

        for number, target in enumerate(targets):
            target_probs[number][target] = 1

        with self.graph.as_default():
            self.sess.run(self.init)
            result_images = self.sess.run(
                [self.assign_fake_image_subs_by_real], feed_dict={
                    self.real_image: images,
                    self.max_perturbation: max_perturbation,
                }
            )

            for i in range(n):
                # clip_fake(images, max_perturbation)
                result_images, loss, softmaxed, fake_logits, top_classes, _ = self.sess.run(
                    [self.fake_image, self.loss, self.softmaxed_fake, self.fake_logits,
                     self.top_classes, self.train],
                    feed_dict={
                        self.real_image: images,
                        self.target_probs: target_probs,
                        self.max_perturbation: max_perturbation,
                        self.alpha: alpha,
                        self.learning_rate: learning_rate,
                    }
                )
                # print(
                #     'Step: {step} | Loss: {loss} | Learning rate: {learning_rate}'.format(
                #         step=i,
                #         loss=loss,
                #         learning_rate=learning_rate,
                #     )
                # )
                # print(top_classes)
                # learning_rate *= (end_lr / start_lr) ** (1 / n)

            # clip_fake(images, max_perturbation)
            return np.round(result_images * 255).astype(np.uint8)  # , loss
