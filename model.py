import tensorflow as tf
class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.x_input = tf.keras.layers.Input(shape=(784,), dtype=tf.float32)
    self.y_input = tf.keras.layers.Input(shape=(), dtype=tf.int64)
    x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])
    # first convolutional layer
    self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')
    self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
    # second convolutional layer
    self.conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')
    self.pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
    # first fully connected layer
    self.fc1 = tf.keras.layers.Dense(1024, activation='relu')
    # output layer
    self.fc2 = tf.keras.layers.Dense(10)
  def call(self, inputs):
    x_image = tf.reshape(inputs, [-1, 28, 28, 1])
    h_conv1 = self.conv1(x_image)
    h_pool1 = self.pool1(h_conv1)
    h_conv2 = self.conv2(h_pool1)
    h_pool2 = self.pool2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = self.fc1(h_pool2_flat)
    pre_softmax = self.fc2(h_fc1)
    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=self.y_input, logits=pre_softmax)
    xent = tf.reduce_sum(y_xent)
    y_pred = tf.argmax(pre_softmax, 1)
    correct_prediction = tf.equal(y_pred, self.y_input)
    num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return pre_softmax, xent, y_pred, num_correct, accuracy
