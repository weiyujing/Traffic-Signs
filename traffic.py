import os
import skimage.data
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from skimage import transform

def load_data(data_directory):
    # 如果在 data_directory 中发现了一些东西，就双重检查这是否是一个目录；如果是，就将其加入到你的列表中。
    # 注意：每个子目录都代表了一个标签。
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))                  #每幅图对应的标签

    return images, labels


def struct_graph():
    # Initialize placeholders
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
    y = tf.placeholder(dtype=tf.int32, shape=[None])

    # Flatten the input data
    images_flat = tf.contrib.layers.flatten(x)

    # Fully connected layer
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

    # Define a loss function
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                         logits=logits))

    # Define an optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # Convert logits to label indexes
    correct_pred = tf.argmax(logits, 1)

    # Define an accuracy metric
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return x,y,train_op,loss,correct_pred



def train_model():
    tf.set_random_seed(1234)
    x, y, train_op, loss,correct_pred=struct_graph()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    m_saver = tf.train.Saver()
    for i in range(6000):
         _, loss_value = sess.run([train_op, loss], feed_dict={x: images28, y: labels})
         if i % 10 == 0:
             print("Loss: ", loss_value)
             m_saver.save(sess, './traffic-model/mnist_slp', global_step=i)

    sample_indexes = random.sample(range(len(images28)), 10)
    sample_images = [images28[i] for i in sample_indexes]
    sample_labels = [labels[i] for i in sample_indexes]

        # Run the "correct_pred" operation
    predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
        # Print the real and predicted labelsprint(sample_labels)
    print(predicted)

        # Display the predictions and the ground truth
    plt.figure(figsize=(10, 10))
    for i in range(len(sample_images)):
            truth = sample_labels[i]
            prediction = predicted[i]
            plt.subplot(5, 2, 1 + i)
            plt.axis('off')
            color = 'green' if truth == prediction else 'red'
            plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
                     fontsize=12, color=color)
            plt.imshow(sample_images[i], cmap="gray")

    plt.show()


def test_model():
    x, y, train_op, loss, correct_pred = struct_graph()
    # Load the test data
    test_images, test_labels = load_data(test_data_directory)
    sess = tf.Session()
    m_saver = tf.train.Saver()
    # load the model

    m_saver.restore(sess, tf.train.latest_checkpoint("./traffic-model/"))

    # Transform the images to 28 by 28 pixels
    test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

    # Convert to grayscale
    from skimage.color import rgb2gray
    test_images28 = rgb2gray(np.array(test_images28))

    # Run predictions against the full test set.
    predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

    # Calculate correct matches
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

    # Calculate the accuracy
    accuracy = match_count / len(test_labels)

    # Print the accuracy
    print("Accuracy: {:.3f}".format(accuracy))





ROOT_PATH = "D:/PyCharm 2018.2.1/workplace/traffic"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")
images, labels = load_data(train_data_directory)
'''
labels=np.array(labels)
print(labels.size,labels[100],len(set(labels)))
plt.hist(labels,62)
plt.show()
'''
'''
# Determine the (random) indexes of the images that you want to see
traffic_signs = [300, 2250, 3650, 4000]  #随机查看交通图

# Fill out the subplots with the random images that you defined
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape,
                                                  images[traffic_signs[i]].min(),
                                                  images[traffic_signs[i]].max()))
plt.show()
'''
# You pick the first image for each label每一类标签输出第一张图
unique_labels = set(labels)
# Set a counter
i = 1
# For each unique label
plt.figure(figsize=(20, 20))
for label in unique_labels:
    image = images[labels.index(label)]

    # Define 64 subplots
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image
    plt.imshow(image)
plt.show()

# Import the `transform` module from `skimage`
from skimage import transform

# Rescale the images in the `images` array
images28 = [transform.resize(image, (28, 28)) for image in images]

images28 = np.array(images28)
# Convert `images28` to grayscale
images28 = rgb2gray(images28)



def main():
  #train_model()
  test_model()

if __name__ == '__main__':
    main()
