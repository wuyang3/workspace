import tensorflow as tf
import numpy as np

batch_size = 5
num_preprocess = 1
min_queue = 256
num_epochs =2
# Create a queue from the name strings of input images
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once('train/run_27/*.jpg'),
    num_epochs=num_epochs)

# Read an entire image file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue.
file_name, image_file = image_reader.read(filename_queue)

# Decode the image into a tensor with shape [height, width, channels]
image = tf.image.decode_jpeg(image_file)

# Cast and resize the tensor
to_float = tf.cast(image, tf.float32)
resized = tf.image.resize_images(to_float,
                                 [227, 227])
resized.set_shape((227,227,3))
shape = tf.shape(resized)

# Create mini batches using tf.train.shuffle_batch
images = tf.train.shuffle_batch([resized],
                                batch_size=batch_size,
                                num_threads=num_preprocess,
                                capacity=min_queue+3*batch_size,
                                min_after_dequeue=min_queue)
shapes = tf.shape(images)
# Start a new session and show example outputs
# Mind that initializer and queue runner need a default session or session as an arg.
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    # start the input queue threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # get an image tensor and print its value
            ss  = sess.run([shapes])
            #print(image_tensor)
            #print(f)
            #print(s)
            print(ss)
    except tf.errors.OutOfRangeError:
        print('Done queuing -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
