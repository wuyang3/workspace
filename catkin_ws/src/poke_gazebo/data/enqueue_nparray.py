# When 'data' is fed in through the placeholder(shape not specified 
# therefore all shape accepted), enqueue_many will enqueue 'data' based
# on the FIFOQueue size (size of each sampe) and slice data along the
# fisrt dimension. Batch operation on dequeued data will organize data
# in batch.

import numpy as np
import tensorflow as tf
import threading

tf.reset_default_graph()
data = np.array([[[1,-1],[-1,1]], [[2,-2],[-2,2]], [[3,-3],[-3,3]], [[4,-4],[-4,4]]])
num_epochs = 3
queue1_input = tf.placeholder(tf.int32)
queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.int32], shapes=[(2,2)])

def create_session():
    config = tf.ConfigProto()
    config.operation_timeout_in_ms=20000
    return tf.InteractiveSession(config=config)

enqueue_op = queue1.enqueue_many(queue1_input)
close_op = queue1.close()
dequeue_op = queue1.dequeue()
batch = tf.train.shuffle_batch([dequeue_op], batch_size=2, capacity=5, min_after_dequeue=4)

sess = create_session()

# enqueue for num_epochs time and dequeue finishes after dequeuing all.
def fill_queue():
    for i in range(num_epochs):
        sess.run(enqueue_op, feed_dict={queue1_input: data})
    sess.run(close_op)

fill_thread = threading.Thread(target=fill_queue, args=())
fill_thread.start()

# read the data from queue shuffled
tf.train.start_queue_runners()
try:
    while True:
        print(batch.eval())
except tf.errors.OutOfRangeError:
    print("Done")
