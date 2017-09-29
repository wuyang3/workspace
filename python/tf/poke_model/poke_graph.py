import tensorflow as tf
#from poke import Poke
from poke_autoencoder import PokeAE

#train_layers = ['inverse', 'forward', 'siamese']
graph = tf.Graph()

with graph.as_default():
    #poke_model = Poke(train_layers, learning_rate=0.001, lamb=0.1)
    poke_ae = PokeAE()
    init = tf.global_variables_initializer()

sess = tf.Session(graph=graph)
sess.run(init)

writer = tf.summary.FileWriter('../logs/pokeAE/', graph=graph)
writer.close()

with graph.as_default():
    saver = tf.train.Saver()
    save_path = saver.save(sess, '../logs/pokeAE/')
