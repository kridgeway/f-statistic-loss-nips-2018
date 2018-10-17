import numpy as np
import tensorflow as tf
import os
import numpy as np
import embedding as emb
import sys
import h5py

class DMLModel(emb.EmbeddingModel):
    def load_dataset(self,params):
        raise ''
    def recognition_net(self):
        height = self.get_output_size()[1]
        part_height = int(height/3)
        top = self.x_rescaled[:,:part_height,:,:]
        mid = self.x_rescaled[:,part_height:part_height*2,:,:]
        bot = self.x_rescaled[:,part_height*2:,:,:]
        #Shared weights for the first layer
        self.c1_w = self.wrecog([7,7,3,64], 'c1_w')
        self.c1_b = self.brecog([64], 'c1_b')

        c1_top_output = tf.nn.relu(tf.nn.conv2d(top, self.c1_w, strides=[1,1,1,1], padding='SAME') + self.c1_b)
        c1_mid_output = tf.nn.relu(tf.nn.conv2d(mid, self.c1_w, strides=[1,1,1,1], padding='SAME') + self.c1_b)
        c1_bot_output = tf.nn.relu(tf.nn.conv2d(bot, self.c1_w, strides=[1,1,1,1], padding='SAME') + self.c1_b)

        s2_top_output = tf.nn.max_pool( tf.nn.lrn(c1_top_output), ksize=(1,7,7,1), strides=(1,2,2,1), padding='SAME')
        s2_mid_output = tf.nn.max_pool( tf.nn.lrn(c1_mid_output), ksize=(1,7,7,1), strides=(1,2,2,1), padding='SAME')
        s2_bot_output = tf.nn.max_pool( tf.nn.lrn(c1_bot_output), ksize=(1,7,7,1), strides=(1,2,2,1), padding='SAME')

        s4_outputs=[]
        for s2_output, name in [ (s2_top_output, 'top'), (s2_mid_output,'mid'), (s2_bot_output,'bot')]:
            c3_w = self.wrecog([5,5,64,64], 'c3_w_%s' % name)
            c3_b = self.brecog([64], 'c3_b_%s' % name)
            c3_output = tf.nn.relu(tf.nn.conv2d(s2_output, c3_w, strides=[1,1,1,1], padding='SAME') + c3_b)
            s4_output = tf.nn.max_pool(tf.nn.lrn(c3_output), ksize=(1,5,5,1), strides=(1,2,2,1), padding='SAME')
            s4_output_size = np.prod(s4_output.get_shape().as_list()[1:])
            #print s4_output, s4_output_size
            s4_out_flat = tf.reshape(s4_output,[-1, s4_output_size])
            s4_outputs.append(s4_out_flat)
        concatenated_outputs = tf.concat(s4_outputs, axis=1)
        f5_w = self.wrecog([concatenated_outputs.get_shape().as_list()[1], self.n_z], 'f5_w')
        f5_b = self.brecog([self.n_z], 'f5_b')
        return tf.matmul(concatenated_outputs, f5_w) + f5_b
