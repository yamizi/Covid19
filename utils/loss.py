import keras.backend as K
import tensorflow as tf
import numpy as np
from keras import losses
from keras.layers import Lambda, Layer
import sys

from simulations import simulate_constantRt

from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score,fbeta_score

def my_auc(y_true,y_pred):
    print("auc",y_true.shape,y_pred.shape)
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 1

def keras_auc(y_true, y_pred):
    return tf.py_function(my_auc, (y_true, y_pred), tf.double)

rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))


@tf.RegisterGradient(rnd_name)
def no_grad(unused_op, grad):
    return 1, 1


def seir_func(R, index, df):
    results = []
    tf.print("inputs",R.shape,output_stream=sys.stdout)
    return R, df["R"].iloc[index] #ConfirmedDeaths

    for Rt in R:
        cases_deaths = Rt+5
        #results.append(cases_deaths)
        expanded = np.expand_dims(cases_deaths, axis=0)
        results.append(expanded)

    result =  np.concatenate(results, axis=0)
    print(result)

    return result

# Define custom loss
def seirhcd_loss(df):
    # Create a loss function that computes the country
    def loss(y_true, y_pred):
        print(y_true.shape, y_pred.shape)

        y_pred, y_true = tf.py_function(seir_func, inp=[y_pred, y_true, df], Tout=tf.float32, name='seirLoss')
        return K.mean(K.square(y_pred - y_true), axis=-1)

        with tf.Graph().as_default() as g:

            with g.gradient_override_map({"PyFunc": rnd_name}):
                y_pred, y_true = tf.py_function(seir_func, inp=[y_pred, y_true, df], Tout=tf.float32, name='seirLoss')

                tf.print(y_true,output_stream=sys.stdout)
                return K.mean(K.square(y_pred - y_true), axis=-1) #+ K.square(layer)

    # Return a function
    return loss


class LossFunction(object):
  def __init__(self, model,df):
    # Use model to obtain the inputs
    self.model = model
    self.df = df

  def __call__(self, y_true, y_pred, sample_weight=None):
    """ ignore y_true value from fit params and compute it instead using
    ext_function
    """
    toto = tf.py_function(seir_func, (y_pred,self.df), Tout=tf.float32)
    v = losses.mean_squared_error(y_true, y_pred)
    return K.mean(v)




class SEIRLayer(Layer):
    def __init__(self):
        super(SEIRLayer, self).__init__()
        self.scale = tf.Variable(1.)

    def call(self, inputs):
        # Need to generate a unique name to avoid duplicates:
        with tf.Graph().as_default() as g:

            with g.gradient_override_map({"PyFunc": self.rnd_name}):
                xout = tf.py_function(seir_func,
                                      [inputs],
                                      Tout=tf.float32,
                                      name='seirOpt')

                print("call", xout.shape)
                xout.set_shape([inputs.shape[0], inputs.shape[-1]])  # explicitly set output shape
                print("call_post", xout.shape)
                return xout

def seir_call(xin):
    xout = tf.py_function(seir_func,
                      [xin],
                      Tout=tf.float32,
                      name='seirOpt')

    #xout = K.eval(xout)  #K.stop_gradient(xout)  # explicitly set no grad
    print("call",xout.shape)
    xout.set_shape([xin.shape[0],xin.shape[-1]])  # explicitly set output shape
    print("call_post", xout.shape)
    return xout