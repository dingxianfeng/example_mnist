###在训练代码中，每1000轮保存一次训练好的模型，这样通过单独的测试程序，更加方便的在滑动平均模型上测试
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS=10 ##每10秒加载一次最新的模型，并在测试数据上测试最新的模型的准确率

def evaluate(mnist):
  with tf.Graph().as_default() as g:
    x=tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],neme='y-input')
    validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}


    y=mnist_inference.inference(x,None)

    ##使用前向传播的结果计算正确率，如果需要分类，使用argmax(y,1)就可以得到分类类别
    correct_prediction=tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    ##通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取均值
    ##variable_average_op
    variable_averages=tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variables_to_restore=variable_averages.variables_to_restore()
    saver=tf.train.Saver(variables_to_restore)

    while True:
      with tf.Session() as sess:
        ckpt=tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess,ckpt.model_checkpoint_path)
          global_step=ckpt.model_checkpoint_path\
            .split('/')[-1].split('-')[-1]
          accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
          print("after %s training steps,validation" "accuracy=%g"%(global_step,accuracy_score))
        else:
          print('NO checkpoint file found')
          return
      time.sleep(EVAL_INTERVAL_SECS)

 def main(argv=None):
   mnist=input_data.read_data_sets("/path/to/mnist_data",one_hot=True)
   evaluate(mnist)

 if __name__=='__main__':
   tf.app.run()



