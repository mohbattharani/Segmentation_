import os
import numpy as np
from datetime import datetime
import time

from Utils import _add_loss_summaries
from model import *

#from augmentation import pre_process_image

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 367
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1
TEST_ITER = 200 # ceil(NUM_EXAMPLES_PER_EPOCH_FOR_TEST / TRAIN_BATCH_SIZE)

# =========== This function converts prediction to image ===========================
def color_image(image, num_classes=11):
    import matplotlib as mpl
    import matplotlib.cm
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

def train(total_loss, global_step):

    """ fix lr """
    lr = INITIAL_LEARNING_RATE
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.AdamOptimizer(lr)
      grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

    return train_op


def training():
  # should be changed if your model stored by different convention
  startstep = 801 #if not is_finetune else int(FLAGS.finetune.split('-')[-1])

  image_filenames, label_filenames = get_filename_list(path_train)
  val_image_filenames, val_label_filenames = get_filename_list(path_val)

  with tf.Graph().as_default():

    train_data_node = tf.placeholder( tf.float32, shape=[TRAIN_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
    train_labels_node = tf.placeholder(tf.int64, shape=[TRAIN_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    global_step = tf.Variable(0, trainable=False)

    # For CamVid
    images, labels = CamVidInputs(image_filenames, label_filenames, TRAIN_BATCH_SIZE)
    print ('Camvid:', images, '===000===', labels)
    val_images, val_labels = CamVidInputs(val_image_filenames, val_label_filenames, TRAIN_BATCH_SIZE)

    # Build a Graph that computes the logits predictions from the inference model.
    loss, eval_prediction = inference(train_data_node, train_labels_node, TRAIN_BATCH_SIZE, phase_train)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_op = train(loss, global_step)
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
      # Build an initialization operation to run below.
      try:
         print("Trying to restore last checkpoint from ", path_ckpt, " ...")
    # Use TensorFlow to find the latest checkpoint - if any.
         last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=path_ckpt)
         print ('last chkr point:', last_chk_path)
    # Try and load the data in the checkpoint.
         saver.restore(sess, save_path=last_chk_path)
    # If we get to this point, the checkpoint was successfully loaded.
         print("Restored checkpoint from:", last_chk_path)
      except:
    # If the above failed for some reason, simply
    # initialize all the variables for the TensorFlow graph.
         print("Failed to restore checkpoint. Initializing variables instead.")
         sess.run(tf.global_variables_initializer())

      # Start the queue runners.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      # Summery placeholders
      summary_writer = tf.summary.FileWriter(path_train, sess.graph)
      average_pl = tf.placeholder(tf.float32)
      acc_pl = tf.placeholder(tf.float32)
      iu_pl = tf.placeholder(tf.float32)
      average_summary = tf.summary.scalar("test_average_loss", average_pl)
      acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
      iu_summary = tf.summary.scalar("Mean_IU", iu_pl)

      for step in range(train_iteration):
        image_batch ,label_batch = sess.run([images, labels])
        # since we still use mini-batches in validation, still set bn-layer phase_train = True
        #print ('Batch:', image_batch, '  ----0000---', label_batch)
        #image_batch_a = pre_process_image (image_batch, True)
        feed_dict = {
          train_data_node: image_batch,
          train_labels_node: label_batch,
          phase_train: True
        }
        start_time = time.time()
        #print ('Step:', step)
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if (step<50):
           print ('Step:',step)

        if step % 100 == 0:
          num_examples_per_step = TRAIN_BATCH_SIZE
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))

          # eval current training batch pre-class accuracy
          pred = sess.run(eval_prediction, feed_dict=feed_dict)
          per_class_acc(pred, label_batch)

        if step % val_iter == 0:
          print("start validating.....")
          total_val_loss = 0.0
          hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
          for test_step in range(TEST_ITER):
            val_images_batch, val_labels_batch = sess.run([val_images, val_labels])

            _val_loss, _val_pred = sess.run([loss, eval_prediction], feed_dict={
              train_data_node: val_images_batch,
              train_labels_node: val_labels_batch,
              phase_train: True
            })
            total_val_loss += _val_loss
            hist += get_hist(_val_pred, val_labels_batch)
          print("val loss: ", total_val_loss / TEST_ITER)
          acc_total = np.diag(hist).sum() / hist.sum()
          iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
          test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / TEST_ITER})
          acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
          iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
          print_hist_summery(hist)
          print(" end validating.... ")

          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)
          summary_writer.add_summary(test_summary_str, step)
          summary_writer.add_summary(acc_summary_str, step)
          summary_writer.add_summary(iu_summary_str, step)
        # Save the model checkpoint periodically.
        if step % save_model_itr == 0 or (step + 1) == train_iteration:
          checkpoint_path = os.path.join(path_ckpt, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=global_step)

      coord.request_stop()
      coord.join(threads)

# --------------------------------------------------------

training()
