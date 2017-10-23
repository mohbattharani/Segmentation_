from model import *
from Utils import convert_colorImage
import model
from config import *


# =============================================================================
def test():

  image_filenames, label_filenames = get_filename_list(path_test)
  test_data_node = tf.placeholder(tf.float32, shape=[TEST_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

  test_labels_node = tf.placeholder(tf.int64, shape=[TEST_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
  phase_train = tf.placeholder(tf.bool, name='phase_train')
  loss, logits = model.inference(test_data_node, test_labels_node, TEST_BATCH_SIZE, phase_train)

  pred = tf.argmax(logits, axis=3)
  # get moving avg
  variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
  variables_to_restore = variable_averages.variables_to_restore()

  saver = tf.train.Saver(variables_to_restore)

  with tf.Session() as sess:
    # Load checkpoint
    try:
      print("Trying to restore last checkpoint from: ",path_ckpt)
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

    images, labels = get_all_test_data(image_filenames, label_filenames)
    i = 0
    threads = tf.train.start_queue_runners(sess=sess)
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for image_batch, label_batch  in zip(images, labels):

      feed_dict = {
        test_data_node: image_batch,
        #test_labels_node: label_batch,
        phase_train: False
      }

      dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)
      im_out = convert_colorImage(dense_prediction)
      # output_image to verify
      if (True):
          writeImage(im[0], path_output+'testing_image'+str(i)+'.png')
          scp.misc.imsave(path_output+'output_image'+str(i)+'.png', im_out)
          #writeImage(dense_prediction, 'pred_image.png')

      hist += get_hist(dense_prediction, label_batch)
      i = i+1
      print ('Batch number:', i)
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print("acc: ", acc_total)
    print("mean IU: ", np.nanmean(iu))

test()
