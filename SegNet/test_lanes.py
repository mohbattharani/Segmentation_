import tensorflow as tf
import os
import scipy as scp
import model
from line_detect_cv import *
from Utils import get_road_pixels
from config import *

def read_images_from_folder(folder_path):
    return os.listdir(folder_path)

def test():
  test_data_node = tf.placeholder(tf.float32,
        shape=[TEST_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
  test_labels_node = tf.placeholder(tf.int64, shape=[TEST_BATCH_SIZE, 360, 480, 1])
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

    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    #im_filename = '/home/mohbat/RoadSegmentation/road lane Seg/SegNet11/road/Seq05VD_f00510.png'
    #im_filename = ['/home/mohbat/RoadSegmentation/road lane Seg/SegNet11/road/im1.png']
    path = '/home/mohbat/RoadSegmentation/DataSet/CamVid/test/'
    #path =''
    im_filename = read_images_from_folder (path)
    for image in im_filename:
        print (image)
        im_rgb = np.array(scp.misc.imread(path+image), np.float32)
        im_original = np.copy(im_rgb)
        #print ('im:',im_original.shape, ' im_rgb:',im_rgb.shape)

        image_batch = im_rgb[np.newaxis]     # converts into list
        feed_dict = {
            test_data_node: image_batch,
            phase_train: False
        }
        dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)
        get_road_pixels(dense_prediction, im_rgb)
        lines = detect_lines(np.uint8 (im_rgb))
        im_lanes = draw_hough_lines(im_original, lines)
        # output_image to verify
        #writeImage(im[0], 'road/pred_image'+str(i)+'.png')
        scp.misc.imsave('road/Result'+str(i)+'.png', im_rgb)
        i+=1
          #writeImage(dense_prediction, 'pred_image.png')


def main(args):
    #checkArgs()
    test()
    #model.training()

if __name__ == '__main__':
  tf.app.run()
