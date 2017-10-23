import numpy as np

dataset_path = '/home/mohbat/RoadSegmentation/DataSet/CamSeq01/'
path_ckpt = 'ckp11_myModel/' # 'ckp32_SegNet' 'ckp11_myModel'  ckp11_SegNet
cnn_model = 'myModel' # 'myModel' SegNet

path_train = '/home/mohbat/RoadSegmentation/DataSet/CamSeq01/train.txt'
path_test = '/home/mohbat/RoadSegmentation/DataSet/CamSeq01/val.txt'
path_val = '/home/mohbat/RoadSegmentation/DataSet/CamSeq01/val.txt'
path_output = 'output/'

train_iteration = 20000  # total training Iterations
save_model_itr = 500    # save checkpoint after this number of Iterations
val_iter       = 500    # validate after this number of Iterations
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1

INITIAL_LEARNING_RATE = 0.001      # Initial learning rate.


# for CamVid
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
IMAGE_DEPTH = 3
NUM_CLASSES = 11 # 11 32

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

#classes_colors =  [[128,128,128], [128,0,0], [192,192,128], [255,69,0], [128,64,128], [60,40,222], [128,128,0],
#[192,128,128], [64,64,128], [64,0,128],[64,64,0],[0,128,192], [0,0,0]]

classes_colors = [
    [64, 128, 64],   # Animal
    [192, 0, 128],   # Archway
    [0, 128, 192],   # Bicyclist
    [0, 128, 64],    # Bridge
    [128, 0, 0],     # Building
    [64, 0, 128],    # Car
    [64, 0, 192],    # CartLuggagePram
    [192, 128, 64],  # Child
    [192, 192, 128], # Column_Pole
    [64, 64, 128],   # Fence
    [128, 0, 192],   # LaneMkgsDriv
    [192, 0, 64],    # LaneMkgsNonDriv
    [128, 128, 64],  # Misc_Text
    [192, 0, 192],   # MotorcycleScooter
    [128, 64, 64],   # OtherMoving
    [64, 192, 128],  # ParkingBlock
    [64, 64, 0],     # Pedestrian
    [128, 64, 128],  # Road
    [128, 128, 192], # RoadShoulder
    [0, 0, 192],     # Sidewalk
    [192, 128, 128], # SignSymbol
    [128, 128, 128], # Sky
    [64, 128, 192],  # SUVPickupTruck
    [0, 0, 64],      # TrafficCone
    [0, 64, 64],     # TrafficLight
    [192, 64, 128],  # Train
    [128, 128, 0],   # Tree
    [192, 128, 192], # Truck_Bus
    [64, 0, 64],     # Tunnel
    [192, 192, 0],   # VegetationMisc
    [0, 0, 0],       # Void
    [64, 192, 0]     # Wall
  ]

colors = {
  'segnet-32': [
    [64, 128, 64],   # Animal
    [192, 0, 128],   # Archway
    [0, 128, 192],   # Bicyclist
    [0, 128, 64],    # Bridge
    [128, 0, 0],     # Building
    [64, 0, 128],    # Car
    [64, 0, 192],    # CartLuggagePram
    [192, 128, 64],  # Child
    [192, 192, 128], # Column_Pole
    [64, 64, 128],   # Fence
    [128, 0, 192],   # LaneMkgsDriv
    [192, 0, 64],    # LaneMkgsNonDriv
    [128, 128, 64],  # Misc_Text
    [192, 0, 192],   # MotorcycleScooter
    [128, 64, 64],   # OtherMoving
    [64, 192, 128],  # ParkingBlock
    [64, 64, 0],     # Pedestrian
    [128, 64, 128],  # Road
    [128, 128, 192], # RoadShoulder
    [0, 0, 192],     # Sidewalk
    [192, 128, 128], # SignSymbol
    [128, 128, 128], # Sky
    [64, 128, 192],  # SUVPickupTruck
    [0, 0, 64],      # TrafficCone
    [0, 64, 64],     # TrafficLight
    [192, 64, 128],  # Train
    [128, 128, 0],   # Tree
    [192, 128, 192], # Truck_Bus
    [64, 0, 64],     # Tunnel
    [192, 192, 0],   # VegetationMisc
    [0, 0, 0],       # Void
    [64, 192, 0]     # Wall
  ]
}
