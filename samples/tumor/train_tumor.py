# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 

# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
#import cv2
import matplotlib
import matplotlib.pyplot as plt

# current datasets
trainingdictionary = {'hcc':{'dbfile':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/datalocation/trainingdata.csv','rootlocation':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse'},
                      'hccnorm':{'dbfile':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/datalocation/trainingnorm.csv','rootlocation':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse'},
                      'hccvol':{'dbfile':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/datalocation/tumordata.csv','rootlocation':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse'},
                      'hccvolnorm':{'dbfile':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/datalocation/tumornorm.csv','rootlocation':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse'},
                      'hccroinorm':{'dbfile':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/datalocation/tumorroi.csv','rootlocation':'/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse'},
                      'dbg':{'dbfile':'./debugdata.csv','rootlocation':'/rsrch1/ip/dtfuentes/objectdetection'},
                      'comp':{'dbfile':'./comptrainingdata.csv','rootlocation':'/rsrch1/ip/dtfuentes/objectdetection' }}

# ## Configurations

# In[2]:

# setup command line parser to control execution
from optparse import OptionParser
parser = OptionParser()
parser.add_option( "--initialize",
                  action="store_true", dest="initialize", default=False,
                  help="build initial sql file ", metavar = "BOOL")
parser.add_option( "--builddb",
                  action="store_true", dest="builddb", default=False,
                  help="load all training data into npy", metavar="FILE")
parser.add_option( "--traintumor",
                  action="store_true", dest="traintumor", default=False,
                  help="train model for tumor segmentation", metavar="FILE")
parser.add_option( "--setuptestset",
                  action="store_true", dest="setuptestset", default=False,
                  help="cross validate test set", metavar="FILE")
parser.add_option( "--setupobjtestset",
                  action="store_true", dest="setupobjtestset", default=False,
                  help="cross validate test set", metavar="FILE")
parser.add_option( "--debug",
                  action="store_true", dest="debug", default=False,
                  help="compare tutorial dtype", metavar="Bool")
parser.add_option( "--ModelID",
                  action="store", dest="modelid", default=None,
                  help="model id", metavar="FILE")
parser.add_option( "--outputModelBase",
                  action="store", dest="outputModelBase", default=None,
                  help="output location ", metavar="Path")
parser.add_option( "--predictmodel",
                  action="store", dest="predictmodel", default=None,
                  help="apply model to image", metavar="Path")
parser.add_option( "--predictimage",
                  action="store", dest="predictimage", default=None,
                  help="apply model to image", metavar="Path")
parser.add_option( "--segmentation",
                  action="store", dest="segmentation", default=None,
                  help="model output ", metavar="Path")
parser.add_option( "--modelpath",
                  action="store", dest="modelpath", default=None,
                  help="model location", metavar="Path")
parser.add_option( "--anonymize",
                  action="store", dest="anonymize", default=None,
                  help="setup info", metavar="Path")
parser.add_option( "--trainingmodel",
                  action="store", dest="trainingmodel", default='full',
                  help="setup info", metavar="string")
parser.add_option( "--trainingloss",
                  action="store", dest="trainingloss", default='dscimg',
                  help="setup info", metavar="string")
parser.add_option( "--trainingsolver",
                  action="store", dest="trainingsolver", default='SGD',
                  help="setup info", metavar="string")
parser.add_option( "--backbone",
                  action="store", dest="backbone", default='resnet50',
                  help="setup info", metavar="string")
parser.add_option( "--databaseid",
                  action="store", dest="databaseid", default='comp',
                  help="available data: hcc, crc, dbg", metavar="string")
parser.add_option( "--root_dir",
                  action="store", dest="root_dir", default=os.path.abspath("../../"),
                  help="code directory", metavar="string")
parser.add_option( "--kfolds",
                  type="int", dest="kfolds", default=5,
                  help="setup info", metavar="int")
parser.add_option( "--idfold",
                  type="int", dest="idfold", default=0,
                  help="setup info", metavar="int")
(options, args) = parser.parse_args()

# Root directory of the project
ROOT_DIR = options.root_dir

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

#get_ipython().run_line_magic('matplotlib', 'inline')

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class TumorConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "tumor"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background +  lesion

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # assume square image
    assert IMAGE_MAX_DIM == IMAGE_MIN_DIM 
    IMAGE_CHANNEL_COUNT = 1
    MEAN_PIXEL = 0
    IMAGE_RESIZE_MODE = 'none'
    BACKBONE = options.backbone
    MYOPTIMIZER = options.trainingsolver

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1
    STEPS_PER_EPOCH = 100

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.0,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    # raw dicom data is usually short int (2bytes) datatype
    # labels are usually uchar (1byte)
    IMG_DTYPE = np.int16
    SEG_DTYPE = np.uint8

    globaldirectorytemplate = '%s/%s/%s/%s/%d/%.2e%.2e%.2e%.2e%.2e/%03d%03d/%03d/%03d'

config = TumorConfig()
config.display()


# options dependency 
options.dbfile       = trainingdictionary[options.databaseid]['dbfile']
options.rootlocation = trainingdictionary[options.databaseid]['rootlocation']
options.sqlitefile = options.dbfile.replace('.csv','.sqlite' )
options.globalnpfile = options.dbfile.replace('.csv','%d.npy' % config.IMAGE_MAX_DIM)
print('database file: %s sqlfile: %s dbfile: %s rootlocation: %s' % (options.globalnpfile,options.sqlitefile,options.dbfile, options.rootlocation ) )

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs", config.globaldirectorytemplate % (options.databaseid,options.trainingloss,config.BACKBONE,options.trainingsolver,config.IMAGE_MAX_DIM,config.LOSS_WEIGHTS['rpn_class_loss'],config.LOSS_WEIGHTS['rpn_bbox_loss'],config.LOSS_WEIGHTS['mrcnn_class_loss'],config.LOSS_WEIGHTS['mrcnn_bbox_loss'],config.LOSS_WEIGHTS['mrcnn_mask_loss'],config.IMAGES_PER_GPU,config.VALIDATION_STEPS,options.kfolds,options.idfold) )
print (MODEL_DIR)

# build data base from CSV file
def GetDataDictionary():
  import sqlite3
  CSVDictionary = {}
  tagsconn = sqlite3.connect(options.sqlitefile)
  cursor = tagsconn.execute(' SELECT aq.* from trainingdata aq ;' )
  names = [description[0] for description in cursor.description]
  sqlStudyList = [ dict(zip(names,xtmp)) for xtmp in cursor ]
  for row in sqlStudyList :
       CSVDictionary[int( row['dataid'])]  =  {'image':row['image'], 'label':row['label'], 'uid':"%s" %row['uid']}  
  return CSVDictionary 

# setup kfolds
def GetSetupKfolds(numfolds,idfold,dataidsfull ):
  from sklearn.model_selection import KFold

  if (numfolds < idfold or numfolds < 1):
     raise("data input error")
  # split in folds
  if (numfolds > 1):
     kf = KFold(n_splits=numfolds)
     allkfolds = [ (list(map(lambda iii: dataidsfull[iii], train_index)), list(map(lambda iii: dataidsfull[iii], test_index))) for train_index, test_index in kf.split(dataidsfull )]
     train_index = allkfolds[idfold][0]
     test_index  = allkfolds[idfold][1]
  else:
     train_index = np.array(dataidsfull )
     test_index  = None  
  return (train_index,test_index)

## Borrowed from
## $(SLICER_DIR)/CTK/Libs/DICOM/Core/Resources/dicom-schema.sql
## 
## --
## -- A simple SQLITE3 database schema for modelling locally stored DICOM files
## --
## -- Note: the semicolon at the end is necessary for the simple parser to separate
## --       the statements since the SQlite driver does not handle multiple
## --       commands per QSqlQuery::exec call!
## -- ;
## TODO note that SQLite does not enforce the length of a VARCHAR. 
## TODO (9) What is the maximum size of a VARCHAR in SQLite?
##
## TODO http://www.sqlite.org/faq.html#q9
##
## TODO SQLite does not enforce the length of a VARCHAR. You can declare a VARCHAR(10) and SQLite will be happy to store a 500-million character string there. And it will keep all 500-million characters intact. Your content is never truncated. SQLite understands the column type of "VARCHAR(N)" to be the same as "TEXT", regardless of the value of N.
initializedb = """
DROP TABLE IF EXISTS 'Images' ;
DROP TABLE IF EXISTS 'Patients' ;
DROP TABLE IF EXISTS 'Series' ;
DROP TABLE IF EXISTS 'Studies' ;
DROP TABLE IF EXISTS 'Directories' ;
DROP TABLE IF EXISTS 'lstat' ;
DROP TABLE IF EXISTS 'overlap' ;

CREATE TABLE 'Images' (
 'SOPInstanceUID' VARCHAR(64) NOT NULL,
 'Filename' VARCHAR(1024) NOT NULL ,
 'SeriesInstanceUID' VARCHAR(64) NOT NULL ,
 'InsertTimestamp' VARCHAR(20) NOT NULL ,
 PRIMARY KEY ('SOPInstanceUID') );
CREATE TABLE 'Patients' (
 'PatientsUID' INT PRIMARY KEY NOT NULL ,
 'StdOut'     varchar(1024) NULL ,
 'StdErr'     varchar(1024) NULL ,
 'ReturnCode' INT   NULL ,
 'FindStudiesCMD' VARCHAR(1024)  NULL );
CREATE TABLE 'Series' (
 'SeriesInstanceUID' VARCHAR(64) NOT NULL ,
 'StudyInstanceUID' VARCHAR(64) NOT NULL ,
 'Modality'         VARCHAR(64) NOT NULL ,
 'SeriesDescription' VARCHAR(255) NULL ,
 'StdOut'     varchar(1024) NULL ,
 'StdErr'     varchar(1024) NULL ,
 'ReturnCode' INT   NULL ,
 'MoveSeriesCMD'    VARCHAR(1024) NULL ,
 PRIMARY KEY ('SeriesInstanceUID','StudyInstanceUID') );
CREATE TABLE 'Studies' (
 'StudyInstanceUID' VARCHAR(64) NOT NULL ,
 'PatientsUID' INT NOT NULL ,
 'StudyDate' DATE NULL ,
 'StudyTime' VARCHAR(20) NULL ,
 'AccessionNumber' INT NULL ,
 'StdOut'     varchar(1024) NULL ,
 'StdErr'     varchar(1024) NULL ,
 'ReturnCode' INT   NULL ,
 'FindSeriesCMD'    VARCHAR(1024) NULL ,
 'StudyDescription' VARCHAR(255) NULL ,
 PRIMARY KEY ('StudyInstanceUID') );

CREATE TABLE 'Directories' (
 'Dirname' VARCHAR(1024) ,
 PRIMARY KEY ('Dirname') );

CREATE TABLE lstat  (
   InstanceUID        VARCHAR(255)  NOT NULL,  --  'studyuid *OR* seriesUID'
   SegmentationID     VARCHAR(80)   NOT NULL,  -- UID for segmentation file 
   FeatureID          VARCHAR(80)   NOT NULL,  -- UID for image feature     
   LabelID            INT           NOT NULL,  -- label id for LabelSOPUID statistics of FeatureSOPUID
   Mean               REAL              NULL,
   StdD               REAL              NULL,
   Max                REAL              NULL,
   Min                REAL              NULL,
   Count              INT               NULL,
   Volume             REAL              NULL,
   ExtentX            INT               NULL,
   ExtentY            INT               NULL,
   ExtentZ            INT               NULL,
   PRIMARY KEY (InstanceUID,SegmentationID,FeatureID,LabelID) );

-- expected csv format
-- FirstImage,SecondImage,LabelID,InstanceUID,MatchingFirst,MatchingSecond,SizeOverlap,DiceSimilarity,IntersectionRatio
CREATE TABLE overlap(
   FirstImage         VARCHAR(80)   NOT NULL,  -- UID for  FirstImage  
   SecondImage        VARCHAR(80)   NOT NULL,  -- UID for  SecondImage 
   LabelID            INT           NOT NULL,  -- label id for LabelSOPUID statistics of FeatureSOPUID 
   InstanceUID        VARCHAR(255)  NOT NULL,  --  'studyuid *OR* seriesUID',  
   -- output of c3d firstimage.nii.gz secondimage.nii.gz -overlap LabelID
   -- Computing overlap #1 and #2
   -- OVL: 6, 11703, 7362, 4648, 0.487595, 0.322397  
   MatchingFirst      int           DEFAULT NULL,     --   Matching voxels in first image:  11703
   MatchingSecond     int           DEFAULT NULL,     --   Matching voxels in second image: 7362
   SizeOverlap        int           DEFAULT NULL,     --   Size of overlap region:          4648
   DiceSimilarity     real          DEFAULT NULL,     --   Dice similarity coefficient:     0.487595
   IntersectionRatio  real          DEFAULT NULL,     --   Intersection / ratio:            0.322397
   PRIMARY KEY (InstanceUID,FirstImage,SecondImage,LabelID) );
"""


# ## Notebook Preferences

# In[3]:


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Dataset
# 
# Create a synthetic dataset
# 
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
# 
# * load_image()
# * load_mask()
# * image_reference()

# In[4]:

class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, idsubset,numpydatabase):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("tumor", 1, "lesion")

        #setup kfolds
        dataidsfull= list(np.unique(numpydatabase['dataid']))
        (train_validation_index,test_index) = GetSetupKfolds(options.kfolds,options.idfold,dataidsfull)

        #break into independent training and validation sets
        trainingsplit = 0.9
        ntotaltrainval    =  len(train_validation_index)
        trainvalsplit     =  int(trainingsplit * ntotaltrainval   )
        train_index       =  train_validation_index[0: trainvalsplit  ]
        validation_index  =  train_validation_index[trainvalsplit:    ]

        print("train_index:",train_index,' validation_index: ',validation_index,' test_index: ',test_index)

        # uses 'views' for efficient memory usage
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
        print('copy data subsets into memory...')
        #axialbounds = numpydatabase['axialliverbounds'].copy()
        #dataidarray = numpydatabase['dataid'].copy()
        axialbounds = numpydatabase['axialliverbounds']
        dataidarray = numpydatabase['dataid']
        

        # setup indicies
        dbtrainindex         = np.isin(dataidarray,      train_index )
        dbvalidationindex    = np.isin(dataidarray, validation_index )
        dbtestindex          = np.isin(dataidarray,      test_index  )
        subsetidx_train      = np.all( np.vstack((axialbounds , dbtrainindex))      , axis=0 )
        subsetidx_validation = np.all( np.vstack((axialbounds , dbvalidationindex)) , axis=0 )
        subsetidx_test       = np.all( np.vstack((axialbounds , dbtestindex ))      , axis=0 )
        # error check
        if  np.sum(subsetidx_train   ) + np.sum(subsetidx_test)  + np.sum(subsetidx_validation ) != np.sum(axialbounds ) :
          raise("data error")
        print('copy memory map from disk to RAM...')

        # load training data as views
        #trainingsubset = numpydatabase[subsetidx   ].copy()
        #trainingsubset   = numpydatabase[subsetidx_train      ]
        #validationsubset = numpydatabase[subsetidx_validation ]
        if   (idsubset =='train'):
          self.dbsubset = numpydatabase[subsetidx_train      ]
        elif (idsubset =='validate'):
          self.dbsubset = numpydatabase[subsetidx_validation ]

        for i in range(len(self.dbsubset)):
            self.add_image("tumor", image_id=self.dbsubset['dataid'][i], path=None,
                           width=config.IMAGE_MAX_DIM, height=config.IMAGE_MAX_DIM,
                           bg_color=0, tumor=[(self.dbsubset['dataid'][i],self.dbsubset['sliceid'][i])])


    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        myimage = self.dbsubset['imagedata'][image_id]
        return myimage[:,:,np.newaxis]
                                      
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "tumor":
            return info["tumor"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        # Map class names to class IDs.
        y_train=self.dbsubset['truthdata'][image_id]
        tvalues=y_train.astype(np.uint8)
        sliceclassid = np.unique(tvalues)
        t_max=np.max(tvalues)
        if t_max> 1 :
          # Convert the labels into a one-hot representation
          from keras.utils.np_utils import to_categorical
          y_train_one_hot = to_categorical(tvalues, num_classes=t_max+1).reshape((y_train.shape)+(t_max+1,))
          mask =  y_train_one_hot[:,:,sliceclassid[2:]]
          class_ids = np.clip(sliceclassid[2:],None,1)
          #print("UID:", sliceclassid,"Range of values: [0, {}]".format(t_max),"class_ids ",class_ids )
          return mask.astype(np.bool), class_ids.astype(np.int32)
        else:
          # tumor only, return empty for liver as BG
          #print("UID:", sliceclassid,"Range of values: [0, {}]".format(t_max),"class_ids ",np.empty([0], np.int32))
          return np.empty([0, 0, 0]), np.empty([0], np.int32)


# define this as a function, variable scope will be local to the function.
def TrainODModel():
  # load database
  print('loading memory map db for large dataset')
  #npdatabase = np.load(options.globalnpfile,mmap_mode='r')
  npdatabase = np.load(options.globalnpfile)

  # Training dataset
  dataset_train = ShapesDataset()
  dataset_train.load_shapes('train',npdatabase )
  dataset_train.prepare()
  
  # Validation dataset
  dataset_val = ShapesDataset()
  dataset_val.load_shapes('validate',npdatabase )
  dataset_val.prepare()

  # ensure we get the same results each time we run the code
  np.random.seed(seed=0) 
  #np.random.shuffle(dataset_train.dbsubset )
  #np.random.shuffle(dataset_val.dbsubset)

  ## # subset within bounding box that has liver
  ## totnslice = len(dataset_train.dbsubset ) + len(dataset_val.dbsubset)
  ## slicesplit =  len(dataset_train.dbsubset )
  ## print("nslice: ",totnslice ," split: " ,slicesplit )

  ## # FIXME - Verify stacking indicies
  ## x_train=np.vstack((dataset_train.dbsubset ['imagedata'],dataset_val.dbsubset['imagedata']))
  ## y_train=np.vstack((dataset_train.dbsubset ['truthdata'],dataset_val.dbsubset['truthdata']))
  ## TRAINING_SLICES      = slice(0,slicesplit)
  ## VALIDATION_SLICES    = slice(slicesplit,totnslice)

  # In[6]:
  
  # Load and display random samples
  image_ids = np.random.choice(dataset_train.image_ids, 20)
  print(image_ids)
  import nibabel as nib  
  for image2did in image_ids:
      image     = dataset_train.load_image(image2did)
      imageinfo = dataset_train.image_reference(image2did)
      mask, class_ids = dataset_train.load_mask(image2did)
      print(imageinfo,image2did,image.shape, mask.shape, class_ids, dataset_train.class_names)
      #visualize.display_top_masks(np.squeeze(image), mask, class_ids, dataset_train.class_names,image2did)
      original_image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset_train, config, image2did, use_mini_mask=False)
      #visualize.display_instances(np.repeat(original_image,3,axis=2), gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))
      objmask = np.zeros( gt_mask.shape[0:2], dtype='uint8' )
      for iii,idclass in enumerate(gt_class_id):
          objmask[gt_bbox[iii][0]:gt_bbox[iii][2], gt_bbox[iii][1]:gt_bbox[iii][3] ] = 1
          objmask = objmask + idclass*gt_mask[:,:,iii].astype('uint8')

      segnii = nib.Nifti1Image(objmask.astype('uint8') , None )
      segnii.to_filename( 'tmp/mask.%05d.nii.gz'  % image2did )
      imgnii = nib.Nifti1Image(image , None )
      imgnii.to_filename( 'tmp/image.%05d.nii.gz' % image2did )

  # ## Create Model
  
  # In[ ]:
  
  # Create model in training mode
  model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=MODEL_DIR)
  
  
  # In[7]:
  
  
  # Which weights to start with?
  init_with = "coco"  # imagenet, coco, or last
  init_with = "last"  # imagenet, coco, or last
  
  # ## Training
  # 
  # Train in two stages:
  # 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
  # 
  # 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.
  
  if init_with == "imagenet":
      model.load_weights(model.get_imagenet_weights(), by_name=True)
      raise(" freeze backbone ?  input error")
  elif init_with == "coco":
      # Load weights trained on MS COCO, but skip layers that
      # are different due to the different number of classes
      # See README for instructions to download the COCO weights
      model.load_weights(COCO_MODEL_PATH, by_name=True,
                         exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                  "mrcnn_bbox", "mrcnn_mask","conv1"])
  
      # Train the head branches
      # Passing layers="heads" freezes all layers except the head
      # layers. You can also pass a regular expression to select
      # which layers to train by name pattern.
      model.train(dataset_train, dataset_val, 
                  learning_rate=config.LEARNING_RATE, 
                  epochs=100, 
                  layers='heads')
  
  elif init_with == "last":
      # Load the last model you trained and continue training
      model.load_weights(model.find_last(), by_name=True)
  
  # Fine tune all layers
  # Passing layers="all" trains all layers. You can also 
  # pass a regular expression to select which layers to
  # train by name pattern.
  model.train(dataset_train, dataset_val, 
              learning_rate=config.LEARNING_RATE/10.,
              epochs=500, 
              layers="all")
  
  # Save weights
  # Typically not needed because callbacks save after every epoch
  # Uncomment to save manually
  model_path = os.path.join(MODEL_DIR, "mask_rcnn_tumor.h5")
  model.keras_model.save_weights(model_path)

# In[5]:


#############################################################
# build initial sql file 
#############################################################
if (options.initialize ):
  import sqlite3
  import pandas
  # build new database
  os.system('rm %s'  % options.sqlitefile )
  tagsconn = sqlite3.connect(options.sqlitefile )
  for sqlcmd in initializedb.split(";"):
     tagsconn.execute(sqlcmd )
  # load csv file
  df = pandas.read_csv(options.dbfile,delimiter='\t')
  df.to_sql('trainingdata', tagsconn , if_exists='append', index=False)

##########################
# preprocess database and store to disk
##########################
elif (options.builddb):
  import nibabel as nib  
  from scipy import ndimage
  import skimage.transform

  # create  custom data frame database type
  mydatabasetype = [('dataid', int),('sliceid', int), ('axialliverbounds',bool), ('imagedata','(%d,%d)int16' %(config.IMAGE_MAX_DIM,config.IMAGE_MAX_DIM)),('truthdata','(%d,%d)uint8' % (config.IMAGE_MAX_DIM,config.IMAGE_MAX_DIM))]

  # initialize empty dataframe
  numpydatabase = np.empty(0, dtype=mydatabasetype  )

  # build data base 
  databaseinfo = GetDataDictionary()

  # load all data 
  totalnslice = 0 
  for idrow in databaseinfo.keys():
    row = databaseinfo[idrow ]
    imagelocation = '%s/%s' % (options.rootlocation,row['image'])
    truthlocation = '%s/%s' % (options.rootlocation,row['label'])

    # load nifti file
    imagedata = nib.load(imagelocation )
    numpyimage= imagedata.get_data().astype(config.IMG_DTYPE )
    nslice = numpyimage.shape[2]
    if (config.IMAGE_MAX_DIM,config.IMAGE_MAX_DIM)  == numpyimage.shape[0:2]:
      print("no resizing")
      resimage=numpyimage
    else:
      print("resizing image")
      resimage=skimage.transform.resize(numpyimage,(config.IMAGE_MAX_DIM,config.IMAGE_MAX_DIM,nslice),order=0,mode='constant',preserve_range=True).astype(config.IMG_DTYPE)

    # load nifti file
    truthdata = nib.load(truthlocation )
    numpytruth= truthdata.get_data().astype(config.SEG_DTYPE)
    # error check
    assert numpytruth.shape[0:2] ==  numpyimage.shape[0:2]
    assert nslice  == numpytruth.shape[2]
    if (config.IMAGE_MAX_DIM,config.IMAGE_MAX_DIM)  == numpyimage.shape[0:2]:
      print("no resizing")
      restruth=numpytruth
    else:
      print("resizing image")
      restruth=skimage.transform.resize(numpytruth,(config.IMAGE_MAX_DIM,config.IMAGE_MAX_DIM,nslice),order=0,mode='constant',preserve_range=True).astype(config.SEG_DTYPE)

    # bounding box for each label
    if( np.max(restruth) ==1 ) :
      (liverboundingbox,)  = ndimage.find_objects(restruth)
    else:
      boundingboxes = ndimage.find_objects(restruth)
      print(boundingboxes)
      liverboundingbox = boundingboxes[0]

    print(idrow, imagelocation,truthlocation, nslice )

    # error check
    if( nslice  == restruth.shape[2]):
      # custom data type to subset  
      datamatrix = np.zeros(nslice  , dtype=mydatabasetype )
      
      # custom data type to subset  
      datamatrix ['dataid']          = np.repeat(idrow ,nslice  ) 
      datamatrix ['sliceid']         = np.arange(1,nslice+1)
      #datamatrix ['xbounds']      = np.repeat(boundingbox[0],nslice  ) 
      #datamatrix ['ybounds']      = np.repeat(boundingbox[1],nslice  ) 
      #datamatrix ['zbounds']      = np.repeat(boundingbox[2],nslice  ) 
      #datamatrix ['nslice' ]      = np.repeat(nslice,nslice  ) 
      # id the slices within the bounding box
      axialliverbounds                              = np.repeat(False,nslice  ) 
      axialliverbounds[liverboundingbox[2]]         = True
      datamatrix ['axialliverbounds'   ]            = axialliverbounds
      ## VERIFY reordering
      ## xxx = np.random.rand(4,4,8)
      ## yyy = np.array([xxx[:,:,iii] for iii in range(8)])
      ## yyy.shape
      ## (8, 4, 4)
      ## yyy[0] - xxx[:,:,0]
      ## array([[0., 0., 0., 0.],
      ##        [0., 0., 0., 0.],
      ##        [0., 0., 0., 0.],
      ##        [0., 0., 0., 0.]])
      datamatrix ['imagedata']                      = np.array([resimage[:,:,iii] for iii in range(nslice)])
      datamatrix ['truthdata']                      = np.array([restruth[:,:,iii] for iii in range(nslice)])
      numpydatabase = np.hstack((numpydatabase,datamatrix))
      # count total slice for QA
      totalnslice = totalnslice + nslice 
    else:
      print('training data error image[2] = %d , truth[2] = %d ' % (nslice,restruth.shape[2]))

  # save numpy array to disk
  np.save( options.globalnpfile,numpydatabase )

##########################
# build NN model for tumor segmentation
##########################
elif (options.traintumor):
  TrainODModel()

##########################
# apply model to test set
##########################
elif (options.setupobjtestset):
  # get id from setupfiles
  databaseinfo = GetDataDictionary()
  dataidsfull = list(databaseinfo.keys()) 

  uiddictionary = {}
  modeltargetlist = []
  makefileoutput = '%skfold%03d.makefile' % (options.databaseid,options.kfolds) 
  # open makefile
  with open(makefileoutput ,'w') as fileHandle:
      for iii in range(options.kfolds):
        (train_set,test_set) = GetSetupKfolds(options.kfolds,iii,dataidsfull)
        uidoutputdir= config.globaldirectorytemplate % (options.databaseid,options.trainingloss,config.BACKBONE,options.trainingsolver,config.IMAGE_MAX_DIM,config.LOSS_WEIGHTS['rpn_class_loss'],config.LOSS_WEIGHTS['rpn_bbox_loss'],config.LOSS_WEIGHTS['mrcnn_class_loss'],config.LOSS_WEIGHTS['mrcnn_bbox_loss'],config.LOSS_WEIGHTS['mrcnn_mask_loss'],config.IMAGES_PER_GPU,config.VALIDATION_STEPS,options.kfolds,iii) 

        modelweights   = '%s/mask_rcnn_tumor.h5' % uidoutputdir
        fileHandle.write('%s: \n' % modelweights )
        fileHandle.write('\tpython train_tumor.py --databaseid=%s --traintumor --idfold=%d --kfolds=%d \n' % (options.databaseid,iii,options.kfolds))
        modeltargetlist.append(modelweights )
        uiddictionary[iii]=[]
        for idtest in test_set:
           # write target
           imageprereq    = '$(TRAININGROOT)/%s' % databaseinfo[idtest]['image']
           labelprereq    = '$(TRAININGROOT)/%s' % databaseinfo[idtest]['label']
           setuptarget    = '$(WORKDIR)/%s/%s/setup' % (databaseinfo[idtest]['uid'],config.BACKBONE)
           uiddictionary[iii].append(databaseinfo[idtest]['uid'] )
           fileHandle.write('%s: \n' % (setuptarget  ) )
           fileHandle.write('\tmkdir -p   $(@D)          \n'                  )
           fileHandle.write('\tln -snf %s $(@D)/image.nii\n' % imageprereq    )
           fileHandle.write('\tln -snf %s $(@D)/label.nii\n' % labelprereq    )
           fileHandle.write('\tln -snf ../../../../../logs/%s $(@D)/mask_rcnn_tumor.h5\n' % modelweights  )

  # build job list
  with open(makefileoutput , 'r') as original: datastream = original.read()
  with open(makefileoutput , 'w') as modified:
     modified.write( 'TRAININGROOT=%s\n' % options.rootlocation + 'SQLITEDB=%s\n' % options.sqlitefile + "models: %s \n" % ' '.join(modeltargetlist))
     for idkey in uiddictionary.keys():
        modified.write("UIDLIST%d=%s \n" % (idkey,' '.join(uiddictionary[idkey])))
     modified.write("UIDLIST=%s \n" % " ".join(map(lambda x : "$(UIDLIST%d)" % x, uiddictionary.keys()))    +datastream)



elif (options.predictimage != None and options.modelpath != None and options.segmentation != None ):
  # ## Detection
  
  # In[11]:
  import nibabel as nib
  imagepredict = nib.load(options.predictimage)
  imageheader  = imagepredict.header
  numpypredict = imagepredict.get_data().astype(config.IMG_DTYPE )
  # error check
  assert numpypredict.shape[0:2] == (config.IMAGE_MAX_DIM,config.IMAGE_MAX_DIM)
  nslice = numpypredict.shape[2]
  print('nslice = %d' % nslice)
  
  class InferenceConfig(TumorConfig):
      GPU_COUNT = 1
      IMAGES_PER_GPU = 1
  
  inference_config = InferenceConfig()
  
  # Recreate the model in inference mode
  model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            model_dir=MODEL_DIR)
  
  # Get path to saved weights
  # Either set a specific path or find last trained weights
  # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
  model_path = options.modelpath
  
  # Load trained weights
  print("Loading weights from ", model_path)
  model.load_weights(model_path, by_name=True)
  
  objmask  = np.zeros( numpypredict.shape, dtype='uint8' )
  scoreimg = np.zeros( numpypredict.shape, dtype='float16' )
  # In[ ]:
  # FIXME - vectorize this
  for iii in range(nslice):
    print(iii,)
    myimage = numpypredict[:,:,iii]
    results = model.detect([myimage[:,:,np.newaxis] ], verbose=1)
    myoutput = results[0]
    for jjj,idclass in enumerate(myoutput['class_ids']):
        scoreimg[myoutput['rois'][jjj][0]:myoutput['rois'][jjj][2], myoutput['rois'][jjj][1]:myoutput['rois'][jjj][3], iii ] = myoutput['scores'][jjj]
        objmask[myoutput['rois'][jjj][0]:myoutput['rois'][jjj][2], myoutput['rois'][jjj][1]:myoutput['rois'][jjj][3], iii ] = 1
        objmask[:,:,iii] = objmask[:,:,iii] + idclass*myoutput['masks'][:,:,jjj].astype('uint8')

  # write out
  segout_img = nib.Nifti1Image(objmask , None, header=imageheader)
  segout_img.to_filename( options.segmentation )
  scrout_img = nib.Nifti1Image(scoreimg, None, header=imageheader)
  scrout_img.to_filename( '/'.join(options.segmentation.split('/')[:-1]) + '/objscore.nii.gz' )

##########################
# print help
##########################
else:
  import keras; import tensorflow as tf
  print("keras version: ",keras.__version__, 'TF version:',tf.__version__)
  print("debug: /opt/apps/miniconda/maskrcnn/lib/python3.6/site-packages/keras/engine/training.py(1450)train_on_batch()->[566.86456, 114.579956, 168.20625, 284.07834, 0.0, 0.0]")
  print("debug: /opt/apps/miniconda/maskrcnn/lib/python3.6/site-packages/keras/engine/training_generator.py(174)fit_generator()")
  dataset_test = LoadDataset()
  parser.print_help()


# print("#test rpn_bbox_loss print")
# tf.Print(loss,[loss], "#my rpn_bbox_loss print")
# with tf.Session() as sess:
#     # initialize all of the variables in the session
#     print('loss       = %f' % sess.run(loss))
# 





