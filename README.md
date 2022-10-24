# Cell-Nuclei Segmentation using U-Net in Python

<img src="images/banner-02.png" width=500 />

## 1. Objective

The objective of this project is to develop, train and evaluate the performance of a U-Net model to segment cells' nuclei from medical cell images.

## 2. Motivation

A human body’s has in the order of 30 trillion cells:

  * Each cell contain a nucleus full of DNA, the genetic code that programs each cell.
  * Identifying the cells’ nuclei is the starting point for most analyses
  * Identifying nuclei allows researchers to identify each individual cell in a sample
  * Measuring how cells react to various treatments, the researcher can understand the underlying biological processes at work.

Thus, identifying cells' nuclei is often a critical first step in analyzing microscopy images of cells, and classical image processing algorithms are most commonly used for this task. Recent developments in deep learning can yield superior accuracy.

In this work, we demonstrate the end-to-end process of segmenting cells' nuclei using U-Net. 

## 3. Data


In order to accomplish our task, we use the following dataset:

* Kaggle 2018 Data Science Bowl competition:
* Objective: Find the nuclei in divergent images to advance medical discovery
* Link: https://www.kaggle.com/c/data-science-bowl-2018/data

The dataset contains a large number of segmented nuclei images:

* Training data subset:
  * 670 nuclei cell annotated images (in the sub-folder: \images)
  * For each image, ground-truth annotation masks associated with each nucleus cell ((in the sub-folder: \masks).

* Test data subset:
  * 65 images without ground-truth (in the sub-folder: \images).
  * The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (bright-field vs. fluorescence)

The dataset is designed to challenge an algorithm's ability to generalize across these variations.

4. Development

In this section, we shall develop, train, deploy and evaluate the performance of a CNN model to classify malaria cells into parasitized (0) or uninfected (1).

  * Author: Mohsen Ghazel (mghazel)
  * Date: April 1st, 2021
  * Project: Cell nuclei segmentation using U-Net:

The this project, we develop a U-Net to detect and segment cells’ nuclei:

  * A human body’s has in the order of 30 trillion cells:
  * Each cell contain a nucleus full of DNA, the genetic code that programs each cell.
  * Identifying the cells’ nuclei is the starting point for most analyses because
  * Identifying nuclei allows researchers to identify each individual cell in a sample
  * Measuring how cells react to various treatments, the researcher can understand the underlying biological processes at work.
  * We shall demonstrate the end-to-end process of segmenting cells' nuclei using U-Net.
  
### 4.1. Step 1: Python imports and global variables
#### 4.1.1/ Imports:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;">print<span style="color:#808030; ">(</span><span style="color:#074726; ">__doc__</span><span style="color:#808030; ">)</span>

Automatically created module <span style="color:#800000; font-weight:bold; ">for</span> IPython interactive environment
</pre>

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Numpy</span>
<span style="color:#800000; font-weight:bold; ">import</span> numpy <span style="color:#800000; font-weight:bold; ">as</span> np
<span style="color:#696969; "># matplot lib</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
<span style="color:#696969; "># opencv</span>
<span style="color:#800000; font-weight:bold; ">import</span> cv2
<span style="color:#696969; "># PIL library</span>
<span style="color:#800000; font-weight:bold; ">from</span> PIL <span style="color:#800000; font-weight:bold; ">import</span> Image
<span style="color:#696969; "># tensorflow</span>
<span style="color:#800000; font-weight:bold; ">import</span> tensorflow <span style="color:#800000; font-weight:bold; ">as</span> tf
<span style="color:#696969; "># keras</span>
<span style="color:#800000; font-weight:bold; ">import</span> keras

<span style="color:#696969; "># sklearn imports</span>
<span style="color:#696969; "># - nededed for splitting the dataset into training and testing subsets</span>
<span style="color:#800000; font-weight:bold; ">from</span> sklearn<span style="color:#808030; ">.</span>model_selection <span style="color:#800000; font-weight:bold; ">import</span> train_test_split
<span style="color:#696969; "># - nededed for 1-hot coding of the image labels</span>
<span style="color:#800000; font-weight:bold; ">from</span> keras<span style="color:#808030; ">.</span>utils <span style="color:#800000; font-weight:bold; ">import</span> to_categorical

<span style="color:#696969; "># tqdm</span>
<span style="color:#800000; font-weight:bold; ">from</span> tqdm <span style="color:#800000; font-weight:bold; ">import</span> tqdm 

<span style="color:#696969; "># skimage</span>
<span style="color:#696969; "># read and imshow</span>
<span style="color:#800000; font-weight:bold; ">from</span> skimage<span style="color:#808030; ">.</span>io <span style="color:#800000; font-weight:bold; ">import</span> imread<span style="color:#808030; ">,</span> imsave<span style="color:#808030; ">,</span> imshow
<span style="color:#696969; "># resize</span>
<span style="color:#800000; font-weight:bold; ">from</span> skimage<span style="color:#808030; ">.</span>transform <span style="color:#800000; font-weight:bold; ">import</span> resize

<span style="color:#696969; "># Image package</span>
<span style="color:#800000; font-weight:bold; ">from</span> IPython<span style="color:#808030; ">.</span>display <span style="color:#800000; font-weight:bold; ">import</span> Image
<span style="color:#696969; "># using HTML code</span>
<span style="color:#800000; font-weight:bold; ">from</span> IPython<span style="color:#808030; ">.</span>core<span style="color:#808030; ">.</span>display <span style="color:#800000; font-weight:bold; ">import</span> HTML 

<span style="color:#696969; "># set the keras backend to tensorflow</span>
<span style="color:#696969; "># os.environ['KERAS_BACKEND'] = 'tensorflow'</span>
<span style="color:#696969; "># I/O</span>
<span style="color:#800000; font-weight:bold; ">import</span> os
<span style="color:#696969; "># sys</span>
<span style="color:#800000; font-weight:bold; ">import</span> sys

<span style="color:#696969; "># datetime</span>
<span style="color:#800000; font-weight:bold; ">import</span> datetime

<span style="color:#696969; "># random</span>
<span style="color:#800000; font-weight:bold; ">import</span> random

<span style="color:#696969; "># check for successful package imports and versions</span>
<span style="color:#696969; "># python</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Python version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>sys<span style="color:#808030; ">.</span>version<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># OpenCV</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"OpenCV version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># numpy</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Numpy version  : {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
</pre>


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;">Mounted at <span style="color:#44aadd; ">/</span>content<span style="color:#44aadd; ">/</span>drive
</pre>


### 4.1.2. Global variables

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># We set the Numpy pseudo-random generator at a fixed value:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - This ensures repeatable results everytime you run the code. </span>
<span style="color:#696969; "># the seed </span>
RANDOM_SEED <span style="color:#808030; ">=</span> <span style="color:#008c00; ">101</span>
<span style="color:#696969; "># set the random seed</span>
np<span style="color:#808030; ">.</span>random<span style="color:#808030; ">.</span>seed <span style="color:#808030; ">=</span> RANDOM_SEED

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Set the random state to 101</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - This ensures repeatable results every time you run the code. </span>
RANDOM_STATE <span style="color:#808030; ">=</span> <span style="color:#008c00; ">101</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Set the sample images path</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - contains sample images such as the implemented U-Net structure</span>
SAMPLE_IMAGES_PATH <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">"/content/drive/.../sample-images/"</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Set the data directory where cell-nuclei data sets are stored</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># training images</span>
TRAIN_IMAGES_PATH <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'/content/drive/.../cell-nuclei-data/stage1_train/'</span>

<span style="color:#696969; "># test images</span>
TEST_IMAGES_PATH <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'/content/drive/.../cell-nuclei-data/stage1_test/'</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Set the input images size: RGB images with size: 128x128 pixels</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># input image-width</span>
INPUT_IMAGE_WIDTH <span style="color:#808030; ">=</span> <span style="color:#008c00; ">128</span>
<span style="color:#696969; "># input image-width</span>
INPUT_IMAGE_HEIGHT <span style="color:#808030; ">=</span> <span style="color:#008c00; ">128</span>
<span style="color:#696969; "># input image-channels</span>
INPUT_IMAGE_CHANNELS <span style="color:#808030; ">=</span> <span style="color:#008c00; ">3</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># A flag to combine seperate ground-truth cell-nuclei masks into one </span>
<span style="color:#696969; "># common mask:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - Set this flag to 1 to combine</span>
<span style="color:#696969; "># - Set this flag to 0 to use the already combined mask</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
COMBINE_GROUND_TRUTH_MASKS <span style="color:#808030; ">=</span> <span style="color:#008c00; ">1</span>
<span style="color:#696969; "># the name of tthe file of the combined mask</span>
COMBINED_MASK_FILE_NAME <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'combined_gt_mask.png'</span>
</pre>

### 4.2. Step 2: Read, resize and visualize the input data set

* We use the following dataset:

  * Kaggle 2018 Data Science Bowl:
  * Objective: Find the nuclei in divergent images to advance medical discovery
  Link: https://www.kaggle.com/c/data-science-bowl-2018/data
  * The dataset contains a large number of segmented nuclei images:
      * Training data subset:
        * 670 nuclei cell annotated images (in the sub-folder: \images)
            * For each image, ground-truth annotation masks associated with each nucleus cell ((in the sub-folder: \masks).
      * Test data subset:
        * 65 images without ground-truth (in the sub-folder: \images).
        * The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (bright-field vs. fluorescence)
        * The dataset is designed to challenge an algorithm's ability to generalize across these variations.

#### 4.2.1. Create data structures to store the read input data:

* There are 670 nuclei cell annotated images (in the sub-folder: \images):
* For each image, ground-truth annotation masks associated with each nucleus cell ((in the sub-folder: \masks

##### 4.2.1.1/ Read and resize the input training images:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># The structure of the training-data folder is as follows:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - For each training image: </span>
<span style="color:#696969; ">#</span>
<span style="color:#696969; ">#  - \stage1_train\52b267e20519174e3ce1e1994b5d677804b16bc670aa5f6ffb6344a0fdf63fde\images</span>
<span style="color:#696969; ">#  - \stage1_train\52b267e20519174e3ce1e1994b5d677804b16bc670aa5f6ffb6344a0fdf63fde\masks</span>
<span style="color:#696969; ">#</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># This needs to be parsed to extract the the ID for each image:</span>
<span style="color:#696969; ">#</span>
<span style="color:#696969; ">#    52b267e20519174e3ce1e1994b5d677804b16bc670aa5f6ffb6344a0fdf63fde</span>
<span style="color:#696969; ">#</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">-------------------------------------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Reading and formatting the training images:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"-------------------------------------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># extract the IDs for the training images</span>
train_ids <span style="color:#808030; ">=</span> next<span style="color:#808030; ">(</span>os<span style="color:#808030; ">.</span>walk<span style="color:#808030; ">(</span>TRAIN_IMAGES_PATH<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span>

<span style="color:#696969; "># the number of train images</span>
num_train_images <span style="color:#808030; ">=</span> <span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>train_ids<span style="color:#808030; ">)</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"The number of train images = {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>num_train_images<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Allocate a data structure to store the read and resized train images: </span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 1) 4D numpy array to store the images:</span>
<span style="color:#696969; ">#     - The number of training images: (len(train_ids)</span>
<span style="color:#696969; ">#     - Each training image will be resized to: </span>
<span style="color:#696969; ">#       - INPUT_IMAGE_HEIGHT x INPUT_IMAGE_WIDTH x INPUT_IMAGE_CHANNELS</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
X_train <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>zeros<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>train_ids<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                    INPUT_IMAGE_HEIGHT<span style="color:#808030; ">,</span> 
                    INPUT_IMAGE_WIDTH<span style="color:#808030; ">,</span> 
                    INPUT_IMAGE_CHANNELS<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> dtype<span style="color:#808030; ">=</span>np<span style="color:#808030; ">.</span>uint8<span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 2) 3D numpy array to store the masks:</span>
<span style="color:#696969; ">#     - The number of training images: (len(train_ids)</span>
<span style="color:#696969; ">#     - Each training grayscale/binary image-mask will be resized to: </span>
<span style="color:#696969; ">#       - INPUT_IMAGE_HEIGHT x INPUT_IMAGE_WIDTH x 1</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
Y_train <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>zeros<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>train_ids<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                    INPUT_IMAGE_HEIGHT<span style="color:#808030; ">,</span> 
                    INPUT_IMAGE_WIDTH<span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                   dtype<span style="color:#808030; ">=</span>np<span style="color:#808030; ">.</span><span style="color:#400000; ">bool</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Save the common mask:</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
<span style="color:#696969; "># Once the seperate masks are combined together, save the combined mask:</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
<span style="color:#696969; "># - This will speed up the prcessing</span>
<span style="color:#696969; "># - It is very slow to read the many individual masks</span>
<span style="color:#696969; "># - Instead, we just read the combined mask</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
<span style="color:#696969; "># the combined mask name</span>
combined_mask_name <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'combined_mask.png'</span><span style="color:#808030; ">;</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Read and resize each training image</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># iterate over the training-images IDs</span>
<span style="color:#800000; font-weight:bold; ">for</span> n<span style="color:#808030; ">,</span> id_ <span style="color:#800000; font-weight:bold; ">in</span> tqdm<span style="color:#808030; ">(</span><span style="color:#400000; ">enumerate</span><span style="color:#808030; ">(</span>train_ids<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> total<span style="color:#808030; ">=</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>train_ids<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#696969; "># set the path of the next training image</span>
    path <span style="color:#808030; ">=</span> TRAIN_IMAGES_PATH <span style="color:#44aadd; ">+</span> id_
    <span style="color:#696969; "># read the next train image</span>
    img <span style="color:#808030; ">=</span> imread<span style="color:#808030; ">(</span>path <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">'/images/'</span> <span style="color:#44aadd; ">+</span> id_ <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">'.png'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span><span style="color:#808030; ">:</span>INPUT_IMAGE_CHANNELS<span style="color:#808030; ">]</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># resize the read training image to the specified desired size:</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># - INPUT_IMAGE_HEIGHT x INPUT_IMAGE_WIDTH</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    img <span style="color:#808030; ">=</span> resize<span style="color:#808030; ">(</span>img<span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span>INPUT_IMAGE_HEIGHT<span style="color:#808030; ">,</span> 
                       INPUT_IMAGE_WIDTH<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                       mode<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'constant'</span><span style="color:#808030; ">,</span> 
                       preserve_range<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># store the resized train image in the X_train 4D arrays </span>
    X_train<span style="color:#808030; ">[</span>n<span style="color:#808030; ">]</span> <span style="color:#808030; ">=</span> img
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># The ground-truth annotation masks stored in /masks subfolder:</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; ">#  - A seperate mask is prvided for each cell nucleus in the image</span>
    <span style="color:#696969; ">#  - These masks need to be merged together into one mask</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># initialize the common mask</span>
    mask <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>zeros<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span>INPUT_IMAGE_HEIGHT<span style="color:#808030; ">,</span> INPUT_IMAGE_WIDTH<span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> dtype<span style="color:#808030; ">=</span>np<span style="color:#808030; ">.</span><span style="color:#400000; ">bool</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># Check if we need to combined the masks into a single common mask:</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> COMBINE_GROUND_TRUTH_MASKS <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">1</span> <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; ">#-----------------------------------------------------------------------</span>
        <span style="color:#696969; "># Read and combined the individual masks into one mask:</span>
        <span style="color:#696969; ">#-----------------------------------------------------------------------</span>
        <span style="color:#696969; "># iterate over each eask mask file in the /masks sub-folder</span>
        <span style="color:#800000; font-weight:bold; ">for</span> mask_file <span style="color:#800000; font-weight:bold; ">in</span> next<span style="color:#808030; ">(</span>os<span style="color:#808030; ">.</span>walk<span style="color:#808030; ">(</span>path <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">'/masks/'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">]</span><span style="color:#808030; ">:</span>
            <span style="color:#696969; "># read the new mask image</span>
            mask_ <span style="color:#808030; ">=</span> imread<span style="color:#808030; ">(</span>path <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">'/masks/'</span> <span style="color:#44aadd; ">+</span> mask_file<span style="color:#808030; ">)</span>
            <span style="color:#696969; ">#-------------------------------------------------------------------</span>
            <span style="color:#696969; "># Resize the mask image as done for the image:</span>
            <span style="color:#696969; ">#-------------------------------------------------------------------</span>
            <span style="color:#696969; "># - Each training grayscale/binary image-mask will be resized to: </span>
            <span style="color:#696969; "># - INPUT_IMAGE_HEIGHT x INPUT_IMAGE_WIDTH x 1 </span>
            <span style="color:#696969; ">#-------------------------------------------------------------------</span>
            mask_ <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>expand_dims<span style="color:#808030; ">(</span>resize<span style="color:#808030; ">(</span>mask_<span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span>INPUT_IMAGE_HEIGHT<span style="color:#808030; ">,</span> 
                                                  INPUT_IMAGE_WIDTH<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> mode<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'constant'</span><span style="color:#808030; ">,</span>  
                                                  preserve_range<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> axis<span style="color:#808030; ">=</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
            <span style="color:#696969; ">#-------------------------------------------------------------------</span>
            <span style="color:#696969; "># Append this resized new mask image to the combined mask image:</span>
            <span style="color:#696969; ">#-------------------------------------------------------------------</span>
            <span style="color:#696969; "># - This is done by taking the max() operator between the 2 masks</span>
            <span style="color:#696969; ">#-------------------------------------------------------------------</span>
            mask <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>maximum<span style="color:#808030; ">(</span>mask<span style="color:#808030; ">,</span> mask_<span style="color:#808030; ">)</span>  
        <span style="color:#696969; ">#-----------------------------------------------------------------------</span>
        <span style="color:#696969; "># Store the combined cell nuclei mask:</span>
        <span style="color:#696969; ">#-----------------------------------------------------------------------</span>
        Y_train<span style="color:#808030; ">[</span>n<span style="color:#808030; ">]</span> <span style="color:#808030; ">=</span> mask 
        <span style="color:#696969; ">#-----------------------------------------------------------------------</span>
        <span style="color:#696969; "># Save the combined mask to file</span>
        <span style="color:#696969; ">#-----------------------------------------------------------------------</span>
        <span style="color:#696969; "># the full-path file</span>
        combined_mask_full_path_file_name <span style="color:#808030; ">=</span> path <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">'/masks/'</span> <span style="color:#44aadd; ">+</span> COMBINED_MASK_FILE_NAME
        <span style="color:#696969; "># save the combined mask image</span>
        imsave<span style="color:#808030; ">(</span>combined_mask_full_path_file_name<span style="color:#808030; ">,</span> np<span style="color:#808030; ">.</span>uint8<span style="color:#808030; ">(</span>mask<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span> 
    <span style="color:#696969; ">#-----------------------------------------------------------------------------</span>
    <span style="color:#696969; "># If the masks have already been combined into a single common mask:</span>
    <span style="color:#696969; ">#-----------------------------------------------------------------------------</span>
    <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span> 
        <span style="color:#696969; ">#-----------------------------------------------------------------------</span>
        <span style="color:#696969; "># Read the combined mask</span>
        <span style="color:#696969; ">#-----------------------------------------------------------------------</span>
        <span style="color:#696969; "># the full-path file</span>
        combined_mask_full_path_file_name <span style="color:#808030; ">=</span> path <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">'/masks/'</span> <span style="color:#44aadd; ">+</span> COMBINED_MASK_FILE_NAME
        <span style="color:#696969; "># read the combined mask image</span>
        mask <span style="color:#808030; ">=</span> imread<span style="color:#808030; ">(</span>combined_mask_full_path_file_name<span style="color:#808030; ">)</span> 
        <span style="color:#696969; ">#-----------------------------------------------------------------------</span>
        <span style="color:#696969; "># Store the combined cell nuclei mask:</span>
        <span style="color:#696969; ">#-----------------------------------------------------------------------</span>
        Y_train<span style="color:#808030; ">[</span>n<span style="color:#808030; ">]</span> <span style="color:#808030; ">=</span> mask 

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">Training images and ground-truth masks are read, resized and stored successfully!"</span><span style="color:#808030; ">)</span>
</pre>


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Reading <span style="color:#800000; font-weight:bold; ">and</span> formatting the training images<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The number of train images <span style="color:#808030; ">=</span> <span style="color:#008c00; ">670</span>
</pre>

##### 4.2.1.2. Visualize 8 sample training images:

* Visualize 8 sample training images and their associated cell nuclei masks:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># - Visualize 8 cell nuclei training images and their associated </span>
<span style="color:#696969; ">#   ground-truth cell neclei masks:</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># set the number of skipped images</span>
<span style="color:#696969; "># - integer division</span>
NUM_SKIPPED_IMAGES <span style="color:#808030; ">=</span> num_train_images <span style="color:#44aadd; ">//</span> <span style="color:#008c00; ">8</span>
<span style="color:#696969; "># specify the overall grid size</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span> 
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Cell nuclei test images"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># iterate over the 16 images</span>
<span style="color:#800000; font-weight:bold; ">for</span> i <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span> 
    <span style="color:#696969; "># image counter </span>
    image_counter <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span><span style="color:#400000; ">min</span><span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>i <span style="color:#44aadd; ">*</span> NUM_SKIPPED_IMAGES<span style="color:#808030; ">,</span> num_train_images <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># step 1: create the subplot for the image </span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">4</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">4</span><span style="color:#808030; ">,</span>i<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>  
    <span style="color:#696969; "># display the image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>X_train<span style="color:#808030; ">[</span>image_counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># figure title</span>
    plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Training image #: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>image_counter<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># turn-off axes</span>
    plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># step 2: create the subplot for the image ground-truth cell nuclei mask</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">4</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">4</span><span style="color:#808030; ">,</span>i<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span>  
    <span style="color:#696969; "># display the image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>Y_train<span style="color:#808030; ">[</span>image_counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'gray'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># figure title</span>
    plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Ground-truth: Cells nuclei mask"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># turn-off axes</span>
    plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
</pre>

<img src="images/sample-8-train-images.jpg" width=1000 />

#### 4.2.2. Read, resize and visualize the test data subset:

* There are 65 test images
* Test images do not have ground-truth segmentation masks.

##### 4.2.2.1. Read and resize the test images:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># The structure of the test-data folder is as follows:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - For each test image: </span>
<span style="color:#696969; ">#</span>
<span style="color:#696969; ">#  ...\stage1_test\0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732\images</span>
<span style="color:#696969; ">#</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># This needs to be parsed to extract the the ID for each image:</span>
<span style="color:#696969; ">#</span>
<span style="color:#696969; ">#    0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732</span>
<span style="color:#696969; ">#</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">-------------------------------------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Reading and formatting the test images:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"-------------------------------------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># extract the IDs for the test images</span>
test_ids <span style="color:#808030; ">=</span> next<span style="color:#808030; ">(</span>os<span style="color:#808030; ">.</span>walk<span style="color:#808030; ">(</span>TEST_IMAGES_PATH<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span>

<span style="color:#696969; "># the number of test images</span>
num_test_images <span style="color:#808030; ">=</span> <span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>test_ids<span style="color:#808030; ">)</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"The number of test images = {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>num_test_images<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Allocate a data structure to store the read and resized test images: </span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 4D numpy array to store:</span>
<span style="color:#696969; ">#   - The number of test images: (len(test_ids)</span>
<span style="color:#696969; ">#   - Each test image is: INPUT_IMAGE_HEIGHT x INPUT_IMAGE_WIDTH x INPUT_IMAGE_CHANNELS</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
X_test <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>zeros<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>test_ids<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                   INPUT_IMAGE_HEIGHT<span style="color:#808030; ">,</span> 
                   INPUT_IMAGE_WIDTH<span style="color:#808030; ">,</span> 
                   INPUT_IMAGE_CHANNELS<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                  dtype<span style="color:#808030; ">=</span>np<span style="color:#808030; ">.</span>uint8<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Allocate a structure to store the original size of the test images</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
sizes_test <span style="color:#808030; ">=</span> <span style="color:#808030; ">[</span><span style="color:#808030; ">]</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Read and resize each test image</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># iterate over the test-images IDs</span>
<span style="color:#800000; font-weight:bold; ">for</span> n<span style="color:#808030; ">,</span> id_ <span style="color:#800000; font-weight:bold; ">in</span> tqdm<span style="color:#808030; ">(</span><span style="color:#400000; ">enumerate</span><span style="color:#808030; ">(</span>test_ids<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> total<span style="color:#808030; ">=</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>test_ids<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#696969; "># set the path of the next test image</span>
    path <span style="color:#808030; ">=</span> TEST_IMAGES_PATH <span style="color:#44aadd; ">+</span> id_
    <span style="color:#696969; "># read the next test image</span>
    img <span style="color:#808030; ">=</span> imread<span style="color:#808030; ">(</span>path <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">'/images/'</span> <span style="color:#44aadd; ">+</span> id_ <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">'.png'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span><span style="color:#808030; ">:</span>INPUT_IMAGE_CHANNELS<span style="color:#808030; ">]</span>
    <span style="color:#696969; "># store its original size and append ut to the sizes_test list</span>
    sizes_test<span style="color:#808030; ">.</span>append<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>img<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> img<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># resize the read test image to the specified desired size:</span>
    <span style="color:#696969; "># - INPUT_IMAGE_HEIGHT x INPUT_IMAGE_WIDTH</span>
    img <span style="color:#808030; ">=</span> resize<span style="color:#808030; ">(</span>img<span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span>INPUT_IMAGE_HEIGHT<span style="color:#808030; ">,</span> INPUT_IMAGE_WIDTH<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> mode<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'constant'</span><span style="color:#808030; ">,</span> preserve_range<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># store the resized test image in the X_test 4D arrays </span>
    X_test<span style="color:#808030; ">[</span>n<span style="color:#808030; ">]</span> <span style="color:#808030; ">=</span> img

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">Test images read, resized and stored successfully!"</span><span style="color:#808030; ">)</span>
</pre>

##### 4.2.2.2. Visualize 16 sample test images:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># - Visualize 16 cell nuclei test images:</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># set the number of skipped images</span>
<span style="color:#696969; "># - integer division</span>
NUM_SKIPPED_IMAGES <span style="color:#808030; ">=</span> num_test_images <span style="color:#44aadd; ">//</span> <span style="color:#008c00; ">16</span>
<span style="color:#696969; "># specify the overall grid size</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span> 
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Cell nuclei test images"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># iterate over the 16 images</span>
<span style="color:#800000; font-weight:bold; ">for</span> i <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#696969; "># image counter </span>
    image_counter <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span><span style="color:#400000; ">min</span><span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>i <span style="color:#44aadd; ">*</span> NUM_SKIPPED_IMAGES<span style="color:#808030; ">,</span> num_test_images <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># create the subplot for the next image</span>
    plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">4</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">4</span><span style="color:#808030; ">,</span>i<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>   
    <span style="color:#696969; "># display the image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>X_test<span style="color:#808030; ">[</span>image_counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># figure title</span>
    plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Test image #: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>image_counter<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># set axis off</span>
    plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
</pre>

<img src="images/sample-16-test-images.jpg" width=1000 />

### 4.3. Step 3: Build the U-Net model:

* Build the U-Net model:
  * A sequence of convolutional and pooling layers
  * With some some normalization and dropout layers in between
  * Experiment with different structures and hyper parameters

#### 4.3.1. Display the structure of the implemented U-Net model:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Display the structure of the implemented U-Net:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># set the image dimensions and preserve its aspect-ratio:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># height</span>
img_height <span style="color:#808030; ">=</span> <span style="color:#008c00; ">400</span>
<span style="color:#696969; "># width</span>
img_width <span style="color:#808030; ">=</span> <span style="color:#008c00; ">600</span>
<span style="color:#696969; "># display the image</span>
Image<span style="color:#808030; ">(</span>filename <span style="color:#808030; ">=</span> SAMPLE_IMAGES_PATH <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">"Implemented-U-Net.JPG"</span><span style="color:#808030; ">,</span> width<span style="color:#808030; ">=</span>img_width<span style="color:#808030; ">,</span> height<span style="color:#808030; ">=</span>img_height<span style="color:#808030; ">)</span>
</pre>

<img src="images/U-Net-Structure.PNG" width=1000 />

#### 4.3.2. Define the U-Net model layers:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Define sequential layers of the U-Net model:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 1) Input layer with image size: </span>
<span style="color:#696969; ">#    - INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_CHANNELS</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
inputs <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span><span style="color:#400000; ">Input</span><span style="color:#808030; ">(</span><span style="color:#808030; ">(</span>INPUT_IMAGE_HEIGHT<span style="color:#808030; ">,</span> INPUT_IMAGE_WIDTH<span style="color:#808030; ">,</span> INPUT_IMAGE_CHANNELS<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># normalize the input image to the interval: [0,1]</span>
s <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span><span style="color:#800000; font-weight:bold; ">Lambda</span><span style="color:#808030; ">(</span><span style="color:#800000; font-weight:bold; ">lambda</span> x<span style="color:#808030; ">:</span> x <span style="color:#44aadd; ">/</span> <span style="color:#008c00; ">255</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>inputs<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 2) Contaction path:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 2.1) Convolutional layers: C1 and P1</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Convolution layer with 32 filters of size: 3x3 and preserve image size </span>
c1 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>s<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Apply 10 % dropout</span>
c1 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dropout<span style="color:#808030; ">(</span><span style="color:#008000; ">0.1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c1<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Convolution layer with 16 filters of size: 3x3 and preserve image size </span>
c1 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c1<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Apply 2x2 max-pooling</span>
p1 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>MaxPooling2D<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c1<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 2.2) Convolutional layers: C2 and P2</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Convolution layer with 32 filters of size: 3x3 and preserve image size</span>
c2 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>p1<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Apply 10 % dropout</span>
c2 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dropout<span style="color:#808030; ">(</span><span style="color:#008000; ">0.1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c2<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Convolution layer with 32 filters of size: 3x3 and preserve image size</span>
c2 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c2<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Apply 2x2 max-pooling</span>
p2 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>MaxPooling2D<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c2<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 2.3) Convolutional layers: C3 and P3</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Convolution layer with 64 filters of size: 3x3 and preserve image size</span>
c3 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>p2<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Apply 20 % dropout</span>
c3 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dropout<span style="color:#808030; ">(</span><span style="color:#008000; ">0.2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c3<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Convolution layer with 64 filters of size: 3x3 and preserve image size</span>
c3 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c3<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Apply 2x2 max-pooling</span>
p3 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>MaxPooling2D<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c3<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 2.4) Convolutional layers: C4 and P4</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Convolution layer with 128 filters of size: 3x3 and preserve image size</span>
c4 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>p3<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Apply 20 % dropout</span>
c4 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dropout<span style="color:#808030; ">(</span><span style="color:#008000; ">0.2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c4<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Convolution layer with 128 filters of size: 3x3 and preserve image size</span>
c4 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c4<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Apply 2x2 max-pooling</span>
p4 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>MaxPooling2D<span style="color:#808030; ">(</span>pool_size<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c4<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 2.4) Convolutional layers: C5 and P5</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Convolution layer with 256 filters of size: 3x3 and preserve image size</span>
c5 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">256</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>p4<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Apply 20 % dropout</span>
c5 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dropout<span style="color:#808030; ">(</span><span style="color:#008000; ">0.3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c5<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Convolution layer with 256 filters of size: 3x3 and preserve image size</span>
c5 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">256</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c5<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 3) Expansive path:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 3.1) Convolutional layers: U6 and C6</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Convolution layer with 128 filters of size: 2x2 and preserve image size</span>
u6 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2DTranspose<span style="color:#808030; ">(</span><span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> strides<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c5<span style="color:#808030; ">)</span>
<span style="color:#696969; "># concatenate to up-sample</span>
u6 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>concatenate<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>u6<span style="color:#808030; ">,</span> c4<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># Convolution layer with 128 filters of size: 3x3 and preserve image size</span>
c6 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>u6<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Apply 20 % dropout</span>
c6 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dropout<span style="color:#808030; ">(</span><span style="color:#008000; ">0.2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c6<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Convolution layer with 128 filters of size: 3x3 and preserve image size</span>
c6 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c6<span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 3.2) Convolutional layers: U7 and C7</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
u7 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2DTranspose<span style="color:#808030; ">(</span><span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> strides<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c6<span style="color:#808030; ">)</span>
<span style="color:#696969; "># concatenate to up-sample</span>
u7 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>concatenate<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>u7<span style="color:#808030; ">,</span> c3<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># Convolution layer with 64 filters of size: 3x3 and preserve image size</span>
c7 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>u7<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Apply 20 % dropout</span>
c7 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dropout<span style="color:#808030; ">(</span><span style="color:#008000; ">0.2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c7<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Convolution layer with 64 filters of size: 3x3 and preserve image size</span>
c7 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c7<span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 3.3) Convolutional layers: U8 and C8</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
u8 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2DTranspose<span style="color:#808030; ">(</span><span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> strides<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c7<span style="color:#808030; ">)</span>
<span style="color:#696969; "># concatenate to up-sample</span>
u8 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>concatenate<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>u8<span style="color:#808030; ">,</span> c2<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># Convolution layer with 32 filters of size: 3x3 and preserve image size</span>
c8 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>u8<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Apply 10 % dropout</span>
c8 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dropout<span style="color:#808030; ">(</span><span style="color:#008000; ">0.1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c8<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Convolution layer with 32 filters of size: 3x3 and preserve image size</span>
c8 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c8<span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 3.4) Convolutional layers: U9 and C9</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Convolution layer with 16 filters of size: 1x2 and preserve image size</span>
u9 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2DTranspose<span style="color:#808030; ">(</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> strides<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c8<span style="color:#808030; ">)</span>
<span style="color:#696969; "># concatenate to up-sample</span>
u9 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>concatenate<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>u9<span style="color:#808030; ">,</span> c1<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> axis<span style="color:#808030; ">=</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># Convolution layer with 16 filters of size: 3x3 and preserve image size</span>
c9 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>u9<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Apply 10 % dropout</span>
c9 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dropout<span style="color:#808030; ">(</span><span style="color:#008000; ">0.1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c9<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Convolution layer with 16 filters of size: 3x3 and preserve image size</span>
c9 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_normal'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c9<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 4) Output layer:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># final output layer </span>
outputs <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'sigmoid'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>c9<span style="color:#808030; ">)</span>
</pre>

#### 4.3.3. Construct the Keras model using the above defined layers:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Define the Keras model using the above defined layers:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>Model<span style="color:#808030; ">(</span>inputs<span style="color:#808030; ">=</span><span style="color:#808030; ">[</span>inputs<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> outputs<span style="color:#808030; ">=</span><span style="color:#808030; ">[</span>outputs<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
</pre>

#### 4.3.4. Compile the CNN model:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Compile the model</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; ">#  - Experiment with using:</span>
<span style="color:#696969; ">#      - binary_crossentropy: suitable for binary classification </span>
<span style="color:#696969; ">#      - categorical_crossentropy: suitable for multi-class classification </span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model<span style="color:#808030; ">.</span><span style="color:#400000; ">compile</span><span style="color:#808030; ">(</span>optimizer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'adam'</span><span style="color:#808030; ">,</span> 
              loss<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'binary_crossentropy'</span><span style="color:#808030; ">,</span> 
              metrics<span style="color:#808030; ">=</span><span style="color:#808030; ">[</span><span style="color:#0000e6; ">'accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
</pre>

#### 4.3.5. Print the model summary:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Printout the model summary</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># print model summary</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span>model<span style="color:#808030; ">.</span>summary<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Model<span style="color:#808030; ">:</span> <span style="color:#0000e6; ">"model"</span>
__________________________________________________________________________________________________
Layer <span style="color:#808030; ">(</span><span style="color:#400000; ">type</span><span style="color:#808030; ">)</span>                    Output Shape         Param <span style="color:#696969; ">#     Connected to                     </span>
<span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span>
input_1 <span style="color:#808030; ">(</span>InputLayer<span style="color:#808030; ">)</span>            <span style="color:#808030; ">[</span><span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span> <span style="color:#008c00; ">0</span>                                            
__________________________________________________________________________________________________
<span style="color:#800000; font-weight:bold; ">lambda</span> <span style="color:#808030; ">(</span><span style="color:#800000; font-weight:bold; ">Lambda</span><span style="color:#808030; ">)</span>                 <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span>  <span style="color:#008c00; ">0</span>           input_1<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                    
__________________________________________________________________________________________________
conv2d <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>                 <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span> <span style="color:#008c00; ">448</span>         <span style="color:#800000; font-weight:bold; ">lambda</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                     
__________________________________________________________________________________________________
dropout <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>               <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span> <span style="color:#008c00; ">0</span>           conv2d<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                     
__________________________________________________________________________________________________
conv2d_1 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>               <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span> <span style="color:#008c00; ">2320</span>        dropout<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                    
__________________________________________________________________________________________________
max_pooling2d <span style="color:#808030; ">(</span>MaxPooling2D<span style="color:#808030; ">)</span>    <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">0</span>           conv2d_1<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                   
__________________________________________________________________________________________________
conv2d_2 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>               <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">4640</span>        max_pooling2d<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>              
__________________________________________________________________________________________________
dropout_1 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>             <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">0</span>           conv2d_2<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                   
__________________________________________________________________________________________________
conv2d_3 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>               <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">9248</span>        dropout_1<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
max_pooling2d_1 <span style="color:#808030; ">(</span>MaxPooling2D<span style="color:#808030; ">)</span>  <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">0</span>           conv2d_3<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                   
__________________________________________________________________________________________________
conv2d_4 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>               <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">18496</span>       max_pooling2d_1<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>            
__________________________________________________________________________________________________
dropout_2 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>             <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">0</span>           conv2d_4<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                   
__________________________________________________________________________________________________
conv2d_5 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>               <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">36928</span>       dropout_2<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
max_pooling2d_2 <span style="color:#808030; ">(</span>MaxPooling2D<span style="color:#808030; ">)</span>  <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">0</span>           conv2d_5<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                   
__________________________________________________________________________________________________
conv2d_6 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>               <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>  <span style="color:#008c00; ">73856</span>       max_pooling2d_2<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>            
__________________________________________________________________________________________________
dropout_3 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>             <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>  <span style="color:#008c00; ">0</span>           conv2d_6<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                   
__________________________________________________________________________________________________
conv2d_7 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>               <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>  <span style="color:#008c00; ">147584</span>      dropout_3<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
max_pooling2d_3 <span style="color:#808030; ">(</span>MaxPooling2D<span style="color:#808030; ">)</span>  <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>    <span style="color:#008c00; ">0</span>           conv2d_7<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                   
__________________________________________________________________________________________________
conv2d_8 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>               <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">256</span><span style="color:#808030; ">)</span>    <span style="color:#008c00; ">295168</span>      max_pooling2d_3<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>            
__________________________________________________________________________________________________
dropout_4 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>             <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">256</span><span style="color:#808030; ">)</span>    <span style="color:#008c00; ">0</span>           conv2d_8<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                   
__________________________________________________________________________________________________
conv2d_9 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>               <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">256</span><span style="color:#808030; ">)</span>    <span style="color:#008c00; ">590080</span>      dropout_4<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
conv2d_transpose <span style="color:#808030; ">(</span>Conv2DTranspo <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>  <span style="color:#008c00; ">131200</span>      conv2d_9<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                   
__________________________________________________________________________________________________
concatenate <span style="color:#808030; ">(</span>Concatenate<span style="color:#808030; ">)</span>       <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">256</span><span style="color:#808030; ">)</span>  <span style="color:#008c00; ">0</span>           conv2d_transpose<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>           
                                                                 conv2d_7<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                   
__________________________________________________________________________________________________
conv2d_10 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>  <span style="color:#008c00; ">295040</span>      concatenate<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                
__________________________________________________________________________________________________
dropout_5 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>             <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>  <span style="color:#008c00; ">0</span>           conv2d_10<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
conv2d_11 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>  <span style="color:#008c00; ">147584</span>      dropout_5<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
conv2d_transpose_1 <span style="color:#808030; ">(</span>Conv2DTrans <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">32832</span>       conv2d_11<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
concatenate_1 <span style="color:#808030; ">(</span>Concatenate<span style="color:#808030; ">)</span>     <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>  <span style="color:#008c00; ">0</span>           conv2d_transpose_1<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>         
                                                                 conv2d_5<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                   
__________________________________________________________________________________________________
conv2d_12 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">73792</span>       concatenate_1<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>              
__________________________________________________________________________________________________
dropout_6 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>             <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">0</span>           conv2d_12<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
conv2d_13 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">36928</span>       dropout_6<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
conv2d_transpose_2 <span style="color:#808030; ">(</span>Conv2DTrans <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">8224</span>        conv2d_13<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
concatenate_2 <span style="color:#808030; ">(</span>Concatenate<span style="color:#808030; ">)</span>     <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">0</span>           conv2d_transpose_2<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>         
                                                                 conv2d_3<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                   
__________________________________________________________________________________________________
conv2d_14 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">18464</span>       concatenate_2<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>              
__________________________________________________________________________________________________
dropout_7 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>             <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">0</span>           conv2d_14<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
conv2d_15 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>   <span style="color:#008c00; ">9248</span>        dropout_7<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
conv2d_transpose_3 <span style="color:#808030; ">(</span>Conv2DTrans <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span> <span style="color:#008c00; ">2064</span>        conv2d_15<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
concatenate_3 <span style="color:#808030; ">(</span>Concatenate<span style="color:#808030; ">)</span>     <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span> <span style="color:#008c00; ">0</span>           conv2d_transpose_3<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>         
                                                                 conv2d_1<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                   
__________________________________________________________________________________________________
conv2d_16 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span> <span style="color:#008c00; ">4624</span>        concatenate_3<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>              
__________________________________________________________________________________________________
dropout_8 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>             <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span> <span style="color:#008c00; ">0</span>           conv2d_16<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
conv2d_17 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span> <span style="color:#008c00; ">2320</span>        dropout_8<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
__________________________________________________________________________________________________
conv2d_18 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>  <span style="color:#008c00; ">17</span>          conv2d_17<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>                  
<span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span><span style="color:#44aadd; ">=</span>
Total params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">941</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">105</span>
Trainable params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">941</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">105</span>
Non<span style="color:#44aadd; ">-</span>trainable params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">0</span>
__________________________________________________________________________________________________
<span style="color:#074726; ">None</span>
</pre>

### 4.4. Step 4: Fit/train the model:

* Train the model on the training data set

#### 4.4.1. Define callbacks

* Define callbacks for early for:
    * Early stopping
    * Monitoring training

##### 4.4.1.1. Saving the model trained model checkpoint in case of failure or early termination:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Save the best trained model checkpoint</span>
checkpointer <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>callbacks<span style="color:#808030; ">.</span>
ModelCheckpoint<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'cell_nuclei_segmentation_u-net_model_temp.h5'</span><span style="color:#808030; ">,</span> verbose<span style="color:#808030; ">=</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> save_best_only<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span>
</pre>

##### 4.4.1.2. Early stopping and TensorBoard monitoring:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Stop training if validation accuracy does not improved after 2 consecutive epochs</span>
<span style="color:#696969; "># - Save files in logs for tensorboard monitoring</span>
callbacks <span style="color:#808030; ">=</span> <span style="color:#808030; ">[</span>
        tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>callbacks<span style="color:#808030; ">.</span>EarlyStopping<span style="color:#808030; ">(</span>patience<span style="color:#808030; ">=</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> monitor<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'val_loss'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>
        tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>callbacks<span style="color:#808030; ">.</span>TensorBoard<span style="color:#808030; ">(</span>log_dir<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'logs'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">]</span>
</pre>

##### 4.4.1.3. Start training the model:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Train the model for the specified number of training epochs:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Set the number of training epochs</span>
num_training_epochs <span style="color:#808030; ">=</span> <span style="color:#008c00; ">25</span><span style="color:#808030; ">;</span>
<span style="color:#696969; "># start training the model</span>
results <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>fit<span style="color:#808030; ">(</span>X_train<span style="color:#808030; ">,</span>                    <span style="color:#696969; "># resized training images</span>
                    Y_train<span style="color:#808030; ">,</span>                    <span style="color:#696969; "># resized training masks</span>
                    validation_split<span style="color:#808030; ">=</span><span style="color:#008000; ">0.1</span><span style="color:#808030; ">,</span>       <span style="color:#696969; "># fraction of training data used for model validation</span>
                    batch_size<span style="color:#808030; ">=</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span>              <span style="color:#696969; "># batch size</span>
                    epochs<span style="color:#808030; ">=</span>num_training_epochs<span style="color:#808030; ">,</span> <span style="color:#696969; "># the number of training epochs</span>
                    verbose<span style="color:#808030; ">=</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span>                  <span style="color:#696969; "># verbose: level of logging details</span>
                    callbacks<span style="color:#808030; ">=</span>callbacks<span style="color:#808030; ">)</span>        <span style="color:#696969; "># callbacks functions</span>


Epoch <span style="color:#008c00; ">1</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">90</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.5364</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8034</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3241</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8323</span>
Epoch <span style="color:#008c00; ">2</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">87</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.2436</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8864</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1690</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9367</span>
Epoch <span style="color:#008c00; ">3</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">87</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1513</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9442</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1421</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9494</span>
Epoch <span style="color:#008c00; ">4</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">87</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1285</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9510</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1195</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9549</span>
Epoch <span style="color:#008c00; ">5</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">87</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1205</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9537</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1089</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9581</span>
Epoch <span style="color:#008c00; ">6</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">87</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1147</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9570</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1187</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9559</span>
Epoch <span style="color:#008c00; ">7</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">87</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1089</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9595</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1021</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9605</span>
Epoch <span style="color:#008c00; ">8</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">88</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0996</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9625</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1053</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9607</span>
Epoch <span style="color:#008c00; ">9</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">88</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1018</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9613</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1016</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9618</span>
Epoch <span style="color:#008c00; ">10</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">88</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0952</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9637</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0965</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9636</span>
Epoch <span style="color:#008c00; ">11</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">87</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0928</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9646</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0952</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9634</span>
Epoch <span style="color:#008c00; ">12</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">88</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0899</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9658</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0923</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9646</span>
Epoch <span style="color:#008c00; ">13</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">88</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0898</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9657</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0885</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9656</span>
Epoch <span style="color:#008c00; ">14</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">88</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0864</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9670</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1058</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9606</span>
Epoch <span style="color:#008c00; ">15</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">38</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">38</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">88</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0893</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9658</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0896</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9658</span>
</pre>

### 4.5. Step 5: Evaluate the model:

* Visualize some of the predictions of the trained U-Net model on the subset:

  * Training data subset
  * Validation data subset
  * Testing data subset

##### 4.5.1. Visualize the predictions of the trained U-Net model for 5 randomly selected training images:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Recall we used 10% of the training data for validation:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - So 90% o fthe training data subset was used fro training the model</span>
<span style="color:#696969; "># - It is assumed to be the first 90% images of the training data subset are </span>
<span style="color:#696969; ">#   used for training the U-Net model.</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># define the last training image index</span>
last_train_image_index <span style="color:#808030; ">=</span> <span style="color:#400000; ">int</span><span style="color:#808030; ">(</span>X_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#44aadd; ">*</span><span style="color:#008000; ">0.9</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Compute the model predictions for all images ued for training:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; ">#  - with indices: 0 to last_train_image_index:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
train_preds <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>predict<span style="color:#808030; ">(</span>X_train<span style="color:#808030; ">[</span><span style="color:#808030; ">:</span>last_train_image_index<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> verbose<span style="color:#808030; ">=</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Apply a threshold to convert the mask to binary:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - values &gt; 0.5 are assumed to belong to the mask</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
train_preds <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span>train_preds <span style="color:#44aadd; ">&gt;</span> <span style="color:#008000; ">0.5</span><span style="color:#808030; ">)</span><span style="color:#808030; ">.</span>astype<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>uint8<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - Visualize 5 cell nuclei training images and their associated </span>
<span style="color:#696969; ">#   ground-truth cell nuclei masks as well as their predicted masks:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># specify the overall grid size</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span> 
<span style="color:#696969; "># the plot title</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"U-Net segmentation of sample training images"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># iterate over the 15 images</span>
<span style="color:#800000; font-weight:bold; ">for</span> i <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">15</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span> 
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># Training image counter:</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># - generate a random index between 0 and last_train_image_index</span>
    image_counter <span style="color:#808030; ">=</span> random<span style="color:#808030; ">.</span>randint<span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span> last_train_image_index<span style="color:#808030; ">)</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># step 1: create the subplot for the next image</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span>i<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>  
    <span style="color:#696969; "># display the image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>X_train<span style="color:#808030; ">[</span>image_counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># figure title</span>
    plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Training image #: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>image_counter<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># turn-off axes</span>
    plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># step 2: create the subplot for the image ground-truth nuclei mask</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span>i<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span>  
    <span style="color:#696969; "># display the image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>Y_train<span style="color:#808030; ">[</span>image_counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'gray'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># figure title</span>
    plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Ground-truth segmentation: cell nuclei mask"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># turn-off axes</span>
    plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># step 3: create the subplot for the image predicted nuclei mask</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span>i<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span>  
    <span style="color:#696969; "># display the image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>train_preds<span style="color:#808030; ">[</span>image_counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'gray'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># figure title</span>
    plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"U-Net segmentation: cell nuclei mask"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># turn-off axes</span>
    plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
</pre>


<img src="images/U-Net-performance-5-sample-train-images.JPG" width=1000 />

#### 4.5.2. Visualize the predictions of the trained U-Net model for 5 randomly selected validation images:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Recall we used 10% of the training data for validation:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - It is assumed that the last 10% images of the training data subset are </span>
<span style="color:#696969; ">#   used for validation.</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># define the index of the first validation image</span>
first_valid_image_index <span style="color:#808030; ">=</span> <span style="color:#400000; ">int</span><span style="color:#808030; ">(</span>X_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#44aadd; ">*</span><span style="color:#008000; ">0.9</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Compute the model predictions for all images used for validation:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># with indices: first_valid_image_index to end</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
valid_preds <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>predict<span style="color:#808030; ">(</span>X_train<span style="color:#808030; ">[</span>first_valid_image_index<span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> verbose<span style="color:#808030; ">=</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Apply a threshold to convert the mask to binary:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - values &gt; 0.5 are assumed to belong to the mask</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
valid_preds <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span>valid_preds <span style="color:#44aadd; ">&gt;</span> <span style="color:#008000; ">0.5</span><span style="color:#808030; ">)</span><span style="color:#808030; ">.</span>astype<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>uint8<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - Visualize 5 cell nuclei validation images and their associated </span>
<span style="color:#696969; ">#   ground-truth cell nuclei masks as well as their predicted masks:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># specify the overall grid size</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span> 
<span style="color:#696969; "># create the figure title</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"U-Net segmentation of sample validation images"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># iterate over the 15 images</span>
<span style="color:#800000; font-weight:bold; ">for</span> i <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">15</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span> 
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># Validation image counter:</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># - generate a random index between first_valid_image_index and num_train_images</span>
    image_counter <span style="color:#808030; ">=</span> random<span style="color:#808030; ">.</span>randint<span style="color:#808030; ">(</span>first_valid_image_index<span style="color:#808030; ">,</span> num_train_images<span style="color:#808030; ">)</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># step 1: create the subplot for the next image</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span>i<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>  
    <span style="color:#696969; "># display the image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>X_train<span style="color:#808030; ">[</span>image_counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># figure title</span>
    plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Training image #: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>image_counter<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># turn-off axes</span>
    plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># step 2: create the subplot for the image ground-truth nuclei mask</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span>i<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span>  
    <span style="color:#696969; "># display the image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>Y_train<span style="color:#808030; ">[</span>image_counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'gray'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># figure title</span>
    plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Ground-truth segmentation: cell nuclei mask"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># turn-off axes</span>
    plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># step 3: create the subplot for the image predicted nuclei mask</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span>i<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span>  
    <span style="color:#696969; "># display the image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>valid_preds<span style="color:#808030; ">[</span>image_counter <span style="color:#44aadd; ">%</span> first_valid_image_index<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'gray'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># figure title</span>
    plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"U-Net segmentation: cell nuclei mask"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># turn-off axes</span>
    plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
</pre>

<img src="images/U-Net-performance-5-sample-validation-images.JPG" width=1000 />


#### 4.5.3. Visualize the predictions of the trained U-Net model for 5 randomly selected test images:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Compute the model predictions for all test images:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
test_preds <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>predict<span style="color:#808030; ">(</span>X_test<span style="color:#808030; ">,</span> verbose<span style="color:#808030; ">=</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Apply a threshold to convert the mask to binary:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - values &gt; 0.5 are assumed to belong to the mask</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
test_preds <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span>test_preds <span style="color:#44aadd; ">&gt;</span> <span style="color:#008000; ">0.5</span><span style="color:#808030; ">)</span><span style="color:#808030; ">.</span>astype<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>uint8<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - Visualize 5 cell nuclei test images and their predicted masks:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># test images do not have ground-truth cell nuclei masks</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># specify the overall grid size</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span> 
<span style="color:#696969; "># create the figure title</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"U-Net segmentation of sample validation images"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># iterate over the 10 images</span>
<span style="color:#800000; font-weight:bold; ">for</span> i <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span> 
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># Test image counter:</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># - generate a random index between 0 and num_test_images</span>
    image_counter <span style="color:#808030; ">=</span> random<span style="color:#808030; ">.</span>randint<span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span> num_test_images <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># step 1: create the subplot for the next image</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span>i<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>  
    <span style="color:#696969; "># display the image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>X_test<span style="color:#808030; ">[</span>image_counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># figure title</span>
    plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Training image #: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>image_counter<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># turn-off axes</span>
    plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    <span style="color:#696969; "># step 2: create the subplot for the image predicted nuclei mask</span>
    <span style="color:#696969; ">#---------------------------------------------------------------------------</span>
    plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span>i<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span>  
    <span style="color:#696969; "># display the image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>test_preds<span style="color:#808030; ">[</span>image_counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'gray'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># figure title</span>
    plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"U-Net segmentation: cell nuclei mask"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># turn-off axes</span>
    plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
</pre>


<img src="images/U-Net-performance-5-sample-testing-images.JPG" width=1000 />

### 4.6. Step 6: Save the trained CNN model:

* Save the trained model for future re-use.

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Save the trained model</span>
model<span style="color:#808030; ">.</span>save<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'cell_nuclei_seg_u-net_model_final_num_epochs_'</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>num_training_epochs<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">'.h5'</span><span style="color:#808030; ">)</span>
</pre>


### 4.7. Step 7: End of Execution:

* Display a successful end of execution message:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># display a final message</span>
<span style="color:#696969; "># current time</span>
now <span style="color:#808030; ">=</span> datetime<span style="color:#808030; ">.</span>datetime<span style="color:#808030; ">.</span>now<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>now<span style="color:#808030; ">.</span>strftime<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">"</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
</pre>
<pre style="color:#000000;background:#ffffff;">Program executed successfully on<span style="color:#808030; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">04</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">03</span> <span style="color:#008c00; ">22</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">03</span><span style="color:#808030; ">:</span><span style="color:#008000; ">44.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>Goodbye!
</pre>


## 5. Analysis

* In view of the presented results, we make the following observations: 
    * The trained U-Net model yields significantly accurate segmentation masks of the cells nuclei, for the training, validation and even the test data subsets.


## 6. Future Work

* We propose to explore the following related issues:

  * Since the test data subset does not have ground-truth cells' nuclei masks, we propose to:
  * Split the 670 annotated images with ground-truth masks into:
      * Training subset: 80%
      * Testing subset: 20%
  * We can then:
    * Retrain the U-Net model on the smaller training subset
    * Evaluate its performance on the labelled testing subset using quantitative metrics.


7. References

1. Kaggle. 2018 Data Science Bowl: Find the nuclei in divergent images to advance medical discovery.  https://www.kaggle.com/c/data-science-bowl-2018/data
2. Long, F.  Microscopy cell nuclei segmentation with enhanced U-Net. https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3332-1
3. Caicedo J. C. Evaluation of Deep Learning Strategies for Nucleus Segmentation in Fluorescence Images. https://www.biorxiv.org/content/10.1101/335216v3.full
4. Sagar A. Nucleus Segmentation using U-Net: How can deep learning be used for segmenting medical images. https://towardsdatascience.com/nucleus-segmentation-using-u-net-eceb14a9ced4
5. Vidhya A. Semantic segmentation to detect Nuclei using U-net Prediction done on the Data Science Bowl 2018 data set using Tensorflow and Keras. https://medium.com/analytics-vidhya/semantic-segmentation-using-u-net-data-science-bowl-2018-data-set-ed046c2004a5
6. Kaggle. Keras U-net for Nuclei Segmentation. https://www.kaggle.com/dingli/keras-u-net-for-nuclei-segmentation.
7. Digital Sreeni. Image segmentation using U-Net - Part 1 - 6.  https://www.youtube.com/watch?v=azM57JuQpQI
















