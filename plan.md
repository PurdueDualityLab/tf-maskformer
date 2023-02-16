# Plan

1. load the test,train,validation images and their annotations from coco:
    1. INPUT:
        1. train:[http://images.cocodataset.org/zips/train2017.zip](http://images.cocodataset.org/zips/train2017.zip)
        2. val:[http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
        3. test:[http://images.cocodataset.org/zips/test2017.zip](http://images.cocodataset.org/zips/test2017.zip)
        4. annotations:[http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip)
    2. PROCESS:
        1. wget and unzip lol
    3. OUTPUT:
        1. directory structured:
            
            ```c
            .(COCO_ROOT)
            +-- train2017
            |   |
            |   +-- *.jpg
            |
            |-- val2017
            |   |
            |   +-- *.jpg
            |
            |-- test2017
            |   |
            |   +-- *.jpg
            |
            +-- annotations
                 |
                 +-- panoptic_{train|val}2017.json
                 +-- panoptic_{train|val}2017
            ```
            
2. convert COCO dataset to TFRecord file format:
    1. INPUT:
        1. step 1
    2. PROCESS: 
        1. Nice link: [`https://github.com/google-research/deeplab2/blob/main/g3doc/setup/coco.md`](https://github.com/google-research/deeplab2/blob/main/g3doc/setup/coco.md)
        2. [`https://github.com/google-research/deeplab2/blob/main/data/build_coco_data.py`](https://github.com/google-research/deeplab2/blob/main/data/build_coco_data.py) to process the data set
        3. imp constants:
            1. dataset: 
            
            [`https://github.com/google-research/deeplab2/blob/main/data/dataset.py`](https://github.com/google-research/deeplab2/blob/main/data/dataset.py)
            (contains the follwoing 2 constants):
            
             
            
            ```c
            _CLASS_HAS_INSTANCE_LIST = dataset.COCO_PANOPTIC_INFORMATION.class_has_instances_list
            _PANOPTIC_LABEL_DIVISOR = dataset.COCO_PANOPTIC_INFORMATION.panoptic_label_divisor
            ```
            
            1. data_utils: stuff for loading images / other things - nice to c&c
            
            [`https://github.com/google-research/deeplab2/blob/main/data/utils/create_step_panoptic_maps.py`](https://github.com/google-research/deeplab2/blob/main/data/utils/create_step_panoptic_maps.py)
            
            1. coco constants: (meta info of coco dataset)
            
            [`https://github.com/google-research/deeplab2/blob/main/data/coco_constants.py`](https://github.com/google-research/deeplab2/blob/main/data/coco_constants.py)
            
        4. Run the thing via:
            
            ```c
            # For generating data for panoptic segmentation task
            python deeplab2/data/build_coco_data.py \
              --coco_root=${COCO_ROOT} \
              --output_dir=${OUTPUT_DIR}
            ```
            
    3. OUTPUT: a file containing TFRECORDS that are memory efficient
3. augment data, do dataloader and 🙏
    1. INPUT: tfrecords file
    2. PROCESS:
        1. run the [`https://github.com/tensorflow/models/blob/master/official/projects/panoptic/dataloaders/panoptic_deeplab_input.py`](https://github.com/tensorflow/models/blob/master/official/projects/panoptic/dataloaders/panoptic_deeplab_input.py) looping over each image and saving it in a dictionary - yes - create a dictionary for each one of the classes, putting it in the format required for the paper
        2. focus on the `class Parser(parser.Parser):`
            1. Parser to parse an image and its annotations into a dictionary of tensors.
        3. do 1 image at a time? what about batches? - not have any

TEAM TODO:

 1. find ways to test pytorch repo and turn augmentation on and off

1. configure pytorch repo for testing
2. start implementing tensorflow repo and making it link with each part

### Final result:

### Model input:

input to forward method:

batched_inputs: a list, batched outptus of :class:`DatasetMapper`

each item contains inputs for one image

For now, each item in the list is a dict that contains:

- "image": Tensor, image in (C, H, W) format.
- "instances": per-region ground truth
- Other information that's included in the original dicts, such as:
    - "height", "width" (int): the output resolution of the model (may be different from input resolution), used in inference.
    
    TODO:
    
    - [ ]  [https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/data/dataset_mappers/mask_former_panoptic_dataset_mapper.py](https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/data/dataset_mappers/mask_former_panoptic_dataset_mapper.py)

### Misc Notes:

TFExample: standard proto storing data for training and inference

- flexible mesage preprests a {”string”: value} mapping

TFRecord: memory efficient file containing seq of records - only read sequentially

- simple format for storing sequence of binary records

Protocol files: file holds 0 or more values based by type of value

 - tfExamples: need to decode the proto files (generated by tfrecords) into a usable dictionary

### Questions:

- do we want to run this on datasets other than coco.. will be pain if yes?
- How will the process described above fit into the overall project? will we generate all data files and then simply run each one through dataloader?
- 

### uncertainties:

- how are tfrecords stored? Do I loop over them
- how go from step 2 to step 3? is there a way to pass tfrecords directly as tfexamples

Question: do we want this to run on other datasets?