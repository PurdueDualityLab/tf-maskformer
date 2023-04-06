# MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation

TensorFlow 2 implementation of MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation


export PYTHONPATH=$PYTHONPATH:/depot/qqiu/data/vishal/projects/tf_maskformer_debug/models

## Imp paths
code path - /depot/qqiu/data/vishal/projects/tf_maskformer_debug/models/official/projects/maskformer/ckpts
data path - /depot/davisjam/data/vishal/datasets/coco
## Environment creation 
conda create -n tfmaskformer
conda activate /depot/qqiu/data/vishal/envs/tmaskformer
pip install -r requirements.txt

## Dataset Download and Prep
```
chmod +x ./data/create_tf_records.sh
cd /depot/qqiu/data/vishal/projects/tf_maskformer_integration/official/projects/maskformer/data
./create_tf_records.sh /depot/davisjam/data/vishal/datasets/coco

```
module load cuda/11.7.0 cudnn/cuda-11.7_8.6 gcc/6.3.0


# For debugging the code
python3 official/projects/maskformer/train.py \
  --experiment=maskformer_coco_panoptic \
  --mode=train_and_eval \
  --model_dir=/depot/qqiu/data/vishal/projects/tf_maskformer_debug/models/official/projects/maskformer/ckpts \