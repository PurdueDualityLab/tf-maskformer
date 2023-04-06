

module load cuda/11.7.0 cudnn/cuda-11.7_8.6 gcc/6.3.0
export PYTHONPATH=$PYTHONPATH:<path_to_models>

## Environment creation 
```
conda create -n tfmaskformer
conda activate tfmaskformer
pip install -r /models/official/requirements.txt
pip install tensorflow-text-nightly
```

## To start training
```
python3 official/projects/maskformer/train.py \
  --experiment=maskformer_coco_panoptic \
  --mode=train_and_eval \
  --model_dir=<model_dir> \
```