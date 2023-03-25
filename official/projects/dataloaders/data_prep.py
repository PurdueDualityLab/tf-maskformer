from official.projects.dataloaders import input_reader
from official.projects.dataloaders import panoptic_input
import tensorflow as tf

from official.projects.configs import maskformer_cfg
from official.core import exp_factory as factory
from official.common import dataset_fn

def build_inputs(params):
    decoder_cfg = params.decoder
    if decoder_cfg.type == 'simple_decoder':
        decoder = panoptic_input.TfExampleDecoder(
            regenerate_source_id = params.regenerate_source_id)
    else:
        raise ValueError('Unknown decoder type: {}!'.format(
            params.decoder.type))
    
    parser_fn = panoptic_input.mask_former_parser(params.parser,
                                decoder.decode)
    reader = input_reader.InputFn(
        params,
        dataset_fn = dataset_fn.pick_dataset_fn(params.file_type),
        parser_fn = parser_fn)
    print(parser_fn)
    print(reader)
    return reader()

cfg_test = factory.get_exp_config("detr_coco_tfrecord")
print(build_inputs(factory.get_exp_config("detr_coco_tfrecord").task.train_data))