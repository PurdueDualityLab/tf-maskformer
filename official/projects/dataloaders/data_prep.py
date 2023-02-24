import input_reader
import factory_config
from official.common import distribute_utils
from official.projects.dataloaders.distributed_executor import DistributedExecutor

train_input_fn = None
eval_input_fn = None

params = factory_config.config_generator('mask_former')
training_file_pattern = params.train.train_file_pattern
eval_file_pattern = params.eval.eval_file_pattern

if not training_file_pattern and not eval_file_pattern:
    raise ValueError('Must provide at least one of training_file_pattern and '
                     'eval_file_pattern.')

if training_file_pattern:
    # Use global batch size for single host.
    train_input_fn = input_reader.InputFn(
        file_pattern=training_file_pattern,
        params=params,
        mode=input_reader.ModeKeys.TRAIN,
        batch_size=params.train.batch_size)

if eval_file_pattern:
    eval_input_fn = input_reader.InputFn(
        file_pattern=eval_file_pattern,
        params=params,
        mode=input_reader.ModeKeys.PREDICT_WITH_GT,
        batch_size=params.eval.batch_size,
        num_examples=params.eval.eval_samples)
# call it this way to get dataset object
strategy_config = params.strategy_config
distribute_utils.configure_cluster(strategy_config.worker_hosts,
                                   strategy_config.task_index)
strategy = distribute_utils.get_distribution_strategy(
    distribution_strategy=params.strategy_type,
    num_gpus=strategy_config.num_gpus,
    all_reduce_alg=strategy_config.all_reduce_alg,
    num_packs=strategy_config.num_packs,
    tpu_address=strategy_config.tpu)

pain_and_suffering = DistributedExecutor(strategy, params)

iterable_ds = pain_and_suffering.get_input_iterator(strategy, params)
raw_dataset = train_input_fn()
