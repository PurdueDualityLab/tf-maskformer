
"""Standard names for input dataloader modes.
The following standard keys are defined:
* `TRAIN`: training mode.
* `EVAL`: evaluation mode.
* `PREDICT`: prediction mode.
* `PREDICT_WITH_GT`: prediction mode with groundtruths in returned variables.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


TRAIN = 'train'
EVAL = 'eval'
PREDICT = 'predict'
PREDICT_WITH_GT = 'predict_with_gt'