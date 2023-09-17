import tensorflow as tf

from official.projects.maskformer.modeling.decoder.transformer_decoder import MaskFormerTransformer
from official.projects.maskformer.modeling.layers.nn_block import MLPHead
from official.projects.maskformer.modeling.decoder.transformer_pixel_decoder import TransformerFPN
from official.projects.maskformer.modeling.decoder.pixel_decoder import CNNFPN
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


class MaskFormer(tf.keras.Model):
	"""Maskformer"""
	def __init__(self,
			  backbone,
			   input_specs,
			   fpn_feat_dims=256,
			   data_format=None,
			   dilation_rate=(1, 1),
			   groups=1,
			   activation='relu',
			   use_bias=False,
			   kernel_initializer="glorot_uniform",
			   bias_initializer="zeros",
			   kernel_regularizer=None,
			   bias_regularizer=None,
			   activity_regularizer=None,
			   kernel_constraint=None,
			   bias_constraint=None,
			   num_queries=100,
			   hidden_size=256,
			   fpn_encoder_layers=6,
			   detr_encoder_layers=0,
			   num_decoder_layers=6,
			   dropout_rate=0.1,
			   backbone_endpoint_name='5',
			   num_classes=199,
			   batch_size=1,
			   bfloat16=False,
			   which_pixel_decoder='fpn',
			   
			   **kwargs):
		super(MaskFormer, self).__init__(**kwargs)
		self._backbone = backbone
		self._input_specs = input_specs
		self._batch_size = batch_size
		self._num_classes = num_classes

		# Pixel Deocder paramters.
		self._fpn_feat_dims = fpn_feat_dims
		self._data_format = data_format
		self._dilation_rate = dilation_rate
		self._groups = groups
		self._activation = activation
		self._use_bias = use_bias
		self._kernel_initializer = kernel_initializer
		self._bias_initializer = bias_initializer
		self._kernel_regularizer = kernel_regularizer
		self._bias_regularizer = bias_regularizer
		self._activity_regularizer = activity_regularizer
		self._kernel_constraint = kernel_constraint
		self._bias_constraint = bias_constraint

		# DETRTransformer parameters.
		self._fpn_encoder_layers = fpn_encoder_layers
		self._detr_encoder_layers = detr_encoder_layers
		self._num_decoder_layers = num_decoder_layers
		self._num_queries = num_queries
		self._hidden_size = hidden_size
		self._dropout_rate = dropout_rate
		if hidden_size % 2 != 0:
			raise ValueError("hidden_size must be a multiple of 2.")
		self._bfloat16 = bfloat16
		self._pixel_decoder = which_pixel_decoder
		
		# Backbone feature extractor.
		self._backbone_endpoint = backbone_endpoint_name
		
		
	def build(self, image_shape = None):
		if self._pixel_decoder == 'transformer_fpn':
			self.pixel_decoder = TransformerFPN(batch_size = self._batch_size,
									fpn_feat_dims=self._fpn_feat_dims,
									data_format=self._data_format,
									dilation_rate=self._dilation_rate,
									groups=self._groups,
									activation=self._activation,
									use_bias=self._use_bias,
									kernel_initializer=self._kernel_initializer,
									bias_initializer=self._bias_initializer,
									kernel_regularizer=self._kernel_regularizer,
									bias_regularizer=self._bias_regularizer,
									activity_regularizer=self._activity_regularizer,
									kernel_constraint=self._kernel_constraint,
									bias_constraint=self._bias_constraint,
									num_encoder_layers = self._fpn_encoder_layers,
									bfloat16=self._bfloat16)
		
		elif self._pixel_decoder == 'fpn':
			# FIXME : Add the input arguments to CNNFPN
			self.pixel_decoder = CNNFPN()
		else:
			raise ValueError("Invalid Pixel Decoder: ", self._pixel_decoder)
		
		self.transformer = MaskFormerTransformer(num_queries=self._num_queries,
												hidden_size=self._hidden_size,
												num_encoder_layers=self._detr_encoder_layers,
												num_decoder_layers=self._num_decoder_layers,
												dropout_rate=self._dropout_rate)
		
		self.head = MLPHead(num_classes=self._num_classes, 
							hidden_dim=self._hidden_size, 
							mask_dim=self._fpn_feat_dims)
		
		super(MaskFormer, self).build(image_shape)
 
	@property
	def backbone(self) -> tf.keras.Model:
		return self._backbone
	
	@property
	def checkpoint_items(self):
		"""Returns a dictionary of items to be additionally checkpointed."""
		items = dict(backbone=self._backbone,
					pixel_decoder=self.pixel_decoder,
					transformer=self.transformer,
					head=self.head)
		return items
	
	def get_config(self):
		return {
			"backbone": self._backbone,
			"backbone_endpoint_name": self._backbone_endpoint_name,
			"num_queries": self._num_queries,
			"hidden_size": self._hidden_size,
			"num_classes": self._num_classes,
			"num_encoder_layers": self._num_encoder_layers,
			"num_decoder_layers": self._num_decoder_layers,
			"dropout_rate": self._dropout_rate,
		}
	
	@classmethod
	def from_config(cls, config):
		return cls(**config)
	
	def process_feature_maps(self, maps):
		new_dict = {}
		for k in maps.keys():
			new_dict[k[0]] = maps[k]
		return new_dict

	def call(self, image, training = False):
		backbone_feature_maps = self._backbone(image)
		backbone_feature_maps_procesed = self.process_feature_maps(backbone_feature_maps)
		if self._pixel_decoder == 'fpn':
			mask_features = self.pixel_decoder(backbone_feature_maps_procesed)
			transformer_enc_feat = backbone_feature_maps_procesed['5']
		elif self._pixel_decoder == 'transformer_fpn':
			mask_features, transformer_enc_feat = self.pixel_decoder(backbone_feature_maps_procesed)
		transformer_features = self.transformer({"features": transformer_enc_feat})
		seg_pred = self.head({"per_pixel_embeddings" : mask_features,
							"per_segment_embeddings": transformer_features})
		return seg_pred
