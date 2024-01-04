"""meter_values_dataset_stage2 dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import process_labels_label_studio

from dataclasses import dataclass, field

from collections import OrderedDict
from utils import augment_data_stage2

@dataclass
class MyDatasetConfig(tfds.core.BuilderConfig):
  width: int = 0
  height: int = 0
  partition: dict = field(default_factory= lambda: {
        'train': 0.8,
        'test': 0.1,
        'val': 0.1,
    })

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for meter_values_dataset_stage2 dataset."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Some description here
  """

  DO_AUGMENT = True

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  BUILDER_CONFIGS = [
      # `name` (and optionally `description`) are required for each config
      MyDatasetConfig(name='Default', width=800, height=800, 
                      partition={'train': 0.8, 'test': 0.1, 'val': 0.1}),
  ]

  str2int = OrderedDict([
      ('0', 0),
      ('1', 1),
      ('2', 2),
      ('3', 3),
      ('4', 4),
      ('5', 5),
      ('6', 6),
      ('7', 7),
      ('8', 8),
      ('9', 9),
      ('r', 10),
      ('t', 11),
      ('m', 12),
      ('_', 13),
      ('floatp', 14),
      ('colon', 15),
      ('arrow', 16),
      ('q', 17),
      ('v', 18),
      ('u', 19)
  ])

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(meter_values_dataset_stage2): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(self.builder_config.height, self.builder_config.width, 3), dtype=tf.uint8),
            'image/filename': tfds.features.Text(),
            'image/id': tf.int64,
            'objects': tfds.features.Sequence({
                'area': tf.int64,
                'bbox': tfds.features.BBoxFeature(),
                'id': tf.int64,
                'is_crowd': tf.bool,
                'label': tfds.features.ClassLabel(num_classes=17),
            }),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        #supervised_keys=('image', 'image/filename', 'image/id', 'objects'),  # Set to `None` to disable
        homepage=None,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(meter_values_dataset_stage2): Downloads the data and defines the splits
    path_to_images = dl_manager.manual_dir
    path_to_labels = path_to_images / '..'

    # TODO(MeterValuesDataset): Returns the Dict[split names, Iterator[Key, Example]]
    partition_train = self.builder_config.partition['train']
    partition_val = self.builder_config.partition['val']
    partition_test = self.builder_config.partition['test']
    images_info = process_labels_label_studio.get_images_info(path_to_labels / 'labels.json')
    images_info = [im_info for im_info in images_info if len(im_info['annotations']) > 0]
    number_of_examples = process_labels_label_studio.get_examples_count_stage2(images_info, path_to_images)
    print(f'Number of labeled samples: {number_of_examples}')
    max_samples_train = np.floor(number_of_examples * partition_train)
    max_samples_val = np.floor(number_of_examples * partition_val)
    max_samples_test = np.floor(number_of_examples * partition_test)
    self.iter = process_labels_label_studio.generate_examples_stage2(
      images_info, path_to_images, self.builder_config.width, self.builder_config.height)
    return {
        'train': self._generate_examples(max_samples_train),
        'validation': self._generate_examples(max_samples_val),
        'test': self._generate_examples(max_samples_test),
    }

  def _prepare_output(self, im, labels, bboxes, image_filename, index):
    bbox_features = []
    label_features = []
    areas = []
    height, width, _ = im.shape
    for bbox, label in zip(bboxes, labels):
      bbox_features.append(tfds.features.BBox(
        ymin=bbox[1] / height,
        xmin=bbox[0] / width,
        ymax=(bbox[1] + bbox[3]) / height,
        xmax=(bbox[0] + bbox[2]) / width)
      )
      areas.append(bbox[2] * bbox[3])
      label_features.append(self.str2int[label.lower()])
    output = {
        'image': im.astype(np.uint8),
        'image/filename': image_filename,
        'image/id': index,
        'objects': {
          'area': areas,
          'bbox': bbox_features,
          'id': [0] * len(bbox_features),
          'is_crowd': [False] * len(bbox_features),
          'label': label_features
        }
    }
    return output

  def _generate_examples(self, max_samples):
    """Yields examples."""

    index = 0
    while index < max_samples:
      
      try:
        im, labels, bboxes, image_filename = next(self.iter)
      except StopIteration:
        return
      index += 1
      
      if self.DO_AUGMENT:
        for index in range(2):
          im_augmented, labels_augmented, bboxes_augmented = augment_data_stage2(
            im, labels, bboxes)
          output = self._prepare_output(
            im_augmented, labels_augmented, bboxes_augmented, image_filename, index)
          yield index, output
      output = self._prepare_output(
        im, labels, bboxes, image_filename, index)

      yield index, output
