"""meter_values_dataset_stage1 dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import sys
sys.path.append('..')

import process_labels_label_studio
from dataclasses import dataclass, field

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
  """DatasetBuilder for meter_values_dataset_stage1 dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Some description here
  """

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  BUILDER_CONFIGS = [
      # `name` (and optionally `description`) are required for each config
      MyDatasetConfig(name='Default', width=1024, height=1024, 
                      partition={'train': 0.8, 'test': 0.1, 'val': 0.1}),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(meter_values_dataset_stage1): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(1024, 1024, 3), dtype=tf.uint8),
            'image/filename': tfds.features.Text(),
            'image/id': tf.int64,
            'objects': tfds.features.Sequence({
                'area': tf.int64,
                'bbox': tfds.features.Tensor(shape=(12,), dtype=tf.float32),#tfds.features.BBoxFeature(),
                'id': tf.int64,
                'is_crowd': tf.bool,
                'label': tfds.features.ClassLabel(num_classes=4),
            }),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        #supervised_keys=('image', 'image/filename', 'image/id', 'objects'),  # Set to `None` to disable
        homepage=None,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    # TODO(meter_values_dataset): Downloads the data and defines the splits
    # data_path is a pathlib-like `Path('<manual_dir>/data.zip')`
    path_to_images = dl_manager.manual_dir
    path_to_labels = path_to_images / '..'
    # Extract the manually downloaded `data.zip`
    #path = dl_manager.extract(archive_path)

    # TODO(MeterValuesDataset): Returns the Dict[split names, Iterator[Key, Example]]
    width = self.builder_config.width
    height = self.builder_config.height
    partition_train = self.builder_config.partition['train']
    partition_val = self.builder_config.partition['val']
    partition_test = self.builder_config.partition['test']
    images_info = process_labels_label_studio.get_images_info(path_to_labels / 'labels.json')
    images_info = [im_info for im_info in images_info if len(im_info['annotations']) > 0]
    number_of_examples = process_labels_label_studio.get_examples_count_stage1(
      images_info, path_to_images)
    print(f'Number of labeled samples: {number_of_examples}')
    max_samples_train = np.floor(number_of_examples * partition_train)
    max_samples_val = np.floor(number_of_examples * partition_val)
    max_samples_test = np.floor(number_of_examples * partition_test)
    self.iter = process_labels_label_studio.generate_examples_stage1(
      images_info, path_to_images, width, height)
    return {
        'train': self._generate_examples(max_samples_train),
        'validation': self._generate_examples(max_samples_val),
        'test': self._generate_examples(max_samples_test),
    }

  def _generate_examples(self, max_samples):
    """Yields examples."""
    # TODO(meter_values_dataset_stage1): Yields (key, example) tuples from the dataset
    str2int = {
      'analog': 0,
      'digital': 1,
      'analog_illegible': 2,
      'digital_illegible': 3
    }
    width = self.builder_config.width
    height = self.builder_config.height

    index = 0
    while index < max_samples:
      
      try:
        im_resized, label, bbox, keypoints, image_filename = next(self.iter)
      except StopIteration:
        return
      index += 1
      
      final_bbox = np.array([
        bbox[1] / height,
        bbox[0] / width,
        (bbox[1] + bbox[3]) / height,
        (bbox[0] + bbox[2]) / width,
        keypoints[0][1] / height,
        keypoints[0][0] / width,
        keypoints[1][1] / height,
        keypoints[1][0] / width,
        keypoints[2][1] / height,
        keypoints[2][0] / width,
        keypoints[3][1] / height,
        keypoints[3][0] / width,
      ], dtype=np.float32)

      yield index, {
          'image': im_resized.astype(np.uint8),
          'image/filename': image_filename,
          'image/id': index,
          'objects': {
            'area': [bbox[2] * bbox[3]],
            'bbox': [final_bbox],
            'id': [0],
            'is_crowd': [False],
            'label': [str2int[label[0].lower()]]
          }
      }
