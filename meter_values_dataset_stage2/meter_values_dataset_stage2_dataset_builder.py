"""meter_values_dataset_stage2 dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import process_labels_label_studio


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for meter_values_dataset_stage2 dataset."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Some description here
  """

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(meter_values_dataset_stage2): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8),
            'image/filename': tfds.features.Text(),
            'image/id': tf.int64,
            'objects': tfds.features.Sequence({
                'area': tf.int64,
                'bbox': tfds.features.BBoxFeature(),
                'id': tf.int64,
                'is_crowd': tf.bool,
                'label': tfds.features.ClassLabel(num_classes=15),
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
    path = dl_manager.manual_dir #/ 'data.zip'
    # Extract the manually downloaded `data.zip`
    #path = dl_manager.extract(archive_path)

    # TODO(MeterValuesDataset): Returns the Dict[split names, Iterator[Key, Example]]
    partition_train = self.builder_config.partition['train']
    partition_val = self.builder_config.partition['val']
    partition_test = self.builder_config.partition['test']
    images_info = process_labels_label_studio.get_images_info(path / 'labels.json')
    images_info = [im_info for im_info in images_info if len(im_info['annotations']) > 0] 
    max_samples_train = np.floor(len(images_info) * partition_train)
    max_samples_val = np.floor(len(images_info) * partition_val)
    max_samples_test = np.floor(len(images_info) * partition_test)
    self.iter = process_labels_label_studio.generate_examples_stage2(
      images_info, path)
    return {
        'train': self._generate_examples(max_samples_train),
        'validation': self._generate_examples(max_samples_val),
        'test': self._generate_examples(max_samples_test),
    }

  def _generate_examples(self, max_samples):
    """Yields examples."""
    # TODO(meter_values_dataset_stage2): Yields (key, example) tuples from the dataset
    partition = 0.5
    # TODO: change labels to stage 2
    str2int = {
      '0': 0,
      '1': 1,
      '2': 2,
      '3': 3,
      '4': 4,
      '5': 5,
      '6': 6,
      '7': 7,
      '8': 8,
      '9': 9,
      'r': 10,
      't': 11,
      '_': 12,
      'point': 13,
      'floatp': 14
    }

    index = 0
    while index < max_samples:
      
      try:
        im, labels, bboxes, image_filename = next(self.iter)
      except StopIteration:
        return
      index += 1
      
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
        label_features.append(str2int[label.lower()])
      
      yield index, {
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
