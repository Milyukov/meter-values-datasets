import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np

if __name__ == '__main__':
    (train_dataset, val_dataset, test_dataset), dataset_info = tfds.load(
        "meter_values_dataset_stage1", split=["train", "validation", "test"], with_info=True, data_dir=".",
        read_config=tfds.ReadConfig(try_autocache=False)
    )

    #stat = tfds.show_statistics(dataset_info)
    # for val in train_dataset:
    #    pass
    vals = np.fromiter(train_dataset.map(lambda x: x['objects']['label']), float)

    plt.hist(vals)
    plt.xticks(range(4))
    plt.title('Label Frequency')
    plt.savefig('label_dist.png')
