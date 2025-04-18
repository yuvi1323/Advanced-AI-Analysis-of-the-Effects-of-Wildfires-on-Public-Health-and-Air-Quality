import tensorflow as tf
import glob
import os
from utils.tfrecord_parser import parse_tfrecord

def load_dataset(tfrecord_dir, batch_size=32):
    tfrecord_files = glob.glob(os.path.join(tfrecord_dir, "*.tfrecord"))

    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    parsed_dataset = (
        raw_dataset
        .map(parse_tfrecord)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return parsed_dataset