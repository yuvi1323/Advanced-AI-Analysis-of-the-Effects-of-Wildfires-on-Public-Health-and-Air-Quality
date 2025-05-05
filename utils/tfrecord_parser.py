import tensorflow as tf

def get_feature_description():
    default = [0.0] * 4096
    return {
        "PrevFireMask": tf.io.FixedLenFeature([4096], tf.float32),
        "FireMask": tf.io.FixedLenFeature([4096], tf.float32),
        "NDVI": tf.io.FixedLenFeature([4096], tf.float32, default_value=default),
        "DroughtIndex": tf.io.FixedLenFeature([4096], tf.float32, default_value=default),
        "Humidity": tf.io.FixedLenFeature([4096], tf.float32, default_value=default),
        "TemperatureMin": tf.io.FixedLenFeature([4096], tf.float32, default_value=default),
        "TemperatureMax": tf.io.FixedLenFeature([4096], tf.float32, default_value=default),
        "WindSpeed": tf.io.FixedLenFeature([4096], tf.float32, default_value=default),
        "WindDirection": tf.io.FixedLenFeature([4096], tf.float32, default_value=default),
        "Precipitation": tf.io.FixedLenFeature([4096], tf.float32, default_value=default),
        "Elevation": tf.io.FixedLenFeature([4096], tf.float32, default_value=default),
        "PopulationDensity": tf.io.FixedLenFeature([4096], tf.float32, default_value=default),
        "EnergyReleas"
        "eComponent": tf.io.FixedLenFeature([4096], tf.float32, default_value=default),
    }

def parse_tfrecord(example_proto):
    return tf.io.parse_single_example(example_proto, get_feature_description())