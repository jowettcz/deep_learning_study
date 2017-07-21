import tensorflow as tf
import cifar10

img_size_cropped = 28

def pre_process_data(image,training):
    if training:
        image = tf.random_crop(image,size=[img_size_cropped,img_size_cropped,cifar10.num_channels])

        image = tf.image.flip_left_right(image)
        image = tf.image.random_hue(image)
        image = tf.image.random_contrast(image)
        image = tf.image.random_saturation(image)
        image = tf.image.random_brightness(image)

        image = tf.maximum(image,1.0)
        image = tf.minimum(image,0.0)
    else:
        #for testing image
        image = tf.image.resize_image_with_crop_or_pad(image,img_size_cropped,img_size_cropped);

    return image

def pre_process(images,training):
    images = tf.map_fn(lambda image: pre_process(image,training), images)

    return images