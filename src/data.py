'''
DOWNLOAD, PREPARE DIV2K DATASET, MAP INTO TENSOR AND APPLY AUGMENTATION
'''
import argparse
import os
import tensorflow as tf

from tensorflow.python.data.experimental import AUTOTUNE


class DIV2K:
    def __init__(self,
                 scale=2,
                 subset='train',
                 downgrade='bicubic',
                 images_dir='../data/div2k/images',
                 caches_dir='../data/div2k/caches'):

        self._ntire_2018 = True

        _scales = [2, 3, 4, 8]

        if scale in _scales:
            self.scale = scale
        else:
            raise ValueError('Scale must be in ${_scales}')

        if subset == 'train':
            self.image_ids = range(1, 801)
        elif subset == 'valid':
            self.image_ids = range(801, 901)
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        _downgrades_a = ['bicubic', 'unknown']
        _downgrades_b = ['mild', 'difficult']

        if scale == 8 and downgrade != 'bicubic':
            raise ValueError('Scale 8 only allowed for bicubic downgrade.')

        if downgrade in _downgrades_b and scale != 4:
            raise ValueError('{} downgrade requires scale 4'.format(downgrade))

        if downgrade == 'bicubic' and scale == 8:
            self.downgrade = 'x8'
        elif downgrade in _downgrades_b:
            self.downgrade = downgrade
        else:
            self.downgrade = downgrade
            self._ntire_2018 = False

        self.subset = subset
        self.images_dir = images_dir
        self.caches_dir = caches_dir

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(caches_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    def dataset(self, batch_size=16, repeat_count=None, random_transform=True):
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
        if random_transform:
            ds = ds.map(lambda lr, hr: random_crop(lr, hr, scale=self.scale), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def hr_dataset(self):
        if not os.path.exists(self._hr_images_dir()):
            print('Start downloading HR dataset...')
            download_archive(self._hr_images_archive(), self.images_dir, extract=True)

        ds = self._images_dataset(self._hr_image_files())

        return ds

    def lr_dataset(self):
        if not os.path.exists(self._lr_images_dir()):
            print('Start downloading LR dataset...')
            download_archive(self._lr_images_archive(), self.images_dir, extract=True)

        ds = self._images_dataset(self._lr_image_files())

        return ds

    # MANAGE CACHES
#     def _hr_cache_file(self):
#         return os.path.join(self.caches_dir, 'DIV2K_{}_HR.cache'.format(self.subset))

#     def _lr_cache_file(self):
#         return os.path.join(self.caches_dir, 'DIV2K_{}_LR_{}_X{}.cache'.format(self.subset, self.downgrade, self.scale))

#     def _hr_cache_index(self):
#         return '{}.index'.format(self._hr_cache_file())

#     def _lr_cache_index(self):
#         return '{}.index'.format(self._lr_cache_file())

    # LOAD IMAGES
    def _hr_image_files(self):
        images_dir = self._hr_images_dir()
        return [os.path.join(images_dir, '{}:04.png'.format(image_id)) for image_id in self.image_ids]

    def _lr_image_files(self):
        images_dir = self._lr_images_dir()
        return [os.path.join(images_dir, self._lr_image_file(image_id)) for image_id in self.image_ids]

    def _lr_image_file(self, image_id):
        if not self._ntire_2018 or self.scale == 8:
            return '{}:04x{}.png'.format(image_id, self.scale)
        else:
            return '{}:04x{}{}.png'.format(image_id, self.scale, self.downgrade[0])

    # LOCATE IMAGE DIRECTORIES
    def _hr_images_dir(self):
        return os.path.join(self.images_dir, 'DIV2K_{}_HR'.format(self.subset))

    def _lr_images_dir(self):
        if self._ntire_2018:
            return os.path.join(self.images_dir, 'DIV2K_{}_LR_{}'.format(self.subset, self.downgrade))
        else:
            return os.path.join(self.images_dir, 'DIV2K_{}_LR_{}'.format(self.subset, self.downgrade), 'X{}'.format(self.scale))

    # DEFINE DOWNLOAD PATH
    def _hr_images_archive(self):
        return 'DIV2K_{}_HR.zip'.format(self.subset)

    def _lr_images_archive(self):
        if self._ntire_2018:
            return 'DIV2K_{}_LR_{}.zip'.format(self.subset, self.downgrade)
        else:
            return 'DIV2K_{}_LR_{}_X{}.zip'.format(self.subset, self.downgrade, self.scale)

    # MAP IMAGES INTO TENSOR
    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds

#     # POPULATE CACHES
#     @staticmethod
#     def _populate_cache(ds, cache_file):
#         print('Caching decoded images in {} ...'.format(cache_file))
#         for _ in ds: pass
#         print('Cached decoded images in {} ...'.format(cache_file))


# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------


def random_crop(lr_img, hr_img, hr_crop_size=96, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


# -----------------------------------------------------------
#  IO
# -----------------------------------------------------------


def download_archive(file, target_dir, extract=True):
    source_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/{}'.format(file)
    target_dir = os.path.abspath(target_dir)
    tf.keras.utils.get_file(file, source_url, cache_subdir=target_dir, extract=extract)
    os.remove(os.path.join(target_dir, file))
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--prepare", action="store_true", default=False)
    parser.add_argument("--train_generator", action="store_true", default=False)
    parser.add_argument("--train_gans", action="store_true", default=False)

    args = parser.parse_args()

    if args.prepare: 
        print('----- Preparing train dataset -----')
        div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')
        train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
        print('----- Finished preparing train dataset -----')
        
        print('----- Preparing validation dataset -----')
        div2k_valid = DIV2K(scale=4, subset='valid', downgrade='bicubic')
        valid_ds = div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1)
        print('----- Finished preparing validation dataset -----')