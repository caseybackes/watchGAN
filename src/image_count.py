import os

processed_image_dir = '../data/train/processed_images'
raw_image_dir = '../data/images'

p_count = len(os.listdir(processed_image_dir))
r_count = len(os.listdir(raw_image_dir))

print('Raw Image count: ', r_count)
print('Processed Image count: ', p_count)