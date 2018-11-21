import numpy as np
import cv2


def blur_image(data, w):
    num_patches, height, width, channel = data.shape
    batch_q = np.zeros((num_patches, height, width, channel))
    for i in range(num_patches):
        for j in range(channel):
            img = data[i, :, :, j]
            img = cv2.blur(img, w)
            batch_q[i, :, :, j] = img
    return batch_q


class Image(object):
    def __init__(self, ms, ms2, ms4, pan2, pan4):
        self.ms = ms
        self.pan2 = pan2
        self.pan4 = pan4
        self.ms2 = ms2
        self.ms4 = ms4
        self.ms_struct = blur_image(ms, (5,5))
        self.ms_detail = self.ms - self.ms_struct
        self.ms2_struct = blur_image(ms2, (11,11))
        self.ms2_detail = self.ms2 - self.ms2_struct
        self.ms4_struct = blur_image(ms4, (21,21))
        self.ms4_detail = self.ms4 - self.ms4_struct
        self.pan2_detail = self.pan2 - blur_image(pan2, (11,11))
        self.pan4_detail = self.pan4 - blur_image(pan4, (21,21))

    def Batch(self, rand_index):
        return {
            'ms': self.ms[rand_index,:,:,:],
            'pan2': self.pan2[rand_index, :, :, :],
            'pan4': self.pan4[rand_index, :, :, :],
            'ms2': self.ms2[rand_index, :, :, :],
            'ms4': self.ms4[rand_index, :, :, :],
            'ms_struct': self.ms_struct[rand_index, :, :, :],
            'ms_detail': self.ms_detail[rand_index, :, :, :],
            'ms2_struct': self.ms2_struct[rand_index, :, :, :],
            'ms2_detail': self.ms2_detail[rand_index, :, :, :],
            'ms4_struct': self.ms4_struct[rand_index, :, :, :],
            'ms4_detail': self.ms4_detail[rand_index, :, :, :],
            'pan2_detail': self.pan2_detail[rand_index, :, :, :],
            'pan4_detail': self.pan4_detail[rand_index, :, :, :]

        }
