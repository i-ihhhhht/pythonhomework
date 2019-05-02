from __future__ import print_function
import numpy
import PIL.Image
import pickle
import matplotlib.pyplot
import pdb

class Operation(object):
    image_base_path = "../image/"
    data_base_path = "../data/"

    def image_to_array(self, filenames):

        n = filenames.__len__()
        result = numpy.array([])
        print("1")
        for i in range(n):
            image = PIL.Image.open(self.image_base_path + filenames[i])
            r, g, b = image.split()

            r_arr = numpy.array(r).reshape(1024)
            g_arr = numpy.array(g).reshape(1024)
            b_arr = numpy.array(b).reshape(1024)

            image_arr = numpy.concatenate((r_arr, g_arr, b_arr))
            result = numpy.concatenate((result, image_arr))

        result = result.reshape(n, 3072)
        print("2")
        file_path = self.data_base_path + 'data2.bin'
        with open(file_path, mode='wb') as f:
            pickle.dump(result, f)
        print("3")

    def array_to_image(self, filename):

        with open(self.data_base_path + filename, mode='rb') as f:
            arr = pickle.load(f)
        rows = arr.shape[0]  # rows=5
        # pdb.set_trace()
        # print("rows:",rows)
        arr = arr.reshape(rows, 3, 32, 32)
        print(arr)
        for index in range(rows):
            a = arr[index]

            r = PIL.Image.fromarray(a[0]).convert('L')
            g = PIL.Image.fromarray(a[1]).convert('L')
            b = PIL.Image.fromarray(a[2]).convert('L')
            image = PIL.Image.merge("RGB", (r, g, b))

            matplotlib.pyplot.imshow(image)
            matplotlib.pyplot.show()
        # image.save(self.image_base_path + "result" + str(index) + ".png",'png')


if __name__ == "__main__":
    my_operator = Operation()
    images = []
    for j in range(5):
        images.append(str(j) + ".png")
    my_operator.array_to_image('data2.bin')

