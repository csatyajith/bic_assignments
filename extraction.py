import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_digits

# splitting the data into 3 sets (train , validation and test)
"""
    returns a tuple containing training , test and validation data set
    
    training data set is a tuple with two items of type ndarray
    x_train - This contains 80% of the initial (1797) dataset images .
              each image is of size 8*8 pixels
    y_train - digit values of corresponding images
    
    Similarly for testing  and validation data set which account
    for 10% and 15% of initial data set 

"""
def data_extraction():
 # data = load_digits(*, n_class=10, return_X_y=False, as_frame=False)
 digits = load_digits()
 # np.array(digits).reshape(-1,1)
 images = digits['images']
 target = digits['target']
 train_ratio = 0.75
 validation_ratio = 0.15
 test_ratio = 0.10
 # train is now 75% of the entire data set
 # the _junk suffix means that we drop that variable completely
 x_train, x_test, y_train, y_test = train_test_split(images, target, test_size=1 - train_ratio)

 # test is now 10% of the initial data set
 # validation is now 15% of the initial data set
 x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio))

 training_data = (x_train, y_train)
 testing_data = (x_test, y_test)
 validation_data = (x_val, y_val)
 data=(training_data,testing_data,validation_data)
 print(digits.data.shape)
 # plt.gray()
 # plt.matshow(digits.images[0])
 # plt.show()

 return data


if __name__ == '__main__':
    train_data,test_data,validation_data = data_extraction()
    print(train_data[0].shape)


