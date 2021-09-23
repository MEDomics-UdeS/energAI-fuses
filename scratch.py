from src.data.SplittingManager import SplittingManager
from numpy.random import randint

if __name__ == '__main__':
    # splitting_manager = SplittingManager(validation_size=0,
    #                                      test_size=0,
    #                                      k_cross_valid=1,
    #                                      seed=54288,
    #                                      google_images=True,
    #                                      image_size=2048,
    #                                      num_workers=24)
    #
    #
    # print('hi')

    my_list = []
    for i in range(10):
        my_list.append(str(randint(0,100000)))

    print(my_list)