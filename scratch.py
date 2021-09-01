from src.data.SplittingManager import SplittingManager

if __name__ == '__main__':
    splitting_manager = SplittingManager(validation_size=0.18, test_size=0.1,
                                         k_cross_valid=1, seed=54288, num_workers=24,
                                         google_images=True)


    print('hi')
