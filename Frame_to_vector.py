import os
import numpy
import cv2

def MainPath():
    main_path = os.path.join(os.path.expanduser('~'), 'my_project')
    return main_path

def getFramesFolderPath(mode = 'grayscale_resize_1to5'):
    if mode == 'raw' or mode == 'grayscale' or mode == 'grayscale_resize_1to10' or mode == 'grayscale_resize_1to5' or mode == 'grayscale_resize_1to2':
        folder_path = os.path.join(MainPath(),'Videos', 'frames', '{}_frames'.format(mode))
    else:
        raise ValueError('Invalid mode = {}'.format(mode))
    return folder_path

def OneHot(Y):
    uniqueY = numpy.unique(Y)
    oneHotY = numpy.zeros([Y.shape[0], uniqueY.shape[0]])
    for num, i in enumerate(Y):
        oneHotY[num][i] = 1
    return oneHotY

def main():
    modelist = [
                'grayscale_resize_1to10',
                # 'grayscale_resize_1to5',
                # 'grayscale_resize_1to2',
                # 'grayscale',
                # 'raw'
                ]
    for mode in modelist:
        print('Vectorization of files from mode = {}'.format(mode))
        folder_path = getFramesFolderPath(mode = mode)
        frame_paths = os.listdir(folder_path)
        frame_paths = sorted(frame_paths)
        first_class_Y = numpy.array([int(p.split('firstclass_', 1)[1].split('_')[0]) for p in frame_paths])
        second_class_Y = numpy.array([int(p.split('secondclass_', 1)[1].split('.jpg')[0]) for p in frame_paths])
        first_class_Y = OneHot(first_class_Y)
        second_class_Y = OneHot(second_class_Y)
        frame_fullpaths = [os.path.join(folder_path, fp) for fp in frame_paths]
        maxlen = len(frame_fullpaths)
        print('Creating empty numpy')
        first_class_data = numpy.empty((len(frame_fullpaths),2), dtype=object)
        second_class_data = numpy.empty((len(frame_fullpaths),2), dtype=object)
        print('Appending to numpy')
        for num, ffp in enumerate(frame_fullpaths):
            print('{} : {} of {} appended to list'.format(mode,num, maxlen))
            second_class_data[num][0] = cv2.imread(ffp, cv2.IMREAD_UNCHANGED)
            first_class_data[num][0] = cv2.imread(ffp, cv2.IMREAD_UNCHANGED)
            second_class_data[num][1] = second_class_Y[num]
            first_class_data[num][1] = first_class_Y[num]
        first_class_npy_path = os.path.join(MainPath(),'Videos', 'numpy', '{}_{}_vector.npy'.format(mode, 'firstclass'))
        second_class_npy_path = os.path.join(MainPath(),'Videos', 'numpy', '{}_{}_vector.npy'.format(mode, 'secondclass'))
        print('Saving File...')
        numpy.save(first_class_npy_path,first_class_data)
        numpy.save(second_class_npy_path,second_class_data)
        print('Done')

if __name__ == '__main__':
    main()

