
import os
import errno
import codecs
import shutil
import numpy as np

from collections import defaultdict

def group_by_class(keyvaluepairs):
    grouping = defaultdict(list)
    for k, v in keyvaluepairs:
        grouping[k].append(v)
    return grouping

def get_drivers():
    with codecs.open('driver_imgs_list.csv') as fin:
        drivers = defaultdict(list)
        
        linenumber = -1
        for line in fin:
            linenumber += 1
            if linenumber == 0:
                continue

            columns = line.split(',')
            driver_id = columns[0].strip()
            classname = columns[1].strip()
            imagename = columns[2].strip()

            drivers[driver_id].append((classname, imagename))
    result = {}
    for d in drivers.keys():
        result[d] = group_by_class(drivers[d])
    return result

def copy_file(sourcedirectory, sourcefile, targetdirectory):
    try:
        os.makedirs(targetdirectory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    sourcefilepath = os.path.join(sourcedirectory, sourcefile)
    targetfilepath = os.path.join(targetdirectory, sourcefile)
    shutil.copyfile(sourcefilepath, targetfilepath)

def copy_partition(drivers, drivernames, targetdir):
    sourcedir = '../data/statefarm/train'
    for dkey in drivernames:
        print('Copying partition for driver {0}'.format(dkey))
        d = drivers[dkey]
        for c, imgs in d.iteritems():
            for i in imgs:
                copy_file(
                    os.path.join(sourcedir, c),
                    i,
                    os.path.join(targetdir, c))    
            
def main():
    drivers = get_drivers()
    print('Number of drivers: {0}'.format(len(drivers.keys())))

    for d, v in drivers.iteritems():
        print('Driver: {0}'.format(d))
        if len(v) != 10:
            print('Alert: missing class!!!!')
        for c, imgs in v.iteritems():
            print('{0}: {1}'.format(c, len(imgs)))

    driver_permutation = np.random.permutation(drivers.keys()).tolist()
    valid_drivers = driver_permutation[:4]
    train_drivers = driver_permutation[4:]

    copy_partition(drivers, valid_drivers,
                   '../data/statefarm/partition/valid')
    copy_partition(drivers, train_drivers,
                   '../data/statefarm/partition/train')

if __name__ == '__main__':
    main()

