import sys, os, csv
import pandas as pd
import numpy as np

y_mask = 0x7FC00000
y_shift = 22

x_mask = 0x003FF000
x_shift = 12

polarity_mask = 0x800
polarity_shift = 11

valid_mask = 0x80000000
valid_shift = 31

def read_bits(arr, mask=None, shift=None):
    if mask is not None:
        arr = arr & mask
    if shift is not None:
        arr = arr >> shift
    return arr

def skip_header(fp):
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == "#":
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    return p

def load_raw_events(fp,bytes_skip=0, bytes_trim=0,filter_dvs=False, times_first=False):
    p = skip_header(fp)
    fp.seek(p + bytes_skip)
    data = fp.read()
    if bytes_trim > 0:
        data = data[:-bytes_trim]
    data = np.fromstring(data, dtype='>u4')
    if len(data) % 2 != 0:
        print(data[:20:2])
        print('---')
        print(data[1:21:2])
        raise ValueError('odd number of data elements')
    raw_addr = data[::2]
    timestamp = data[1::2]
    if times_first:
        timestamp, raw_addr = raw_addr, timestamp
    '''
    if filter_dvs:
        valid = read_bits(raw_addr, valid_mask, valid_shift)
        timestamp = timestamp[valid]
        raw_addr = raw_addr[valid]
    '''
    return timestamp, raw_addr

def parse_raw_address(addr,
                      x_mask=x_mask,x_shift=x_shift,
                      y_mask=y_mask,y_shift=y_shift,
                      polarity_mask=polarity_mask,
                      polarity_shift=polarity_shift):
    polarity = read_bits(addr, polarity_mask, polarity_shift).astype(np.bool)
    x = read_bits(addr, x_mask, x_shift)
    y = read_bits(addr, y_mask, y_shift)
    return x, y, polarity

def load_events(fp,filter_dvs=False, **kwargs):
    timestamp, addr = load_raw_events(fp,filter_dvs=filter_dvs)
    x, y, polarity = parse_raw_address(addr, **kwargs)
    return timestamp, x, y, polarity

'''
    :param file_name: path of the events file
    :type file_name: str
    :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
    :rtype: Dict

    This function defines how to read the origin binary data.
'''
def load_origin_data(file_name: str, np_file: str):
    aedat = file_name + ".aedat"
    csv = file_name + ".csv"
    with open(aedat, 'rb') as fp:
        t, x, y, p = load_events(fp,x_mask=0xfE,x_shift=1,y_mask=0x7f00,y_shift=8,
                    polarity_mask=1,polarity_shift=None)
        # leemos csv para hacer split por filas            
        csv_data = np.loadtxt(csv, dtype=np.uint32, delimiter=',', skiprows=1)
        # user26_fluorescent_labels.csv
        label_file_num = [0] * 19

        print("SHAPE - tiene que ser 19 y da: ", csv_data.shape[0])
        for i in range(csv_data.shape[0]):
            label = csv_data[i][0] - 1
            t_start = csv_data[i][1]
            t_end = csv_data[i][2]
            # aqui recoged
            mask = np.logical_and(t >= t_start, t < t_end)

            # aqui hay que guardar en la carpeta train o test y luego en la subcarpeta del label
            label_str = str(label)
            if label < 10: label_str = "0" + label_str
            print(label_str)
            #if os.path.isdir(os.path.join(np_file, "train/", label_str+"/")):
            if aedat in train_folders:
                print("AEDAT: ", aedat)
                path = os.path.join(np_file, "train/", label+"/")
                print(path)
            else:
                print("AEDAT: ", aedat)
                path = os.path.join(np_file, "test/", label+"/")
                print(path)
            print(path)

            final_file_name = path + file_name + f'{label}_{label_file_num[label]}.npz'
            np.savez(final_file_name,
                     t= t[mask],
                     x= x[mask],
                     y= y[mask],
                     p= p[mask])
            print(f'[{file_name}] saved.')
            label_file_num[label] += 1

        # return {'t': t, 'x': 127 - x, 'y': y, 'p': 1 - p.astype(int)}  # this will get the same data with http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/dat2mat.m
        # see https://github.com/jackd/events-tfds/pull/1 for more details about this problem
        return {'t': t, 'x': 127 - y, 'y': 127 - x, 'p': 1 - p.astype(int)}

def read_aedat_save_to_np(bin_file: str, np_file: str):
        events = load_origin_data(bin_file, np_file)

# Obtenemos los AEDATs
files = os.listdir(sys.argv[2])
#print(files)
print("Length of AEDAT + CSV list: ", len(files))
#print(os.getcwd())

# Creamos la carpeta events_np
print("Created events_np folder")
events_np_path = os.path.join(sys.argv[1], 'events_np/')
os.mkdir(events_np_path)

# Creamos la carpeta train y test
os.mkdir(os.path.join(events_np_path, "test/"))
os.mkdir(os.path.join(events_np_path, "train/"))

# Train y test
# fichero csv que contiene cuantas carpetas hacer
csv_data = pd.read_csv('gesture_mapping.csv', delimiter=',')
for i in range(len(csv_data)-1):
    print(i)
    os.mkdir(os.path.join(events_np_path, "train/", str(i)))
    os.mkdir(os.path.join(events_np_path, "test/", str(i)))

# con esto cogemos los usuarios que pertenecen al conjunto de train y test
with(open(sys.argv[3], "r")) as f:
    train_folders = f.readlines()

with(open(sys.argv[4], "r")) as f:
    test_folders = f.readlines()

# argv[3] tiene que ser la carpeta de los archivos aedat solo
final_names = []
for i in os.listdir(sys.argv[2]):
    final_names.append(i[:-6])

for i in range(len(final_names)):
    file = final_names[i]
    print("Working with file: ", file)
    if file != 'convertToNPZ.py':
        # user00
        # events np
        read_aedat_save_to_np(file, events_np_path)
