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
        return {'t': t, 'x': 127 - y, 'y': 127 - x, 'p': 1 - p.astype(int)}
   

def read_aedat_save_to_np(bin_file: str, np_file: str):
        aedat = bin_file + ".aedat"
        events = load_origin_data(bin_file, np_file)
        label_file_num = [0] * 19
        final_file_name = bin_file + '.npz'
        np.savez(final_file_name, t=events['t'], x=events['x'],y=events['y'],p=events['p'])
        print(f'Save [{bin_file}] to [{np_file}].')


file = sys.argv[1]  
read_aedat_save_to_np(file[:-6], ".")
