import struct

def read_binary_file(path):
    ragged_array = []
    with open(path, "rb") as r:
        while True:
            size_bytes = r.read(4)
            if not size_bytes:
                break
            signal_length = struct.unpack('i', size_bytes)[0]
            signal_data = struct.unpack(f'{signal_length}h', r.read(signal_length * 2))
            ragged_array.append(list(signal_data))
    return ragged_array
