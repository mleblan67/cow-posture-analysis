import numpy as np
import struct

'''
Converts a float to a half-precision, signed binary (16-bits)
'''
def float_to_bin(float_val):
    # Convert to half-precision (16-bit) float
    half_val = np.float16(float_val)
    # Get binary representation
    bin_ = bin(half_val.view('H'))[2:].zfill(16)

    return bin_

'''
Converts a 16-bit binary to the given bit length
'''
def truncate_bits(bin_val, shift_amount):
    # Right shift the bits the given amount
    shifted_binary = bin_val[:-shift_amount]
    # Pad the right side to get back to 16 bits
    shifted_binary = shifted_binary.ljust(16,'0')

    return shifted_binary

'''
Converts a given binary string to its hexadecimal equivalent
'''
def bin_to_hex(bin_val):
    # Convert base to to int
    int_equivalent = int(bin_val, 2)
    # convert int to hex
    hex_equivalent = hex(int_equivalent)

    return hex_equivalent


'''
Converts a given float data into a hex equivalent with a specified truncation of bits
'''
def float_to_hex(float_val, shift_amount = 0):
    # Convert to binary
    bin_val = float_to_bin(float_val)
    # Truncate bits
    bin_val = truncate_bits(bin_val, shift_amount)
    # Convert binary to hex
    hex_val = bin_to_hex(bin_val)

    return hex_val

'''
Converts a given hex value into its binary representation
'''
def hex_to_binary(hex_val):
    # Convert to an int
    hex_int = int(hex_val, 16)
    # Convert that into binary
    binary_str = bin(hex_int)
    # Get rid of 0b from string
    binary_str = binary_str[2:]

    return binary_str.zfill(16)

'''
From ChatGPT
Converts a given 16-bit binary string to a 16-bit float
'''
def bin_to_float16(binary):
    # Make sure our binary is 16 bits
    # If it isn't, pad with zeros on the left
    binary = binary.ljust(16, '0')

    # Convert the binary string to an integer
    int_value = int(binary, 2)

    # Pack the integer into a 2-byte bytes object (little-endian)
    packed = int_value.to_bytes(2, byteorder='little')

    # Unpack the bytes into a half-precision float
    result = struct.unpack('e', packed)[0]
    return np.float16(result)

def hex_to_float(hex_val):
    # Convert back to binary
    binary = hex_to_binary(hex_val)
    # Convert binary back to a 16-bit float
    float_val = bin_to_float16(binary)

    return float_val