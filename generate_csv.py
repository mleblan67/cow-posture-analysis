from utils.load_data_utils import load_to_df
from utils.bit_conversion_utils import float_to_hex, hex_to_float

# Load in csvs
filenames = ['sensor_data_T03S0728.csv','sensor_data_T03S0729.csv']
df,_ = load_to_df(filenames,'T03_groundtruths.csv',prefix='converted_data/T03/')

# Only keep timestamps and accel data
features_to_keep = ['timestamp','accel_x_mps2','accel_y_mps2','accel_z_mps2']
df = df.loc[:, df.columns.intersection(features_to_keep)]

# Add 14-bit hex columns for x,y,z data
print("Adding 14-bit hex data...")
df['accel_x_14_hex'] = df.apply((lambda row: float_to_hex(row.accel_x_mps2, 2)), axis=1)
df['accel_y_14_hex'] = df.apply((lambda row: float_to_hex(row.accel_y_mps2, 2)), axis=1)
df['accel_z_14_hex'] = df.apply((lambda row: float_to_hex(row.accel_z_mps2, 2)), axis=1)

# Add 14-bit float16 columns for x,y,z data
print("Adding 14-bit float data...")
df['accel_x_14_float'] = df.apply((lambda row: hex_to_float(row.accel_x_14_hex)), axis=1)
df['accel_y_14_float'] = df.apply((lambda row: hex_to_float(row.accel_y_14_hex)), axis=1)
df['accel_z_14_float'] = df.apply((lambda row: hex_to_float(row.accel_z_14_hex)), axis=1)

# Add 12-bit hex columns for x,y,z data
print("Adding 12-bit data...")
df['accel_x_12_hex'] = df.apply((lambda row: float_to_hex(row.accel_x_mps2, 4)), axis=1)
df['accel_y_12_hex'] = df.apply((lambda row: float_to_hex(row.accel_y_mps2, 4)), axis=1)
df['accel_z_12_hex'] = df.apply((lambda row: float_to_hex(row.accel_z_mps2, 4)), axis=1)

# Add 12-bit float16 columns for x,y,z data
print("Adding 12-bit float data...")
df['accel_x_12_float'] = df.apply((lambda row: hex_to_float(row.accel_x_12_hex)), axis=1)
df['accel_y_12_float'] = df.apply((lambda row: hex_to_float(row.accel_y_12_hex)), axis=1)
df['accel_z_12_float'] = df.apply((lambda row: hex_to_float(row.accel_z_12_hex)), axis=1)

# Add 10-bit hex columns for x,y,z data
print("Adding 10-bit data...")
df['accel_x_10_hex'] = df.apply((lambda row: float_to_hex(row.accel_x_mps2, 6)), axis=1)
df['accel_y_10_hex'] = df.apply((lambda row: float_to_hex(row.accel_y_mps2, 6)), axis=1)
df['accel_z_10_hex'] = df.apply((lambda row: float_to_hex(row.accel_z_mps2, 6)), axis=1)

# Add 10-bit float16 columns for x,y,z data
print("Adding 10-bit float data...")
df['accel_x_10_float'] = df.apply((lambda row: hex_to_float(row.accel_x_10_hex)), axis=1)
df['accel_y_10_float'] = df.apply((lambda row: hex_to_float(row.accel_y_10_hex)), axis=1)
df['accel_z_10_float'] = df.apply((lambda row: hex_to_float(row.accel_z_10_hex)), axis=1)

# Add 8-bit hex columns for x,y,z data
print("Adding 8-bit data...")
df['accel_x_8_hex'] = df.apply((lambda row: float_to_hex(row.accel_x_mps2, 8)), axis=1)
df['accel_y_8_hex'] = df.apply((lambda row: float_to_hex(row.accel_y_mps2, 8)), axis=1)
df['accel_z_8_hex'] = df.apply((lambda row: float_to_hex(row.accel_z_mps2, 8)), axis=1)

# Add 8-bit float16 columns for x,y,z data
print("Adding 8-bit float data...")
df['accel_x_8_float'] = df.apply((lambda row: hex_to_float(row.accel_x_8_hex)), axis=1)
df['accel_y_8_float'] = df.apply((lambda row: hex_to_float(row.accel_y_8_hex)), axis=1)
df['accel_z_8_float'] = df.apply((lambda row: hex_to_float(row.accel_z_8_hex)), axis=1)

# Add 6-bit hex columns for x,y,z data
print("Adding 6-bit data...")
df['accel_x_6_hex'] = df.apply((lambda row: float_to_hex(row.accel_x_mps2, 10)), axis=1)
df['accel_y_6_hex'] = df.apply((lambda row: float_to_hex(row.accel_y_mps2, 10)), axis=1)
df['accel_z_6_hex'] = df.apply((lambda row: float_to_hex(row.accel_z_mps2, 10)), axis=1)

# Add 6-bit float16 columns for x,y,z data
print("Adding 6-bit float data...")
df['accel_x_6_float'] = df.apply((lambda row: hex_to_float(row.accel_x_6_hex)), axis=1)
df['accel_y_6_float'] = df.apply((lambda row: hex_to_float(row.accel_y_6_hex)), axis=1)
df['accel_z_6_float'] = df.apply((lambda row: hex_to_float(row.accel_z_6_hex)), axis=1)

df.to_csv('converted_bits_data_T03.csv', index=False)