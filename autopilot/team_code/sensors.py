"""
defines
"""
# sensor ids
CENTER_CAMERA = 'rgb'
LEFT_CAMERA = 'rgb_left'
RIGHT_CAMERA = 'rgb_right'
IMU = 'imu'
GPS = 'gps'
SPEEDOMETER = 'speed'

# sensor types
CAMERA_SENSOR_TYPE = 'sensor.camera.rgb'
IMU_SENSOR_TYPE = 'sensor.other.imu'
GPS_SENSOR_TYPE = 'sensor.other.gnss'
SPEEDOMETER_SENSOR_TYPE ='sensor.speedometer'

# id:type dict
SENSORS_DICT = {
  CENTER_CAMERA: CAMERA_SENSOR_TYPE,
  LEFT_CAMERA: CAMERA_SENSOR_TYPE,
  RIGHT_CAMERA: CAMERA_SENSOR_TYPE,
  IMU: IMU_SENSOR_TYPE, 
  GPS: GPS_SENSOR_TYPE,
  SPEEDOMETER: SPEEDOMETER_SENSOR_TYPE
}

# defaults
DEFAULT_SENSOR_FREQUENCY = 20                            # why?

# camera
DEFAULT_CAMERA_DIM_WIDTH = 256                           #
DEFAULT_CAMERA_DIM_HEIGHT = 144                          #

# camera fov
DEFAULT_CAMERA_FOV_DEG = 90

# camera rotation
DEFAULT_SIDE_CAMERA_YAW = 45.0
DEFAULT_RIGHT_CAMERA_YAW = DEFAULT_SIDE_CAMERA_YAW       # 
DEFAULT_LEFT_CAMERA_YAW = DEFAULT_SIDE_CAMERA_YAW * -1   # mirror right

# camera positions
DEFAULT_SIDE_CAMERA_X = 1.2                              #
DEFAULT_SIDE_CAMERA_Z = 1.3                              #
DEFAULT_CAMERA_Y = 0.0                                   #
DEFAULT_SIDE_CAMERA_Y = 0.25                             #
DEFAULT_RIGHT_SIDE_CAMERA_Y =  DEFAULT_SIDE_CAMERA_Y     # top
DEFAULT_LEFT_SIDE_CAMERA_Y =  DEFAULT_SIDE_CAMERA_Y * -1 # bottom 

# speedometer
DEFAULT_SPEEDOMETER_FREQUENCY = DEFAULT_SENSOR_FREQUENCY

# imu 
DEFAULT_IMU_TICK = 1/DEFAULT_SENSOR_FREQUENCY

# gps 
DEFAULT_GPS_TICK = 1/(DEFAULT_SENSOR_FREQUENCY * 5)

NO_OFFSET_VALUE = 0.0

"""
data
"""

SENSORS = [
{
  'type': SENSORS_DICT[CENTER_CAMERA],
  'x': DEFAULT_SIDE_CAMERA_X, 
  'y': DEFAULT_CAMERA_Y, 
  'z': DEFAULT_SIDE_CAMERA_Z,
  'roll': NO_OFFSET_VALUE, 
  'pitch': NO_OFFSET_VALUE, 
  'yaw': NO_OFFSET_VALUE,
  'width': DEFAULT_CAMERA_DIM_WIDTH, 
  'height': DEFAULT_CAMERA_DIM_HEIGHT, 
  'fov': DEFAULT_CAMERA_FOV_DEG,
  'id': CENTER_CAMERA
},
{
  'type': SENSORS_DICT[LEFT_CAMERA],
  'x': DEFAULT_SIDE_CAMERA_X, 
  'y': DEFAULT_LEFT_SIDE_CAMERA_Y, # bottom side camera
  'z': DEFAULT_SIDE_CAMERA_Z,
  'roll': NO_OFFSET_VALUE, 
  'pitch': NO_OFFSET_VALUE, 
  'yaw': DEFAULT_LEFT_CAMERA_YAW, # rotate left
  'width': DEFAULT_CAMERA_DIM_WIDTH, 
  'height': DEFAULT_CAMERA_DIM_HEIGHT, 
  'fov': DEFAULT_CAMERA_FOV_DEG,
  'id': LEFT_CAMERA
},
{
  'type': SENSORS_DICT[RIGHT_CAMERA],
  'x': DEFAULT_SIDE_CAMERA_X, 
  'y': DEFAULT_RIGHT_SIDE_CAMERA_Y, # top side camera
  'z': DEFAULT_SIDE_CAMERA_Z,
  'roll': NO_OFFSET_VALUE, 
  'pitch': NO_OFFSET_VALUE, 
  'yaw': DEFAULT_RIGHT_CAMERA_YAW, # rotate right
  'width': DEFAULT_CAMERA_DIM_WIDTH, 
  'height': DEFAULT_CAMERA_DIM_HEIGHT, 
  'fov': DEFAULT_CAMERA_FOV_DEG,
  'id': RIGHT_CAMERA
},
{ 
  'type': SENSORS_DICT[IMU],
  'x': NO_OFFSET_VALUE, 
  'y': NO_OFFSET_VALUE, 
  'z': NO_OFFSET_VALUE,
  'roll': NO_OFFSET_VALUE, 
  'pitch': NO_OFFSET_VALUE, 
  'yaw': NO_OFFSET_VALUE,
  'sensor_tick': DEFAULT_IMU_TICK,
  'id': IMU
},
{
  'type': SENSORS_DICT[GPS],
  'x': NO_OFFSET_VALUE, 
  'y': NO_OFFSET_VALUE, 
  'z': NO_OFFSET_VALUE,
  'roll': NO_OFFSET_VALUE,
  'pitch': NO_OFFSET_VALUE,
  'yaw': NO_OFFSET_VALUE,
  'sensor_tick': DEFAULT_GPS_TICK,
  'id': GPS
},
{
  'type': SENSORS_DICT[SPEEDOMETER],
  'reading_frequency': DEFAULT_SPEEDOMETER_FREQUENCY,
  'id': SPEEDOMETER
}
]
