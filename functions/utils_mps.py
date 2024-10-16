# ============================ IMPORTS ====================================== #
 
import numpy as np

import struct
 
# ============================= CONFIG ====================================== #

# ------------------------------ SIZE IN BYTES ------------------------------ #

nBytesMpsHeader = 20
nBytesLengthImageStruct = 2
nBytesLengthSSt = 2
nBytesLengthTail = 8  # SSC + CRC

# COMMON sizes
nBytesCameraId        = 1           # cameraId                uint8_t     --> struct.unpack('B') 
nBytesTimeStamp_start = 8           # timeStamp_start         Uint64_t    --> struct.unpack('Q')
nBytesTimeStamp_end   = 8           # timeStamp_end           Uint64_t    --> struct.unpack('Q')
nBytesImageSize       = 4           # imageSize               uint32_t    --> struct.unpack('L')
nBytesObservationMode = 1           # observationMode         uint8_t     --> struct.unpack('B')
nBytesComponentID     = 1           # componentID             uint8_t     --> struct.unpack('B')
nBytesPipelineConfig  = 1           # pipelineConfig          uint8_t     --> struct.unpack('B')
nBytesnAcc            = 2           # NAcc                    uint16_t    --> struct.unpack('H')
nBytesImageIndex      = 4           # imageIndex              uint32_t    --> struct.unpack('L')
nBytesImageIndex_end  = 4           # imageIndex_end          uint32_t    --> struct.unpack('L')
nBytesRoi_x_offset    = 2           # roi_x_offset            uint16_t    --> struct.unpack('H')
nBytesRoi_x_size      = 2           # roi_x_size              uint16_t    --> struct.unpack('H')
nBytesRoi_y_offset    = 2           # roi_y_offset            uint16_t    --> struct.unpack('H')
nBytesRoi_y_size      = 2           # roi_y_size              uint16_t    --> struct.unpack('H')
nBytesImageType       = 1           # Image type              uint8_t     --> struct.unpack('B')

# TuMAG sizes
nBytesFW1               = 1           # position FW1            uint8_t     --> struct.unpack('B')
nBytesFW2               = 1           # position FW2            uint8_t     --> struct.unpack('B')
nBytesEtalonDN          = 2           # voltage etalon DN       uint16_t    --> struct.unpack('H')
nBytesEtalonSign        = 1           # sign etalon             uint8_t     --> struct.unpack('B')
nBytesRocli1_LCVR       = 2           # Rocli1_LCVR             uint16_t    --> struct.unpack('H')
nBytesRocli2_LCVR       = 2           # Rocli2_LCVR             uint16_t    --> struct.unpack('H')
nBytesEtalonVoltLecture = 2           # Etalon Volts reading    uint16_t    --> struct.unpack('H')

# --------------------------------------------------------------------------- #

defaultImgRowSize = 2048
defaultImgColSize = 2048

# =========================================================================== #


def GetDatafromHeader(receivedHeader):
    
    """
    Parameters
    ----------
    receivedHeader : bytes array
        Array of bytes containing the header info

    Returns
    -------
    Header : Dict
        Dictionary containing the header data.

    """
    
    dataLine = []
    
    Keys = [ 'CameraID', 'TimeStamp_start', 'TimeStamp_end', 'ImageSize', 
            'ObservationMode', 'PipelineConfig', 'nAcc', 'ImageIndex', 'ImageIndex_end',
            'Roi_x_offset', 'Roi_x_size', 'Roi_y_offset', 'Roi_y_size', 'ImageType',
            'FW1', 'FW2', 'EtalonDN', 'EtalonSign', 'Rocli1_LCVR', 'Rocli2_LCVR', 'EtalonVoltsReading']
    
    # Updating size of Headers
    positionStartSstLength = nBytesMpsHeader
    nBytesSSt = struct.unpack('H', receivedHeader[positionStartSstLength: positionStartSstLength + nBytesLengthSSt])[0]
    positionStartImageStructLength = positionStartSstLength + nBytesLengthSSt + nBytesSSt
    
    # COMMON positions
    positionStartCameraId        = positionStartImageStructLength + nBytesLengthImageStruct 
    positionStartTimeStamp_start = positionStartCameraId + nBytesCameraId
    positionStartTimeStamp_end   = positionStartTimeStamp_start + nBytesTimeStamp_start
    positionStartImageSize       = positionStartTimeStamp_end + nBytesTimeStamp_end
    positionStartObservationMode = positionStartImageSize + nBytesImageSize
    positionStartPipelineConfig  = positionStartObservationMode + nBytesObservationMode
    positionStartnAcc            = positionStartPipelineConfig + nBytesPipelineConfig
    positionStartImageIndex      = positionStartnAcc + nBytesnAcc
    positionStartImageIndex_end  = positionStartImageIndex + nBytesImageIndex
    positionStartRoi_x_offset    = positionStartImageIndex_end + nBytesImageIndex_end
    positionStartRoi_x_size      = positionStartRoi_x_offset + nBytesRoi_x_offset
    positionStartRoi_y_offset    = positionStartRoi_x_size + nBytesRoi_x_size
    positionStartRoi_y_size      = positionStartRoi_y_offset + nBytesRoi_y_offset
    positionStartImageType       = positionStartRoi_y_size + nBytesRoi_y_size
    
    # TuMAG positions
    positionFW1                  = positionStartImageType + nBytesImageType
    positionFW2                  = positionFW1 + nBytesFW1
    positionEtalonDN             = positionFW2 + nBytesFW2
    positionEtalonSign           = positionEtalonDN + nBytesEtalonDN
    positionRocli1_LCVR          = positionEtalonSign + nBytesEtalonSign
    positionRocli2_LCVR          = positionRocli1_LCVR + nBytesRocli1_LCVR
    positionEtalonVoltReading    = positionRocli2_LCVR + nBytesRocli2_LCVR

    # Extract COMMON Data
    dataLine.append(struct.unpack('B', receivedHeader[positionStartCameraId: positionStartCameraId + nBytesCameraId])[0])
    dataLine.append(struct.unpack('Q', receivedHeader[positionStartTimeStamp_start: positionStartTimeStamp_start + nBytesTimeStamp_start])[0])
    dataLine.append(struct.unpack('Q', receivedHeader[positionStartTimeStamp_end: positionStartTimeStamp_end + nBytesTimeStamp_end])[0])
    dataLine.append(struct.unpack('i', receivedHeader[positionStartImageSize: positionStartImageSize + nBytesImageSize])[0])
    dataLine.append(struct.unpack('B', receivedHeader[positionStartObservationMode: positionStartObservationMode + nBytesObservationMode])[0])
    dataLine.append(struct.unpack('B', receivedHeader[positionStartPipelineConfig: positionStartPipelineConfig + nBytesPipelineConfig])[0])
    dataLine.append(struct.unpack('H', receivedHeader[positionStartnAcc: positionStartnAcc + nBytesnAcc])[0])
    dataLine.append(struct.unpack('i', receivedHeader[positionStartImageIndex: positionStartImageIndex + nBytesImageIndex])[0])
    dataLine.append(struct.unpack('i', receivedHeader[positionStartImageIndex_end: positionStartImageIndex_end + nBytesImageIndex_end])[0])
    dataLine.append(struct.unpack('H', receivedHeader[positionStartRoi_x_offset: positionStartRoi_x_offset + nBytesRoi_x_offset])[0])
    dataLine.append(struct.unpack('H', receivedHeader[positionStartRoi_x_size: positionStartRoi_x_size + nBytesRoi_x_size])[0])
    dataLine.append(struct.unpack('H', receivedHeader[positionStartRoi_y_offset: positionStartRoi_y_offset + nBytesRoi_y_offset])[0])
    dataLine.append(struct.unpack('H', receivedHeader[positionStartRoi_y_size: positionStartRoi_y_size + nBytesRoi_y_size])[0])
    dataLine.append(struct.unpack('B', receivedHeader[positionStartImageType: positionStartImageType + nBytesImageType])[0])
     
	# Extract TuMAG Data
    dataLine.append(struct.unpack('B', receivedHeader[positionFW1: positionFW1 + nBytesFW1])[0])
    dataLine.append(struct.unpack('B', receivedHeader[positionFW2: positionFW2 + nBytesFW2])[0])
    dataLine.append(struct.unpack('H', receivedHeader[positionEtalonDN: positionEtalonDN + nBytesEtalonDN])[0])
    dataLine.append(struct.unpack('B', receivedHeader[positionEtalonSign: positionEtalonSign + nBytesEtalonSign])[0])
    dataLine.append(struct.unpack('H', receivedHeader[positionRocli1_LCVR: positionRocli1_LCVR + nBytesRocli1_LCVR])[0])
    dataLine.append(struct.unpack('H', receivedHeader[positionRocli2_LCVR: positionRocli2_LCVR + nBytesRocli2_LCVR])[0])
    dataLine.append(struct.unpack('H', receivedHeader[positionEtalonVoltReading: positionEtalonVoltReading + nBytesEtalonVoltLecture])[0])
   
    # Saving data in dictionary
    Header = {}
    for ind, key in enumerate(Keys): 
        Header[key] = dataLine[ind]
    
    return Header

def HeadernImageSeparator(Image_path):
    
    """
    Function that given image path separates the header from the image in byte 
    arrays

    Parameters
    ----------
    Image_path : str
        Path to the image.

    Returns
    -------
    receivedHeader : bytes array
        bytes array containing header data.
    headerLength : int
        Length of the header in bytes.
    receivedImage : bytes array
        Bytes array containing image data.

    """
    
    # Process the MPS Science Header (nBytesMpsHeader bytes --> 20 bytes)
    nBytesSync = 2              # Identify start of record --> 0x44 0x54
    nBytesSystemId = 1          # System ID of generated data
    nBytesDataId = 1            # Description of data type
    nBytesTotalLength = 4       # Total data packet length (including header)
    nBytesTime = 8              # Data acquisition start time
    nBytesSensorId = 1          # 
    nBytesHeaderVersion = 1     # 
    nBytesHeaderLength = 2      # Total length of header (in bytes). Take it as an offset to Data
    
    positionStartTotalLength = nBytesSync + nBytesSystemId + nBytesDataId
    positionStartHeaderLength = nBytesSync + nBytesSystemId + nBytesDataId + nBytesTotalLength + nBytesTime + nBytesSensorId + nBytesHeaderVersion
    
    file = open(Image_path,"rb")
    fullReceivedImage = file.read()
    mpsHeader = fullReceivedImage[0 : nBytesMpsHeader]
    
    totalLength = struct.unpack('i', mpsHeader[positionStartTotalLength: positionStartTotalLength + nBytesTotalLength])[0]
    headerLength = struct.unpack('H', mpsHeader[positionStartHeaderLength: positionStartHeaderLength + nBytesHeaderLength])[0]

    receivedHeader = fullReceivedImage[0:headerLength]
    receivedImage = fullReceivedImage[headerLength:totalLength-nBytesLengthTail]

    file.close()

    return receivedHeader, headerLength, receivedImage

def read_image(file, width, height, headerLength):
    
    """
    This function reads a RAW image as a Numpy array
    """
    
    image = np.zeros([height, width],dtype=np.int16)
    im_dummy = np.fromfile(file,dtype='<i2', count=width*height, offset=headerLength)

    if ( im_dummy.size > 0 ):
        image = im_dummy.reshape([height, width])

    return image





