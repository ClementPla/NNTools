from enum import Enum, auto

supportedExtensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".jp2", ".exr", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm"}
supportedExtensions.update({ext.upper() for ext in supportedExtensions})


class NNOpt(Enum):
    FILL_DOWNSAMPLE = "downsample"
    FILL_UPSAMPLE = "upsample"
    MISSING_DATA_FLAG = "missing"
    AUTO_INTERPRET_RGB_MASK = auto()
    CACHE_DISK = "disk"
    CACHE_MEMORY = "memory"
    CACHE_DUPLICATED_MEMORY = "duplicated_memory"
