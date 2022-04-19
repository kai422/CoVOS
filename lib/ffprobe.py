import subprocess as sp
import skvideo

from skvideo.utils import *
from skvideo import _HAS_FFMPEG
from skvideo import _FFMPEG_PATH
from skvideo import _FFPROBE_APPLICATION
from path_config import path_config

skvideo.setFFmpegPath(path_config.ffmpeg_path())

def ffprobe(filename):
    """get metadata by using ffprobe

    Checks the output of ffprobe on the desired video
    file. MetaData is then parsed into a dictionary.

    Parameters
    ----------
    filename : string
        Path to the video file

    Returns
    -------
    metaDict : dict
       Dictionary containing all header-based information 
       about the passed-in source video.

    """
    # check if FFMPEG exists in the path
    assert _HAS_FFMPEG, "Cannot find installation of real FFmpeg (which comes with ffprobe)."

    try:
        command = [_FFMPEG_PATH + "/" + _FFPROBE_APPLICATION, "-v", "error", "-show_streams", "-show_packets", "-print_format", "xml", filename]
        # simply get std output
        xml = check_output(command)

        d = xmltodictparser(xml)["ffprobe"]

        return d["streams"], d["packets"]
    except:
        return {}
