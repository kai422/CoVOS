import subprocess
import os


class path_config():
    @staticmethod
    def hevc_feature_decoder_path():
        return "decoder/bin/hevc"

    @staticmethod
    def ffmpeg_path():
        with open(os.devnull, 'w') as devnull:
            ffmpeg = subprocess.check_output(['which', 'ffmpeg'],
                                             stderr=devnull).decode().rstrip('\r\n')
            ffmpeg_home = os.path.dirname(ffmpeg)
        return ffmpeg_home

    @staticmethod
    def data_path():
        return '/data/kai422/'
