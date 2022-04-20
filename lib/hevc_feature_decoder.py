# Based on Almar Klein's imageio code and skivideo.io.ffmepg code.
# Copyright (c) 2015, imageio contributors
# distributed under the terms of the BSD License.
‘’‘
Copyright (c) 2017, Scikit-Video Developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of [project] nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
’‘’

import os
import cv2
import time
import subprocess as sp
import math
import numpy as np
from path_config import path_config
from lib.ffprobe import ffprobe

_HEVC_FEAT_DECODER = path_config.hevc_feature_decoder_path()

_FFMPEG_SUPPORTED_DECODERS = [b".mp4"]


class HevcFeatureReader:
    """Reads frame features using HevcFeatureReader

    Return quadtree structure, yuv data, residual, raw motion vectors.

    """

    def __init__(self, filename, nb_frames, n_parallel):
        # General information
        _, self.extension = os.path.splitext(filename)
        if not os.path.exists(filename):
            print(filename, " not exist.")
        viddict, packets = ffprobe(filename)
        viddict = viddict["stream"]
        packets = packets["packet"]
        packets_pts = [int(packet["@pts"]) for packet in packets]
        self.viddict = viddict
        self.bitstream_pts_order = np.argsort(packets_pts)
        self.decode_order = np.argsort(self.bitstream_pts_order)
        self.bpp = -1  # bits per pixel
        self.pix_fmt = viddict["@pix_fmt"]
        if nb_frames is not None:
            self.nb_frames = nb_frames
        else:
            self.nb_frames = int(viddict["@nb_frames"])

        self.width = int(viddict["@width"])
        self.height = int(viddict["@height"])

        self.coded_width = int(viddict["@coded_width"])
        self.coded_height = int(viddict["@coded_height"])

        self.ctu_width = math.ceil(self.width / 64.0)
        self.ctu_height = math.ceil(self.height / 64.0)
        self.nb_ctus = self.ctu_width * self.ctu_height

        if self.pix_fmt not in ["yuv420p"]:
            print(self.pix_fmt)
            print(filename)
            raise NameError("Expect a yuv420p input.")

        assert str.encode(self.extension).lower() in _FFMPEG_SUPPORTED_DECODERS, (
            "Unknown decoder extension: " + self.extension.lower()
        )

        self._filename = filename

        self.DEVNULL = open(os.devnull, "wb")

        # Create process
        self._parallel = str(n_parallel)
        cmd = [_HEVC_FEAT_DECODER] + ["-i", self._filename] + ["-p", self._parallel]
        print(" ".join(cmd))
        self._proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=self.DEVNULL)

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.stdout.close()
            # self._proc.stderr.close()
            self._terminate(0.2)
        self._proc = None

    def _terminate(self, timeout=1.0):
        """Terminate the sub process."""
        # Check
        if self._proc is None:  # pragma: no cover
            return  # no process
        if self._proc.poll() is not None:
            return  # process already dead
        # Terminate process
        self._proc.terminate()
        # Wait for it to close (but do not get stuck)
        etime = time.time() + timeout
        while time.time() < etime:
            time.sleep(0.01)
            if self._proc.poll() is not None:
                break

    def _read_frame_data(self):
        self.pvY_size = self.width * self.height
        self.pvU_size = (self.width >> 1) * (self.height >> 1)
        self.pvV_size = (self.width >> 1) * (self.height >> 1)

        pvMV_size = (self.width >> 2) * (self.height >> 2) * 2
        pvOFF_size = (self.width >> 2) * (self.height >> 2)
        pvSize_size = (self.width >> 3) * (self.height >> 3)
        pvOffset = (3 * self.width * self.height >> 2) - (
            pvMV_size * 5 + pvOFF_size * 2
        )
        assert self._proc is not None

        try:

            arr_YUV420 = np.frombuffer(
                self._proc.stdout.read(self.pvY_size + self.pvU_size + self.pvV_size),
                dtype=np.uint8,
            )
            arr_MVX_L0 = np.frombuffer(
                self._proc.stdout.read(pvMV_size), dtype=np.int16
            )
            arr_MVY_L0 = np.frombuffer(
                self._proc.stdout.read(pvMV_size), dtype=np.int16
            )
            arr_MVX_L1 = np.frombuffer(
                self._proc.stdout.read(pvMV_size), dtype=np.int16
            )
            arr_MVY_L1 = np.frombuffer(
                self._proc.stdout.read(pvMV_size), dtype=np.int16
            )

            arr_REF_OFF_L0 = np.frombuffer(
                self._proc.stdout.read(pvOFF_size), dtype=np.uint8
            )
            arr_REF_OFF_L1 = np.frombuffer(
                self._proc.stdout.read(pvOFF_size), dtype=np.uint8
            )
            arr_Size = np.frombuffer(self._proc.stdout.read(pvMV_size), dtype=np.uint8)[
                :pvSize_size
            ]
            _ = self._proc.stdout.read(pvOffset)
            arr_meta = np.frombuffer(
                self._proc.stdout.read(self.pvY_size >> 2), dtype=np.uint8
            )
            arr_YUV420_residual = np.frombuffer(
                self._proc.stdout.read(self.pvY_size + self.pvU_size + self.pvV_size),
                dtype=np.uint8,
            )
            assert arr_meta[0] == 4 and arr_meta[1] == 2
            assert len(arr_meta) == self.pvY_size >> 2

        except Exception as e:
            print(e)
            self._terminate()
            raise RuntimeError(
                "Failed to decode video. video information: ", self.viddict
            )

        return (
            arr_meta,
            arr_YUV420,
            arr_MVX_L0,
            arr_MVY_L0,
            arr_MVX_L1,
            arr_MVY_L1,
            arr_REF_OFF_L0,
            arr_REF_OFF_L1,
            arr_Size,
            arr_YUV420_residual,
        )

    def _readFrame(self):
        (
            arr_meta,
            arr_YUV420,
            arr_MVX_L0,
            arr_MVY_L0,
            arr_MVX_L1,
            arr_MVY_L1,
            arr_REF_OFF_L0,
            arr_REF_OFF_L1,
            arr_Size,
            arr_YUV420_residual,
        ) = self._read_frame_data()

        frame_type = arr_meta[2]
        quadtree_stru = arr_meta[1024 : 1024 + self.nb_ctus * 12]

        all_yuv_data = arr_YUV420.reshape(self.height + (self.height >> 1), self.width)
        all_yuv_data_residual = arr_YUV420_residual.reshape(
            self.height + (self.height >> 1), self.width
        )

        rgb = cv2.cvtColor(all_yuv_data, cv2.COLOR_YUV420p2BGR)
        residual = cv2.cvtColor(all_yuv_data_residual, cv2.COLOR_YUV420p2BGR)

        mv_x_L0 = arr_MVX_L0.reshape(self.height >> 2, self.width >> 2)
        mv_y_L0 = arr_MVY_L0.reshape(self.height >> 2, self.width >> 2)
        mv_x_L1 = arr_MVX_L1.reshape(self.height >> 2, self.width >> 2)
        mv_y_L1 = arr_MVY_L1.reshape(self.height >> 2, self.width >> 2)
        ref_off_L0 = arr_REF_OFF_L0.reshape(self.height >> 2, self.width >> 2)
        ref_off_L1 = arr_REF_OFF_L1.reshape(self.height >> 2, self.width >> 2)

        size = arr_Size.reshape(self.height >> 3, self.width >> 3)

        self._lastread = (
            frame_type,
            quadtree_stru,
            rgb,
            mv_x_L0,
            mv_y_L0,
            mv_x_L1,
            mv_y_L1,
            ref_off_L0,
            ref_off_L1,
            size,
            residual,
        )

        return self._lastread

    def nextFrame(self):
        """Yields hevc features using a generator"""
        for i in range(self.nb_frames):
            yield self._readFrame()

    def getFrameNums(self):
        return self.nb_frames

    def getShape(self):
        return self.width, self.height

    def getDecodeOrder(self):
        return self.decode_order
