"""
magnify_casme_demo
Copyright 2024 by Philipp Flotho, All rights reserved.
"""

import cv2

from os.path import isdir, join
from os import mkdir

from neurovc.momag import OnlineLandmarkMagnifier
import neurovc as nvc


if __name__ == '__main__':

    # Put the base folder of CASME2-RAW here:
    casme_folder = ""
    file_path = join(casme_folder, "sub14\\EP04_04f")

    ref = cv2.imread(join(file_path, 'img1.jpg'))
    frame = cv2.imread(join(file_path, 'img32.jpg'))

    if ref is None or frame is None:
        print("images could not be loaded!")

    motion_magnifier = OnlineLandmarkMagnifier(landmarks=nvc.LM_MOUTH + nvc.LM_EYE_LEFT + nvc.LM_EYE_RIGHT,
                                               reference=ref, alpha=15)
    magnified, (flow, flow_global, flow_local) = motion_magnifier(frame)
    ref, _ = motion_magnifier.get_reference()

    output_folder = 'test'
    if not isdir(output_folder):
        mkdir(output_folder)

    cv2.imwrite(join(output_folder, 'flow_local.png'), flow_local)
    cv2.imwrite(join(output_folder, 'flow_combined.png'), flow)
    cv2.imwrite(join(output_folder, 'flow_global.png'), flow_global)
    cv2.imwrite(join(output_folder, 'landmarks.png'), ref)
    cv2.imwrite(join(output_folder, 'frame.png'), frame)
    cv2.imwrite(join(output_folder, 'magnified.png'), magnified)

    cv2.imshow("magnified", magnified)
    cv2.imshow("test2", ref)
    cv2.imshow("test", frame)
    cv2.waitKey()
