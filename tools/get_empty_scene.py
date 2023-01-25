import cv2, random, json
import numpy as np

img_R1_org = cv2.imread('/home/bak/Datasets/kisan_sample_data/1061/TRAY1/DR/TP1/BA000/V0_0_1_123622270639_R2_FOV090_ANG20_MIL500_LI3_TRAY1_DR_LY_TP1_BA000_1061_20221126_194802_Color.png')
img_T1_org = cv2.imread('/home/bak/Datasets/kisan_sample_data/1061/TRAY1/DR/TP1/BA000/V0_0_1_126122270804_T2_FOV090_ANG20_MIL500_LI3_TRAY1_DR_LY_TP1_BA000_1061_20221126_194802_Color.png')
img_L1_org = cv2.imread('/home/bak/Datasets/kisan_sample_data/1061/TRAY1/DR/TP1/BA000/V0_0_1_126122270991_L2_FOV090_ANG20_MIL500_LI3_TRAY1_DR_LY_TP1_BA000_1061_20221126_194802_Color.png')

img_R2_org = cv2.imread('/home/bak/Datasets/kisan_sample_data/1061/TRAY1/DR/TP9/BA000/V0_0_1_123622270639_R2_FOV090_ANG20_MIL500_LI3_TRAY1_DR_LY_TP9_BA000_1061_20221126_195920_Color.png')
img_T2_org = cv2.imread('/home/bak/Datasets/kisan_sample_data/1061/TRAY1/DR/TP9/BA000/V0_0_1_126122270804_T2_FOV090_ANG20_MIL500_LI3_TRAY1_DR_LY_TP9_BA000_1061_20221126_195920_Color.png')
img_L2_org = cv2.imread('/home/bak/Datasets/kisan_sample_data/1061/TRAY1/DR/TP9/BA000/V0_0_1_126122270991_L2_FOV090_ANG20_MIL500_LI3_TRAY1_DR_LY_TP9_BA000_1061_20221126_195920_Color.png')

half = int(img_R1_org.shape[1] / 2)

img_R1 = img_R1_org[:, half:, :]
img_R2 = img_R2_org[:, :half, :]
img_R = cv2.hconcat([img_R2, img_R1])
cv2.imwrite('tray_image/tray_R.png', img_R)

img_T1 = img_T1_org[:, half:, :]
img_T2 = img_T2_org[:, :half, :]
img_T = cv2.hconcat([img_T2, img_T1])
cv2.imwrite('tray_image/tray_T.png', img_T)

img_L1 = img_L1_org[:, half:, :]
img_L2 = img_L2_org[:, :half, :]
img_L = cv2.hconcat([img_L2, img_L1])
cv2.imwrite('tray_image/tray_L.png', img_L)
