import cv2


def op_one_img(i):
    return i



# videoinpath = "/home/SENSETIME/renqin/PycharmProjects/DeOldify-demo/test.mp4"
videoinpath = ""
capture = cv2.VideoCapture(videoinpath)

videooutpath = "/home/SENSETIME/renqin/PycharmProjects/DeOldify-demo/test_1.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = capture.get(5)
size = (int(capture.get(3)), int(capture.get(4)))  # 宽度、高度
writer = cv2.VideoWriter(videooutpath, fourcc, fps, size, True)

while True:
    ret, img_src = capture.read()
    if not ret:
        break
    img_out = op_one_img(img_src)  # 自己写函数op_one_img()逐帧处理
    writer.write(img_out)
writer.release()



