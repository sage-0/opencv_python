import cv2

cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture("./images/testmovie3.avi")

def 背景差分():
    th = 30 # 差分画像の閾値

    # background
    ret, bg = cap.read()

    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = cv2.absdiff(gray, bg)

        mask[mask < th] = 0
        mask[mask >= th] = 255

        cv2.imshow("mask", mask)
        cv2.imshow("frame", frame)
        cv2.imshow("bg", bg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def フレーム間差分(time_interval=0.03, interval=30):
    import time

    i = 0
    th = 30 # 差分画像の閾値

    # background
    ret, bg = cap.read()

    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = cv2.absdiff(gray, bg)

        mask[mask < th] = 0
        mask[mask >= th] = 255

        cv2.imshow("mask", mask)
        cv2.imshow("frame", frame)
        cv2.imshow("bg", bg)

        time.sleep(time_interval)
        i += 1

        if(i > interval):
            ret, bg = cap.read()
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            i = 0


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def テンプレートマッチング():
    import numpy as np
    from matplotlib import pyplot as plt
    template = cv2.imread("./images/tmpmint.jpg")
    while True:
        ret,img = cap.read()
        if not ret:
            break

        _, w, h = template.shape[::-1]
        method=[
            cv2.TM_CCOEFF,
            cv2.TM_CCOEFF_NORMED,
            cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED,
            cv2.TM_SQDIFF,
            cv2.TM_SQDIFF_NORMED
            ]

        res_0 = cv2.matchTemplate(img, template, method[0])
        min_val_0, max_val_0, min_loc_0, max_loc_0 = cv2.minMaxLoc(res_0)
        top_left_0 = max_loc_0
        btm_right_0 = (top_left_0[0] + w, top_left_0[1] + h)

        cv2.rectangle(img,top_left_0, btm_right_0, 255, 2)
        cv2.imshow("TM_CCOEFF", img)

        res_1 = cv2.matchTemplate(img, template, method[1])
        min_val_1, max_val_1, min_loc_1, max_loc_1 = cv2.minMaxLoc(res_1)
        top_left_1 = max_loc_1
        btm_right_1 = (top_left_1[0] + w, top_left_1[1] + h)

        cv2.rectangle(img,top_left_1, btm_right_1, 255, 2)
        cv2.imshow("TM_CCOEFF_NORMED", img)

        res_2 = cv2.matchTemplate(img, template, method[2])
        min_val_2, max_val_2, min_loc_2, max_loc_2 = cv2.minMaxLoc(res_2)
        top_left_2 = max_loc_2
        btm_right_2 = (top_left_2[0] + w, top_left_2[1] + h)

        cv2.rectangle(img,top_left_2, btm_right_2, 255, 2)
        cv2.imshow("TM_CCORR", img)


        res_3 = cv2.matchTemplate(img, template, method[3])
        min_val_3, max_val_3, min_loc_3, max_loc_3 = cv2.minMaxLoc(res_3)
        top_left_3 = max_loc_3
        btm_right_3 = (top_left_3[0] + w, top_left_3[1] + h)

        cv2.rectangle(img,top_left_3, btm_right_3, 255, 2)
        cv2.imshow("TM_CCORR_NORMED", img)

        res_4 = cv2.matchTemplate(img, template, method[4])
        min_val_4, max_val_4, min_loc_4, max_loc_4 = cv2.minMaxLoc(res_4)
        top_left_4 = max_loc_4
        btm_right_4 = (top_left_4[0] + w, top_left_4[1] + h)

        cv2.rectangle(img,top_left_4, btm_right_4, 255, 2)
        cv2.imshow("TM_SQDIFF", img)

        res_5 = cv2.matchTemplate(img, template, method[5])
        min_val_5, max_val_5, min_loc_5, max_loc_5 = cv2.minMaxLoc(res_5)
        top_left_5 = max_loc_5
        btm_right_5 = (top_left_5[0] + w, top_left_5[1] + h)

        cv2.rectangle(img,top_left_5, btm_right_5, 255, 2)
        cv2.imshow("TM_SQDIFF_NORMED", img)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def オプティカルフロー():
    import numpy as np

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):
        ret,frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

def DISオプティカルフロー():
    import numpy as np
    import cv2 as cv

    def draw_flow(img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    def draw_hsv(flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,1] = 255
        hsv[...,2] = np.minimum(v*4, 255)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return bgr

    def warp_flow(img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        res = cv.remap(img, flow, None, cv.INTER_LINEAR)
        return res

    import sys
    # cam = cv.VideoCapture("./images/testmovie3.avi")
    cam = cv.VideoCapture(1)
    ret, prev = cam.read()
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    use_spatial_propagation = False
    use_temporal_propagation = True
    cur_glitch = prev.copy()
    inst = cv.DISOpticalFlow.create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    inst.setUseSpatialPropagation(use_spatial_propagation)
    flow = None
    while True:
        ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if flow is not None and use_temporal_propagation:
            #warp previous flow to get an initial approximation for the current flow:
            flow = inst.calc(prevgray, gray, warp_flow(flow,flow))
        else:
            flow = inst.calc(prevgray, gray, None)
        prevgray = gray
        cv.imshow('flow', draw_flow(gray, flow))
        if show_hsv:
            cv.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv.imshow('glitch', cur_glitch)
        ch = 0xFF & cv.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print('glitch is', ['off', 'on'][show_glitch])
        if ch == ord('3'):
            use_spatial_propagation = not use_spatial_propagation
            inst.setUseSpatialPropagation(use_spatial_propagation)
            print('spatial propagation is', ['off', 'on'][use_spatial_propagation])
        if ch == ord('4'):
            use_temporal_propagation = not use_temporal_propagation
            print('temporal propagation is', ['off', 'on'][use_temporal_propagation])

if __name__ == "__main__":
    # 背景差分()
    # フレーム間差分(0.01, 5)
    # テンプレートマッチング()
    # オプティカルフロー()
    # DISオプティカルフロー()
    cv2.destroyAllWindows
    cap.release()
