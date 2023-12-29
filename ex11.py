import cv2
import numpy as np

filename = "./images/lena.jpeg"
gray = cv2.imread(filename, 1)

# ORB
detector_orb = cv2.ORB_create()
# FAST
detector_fast = cv2.FastFeatureDetector_create()
# AKAZE
detector_akaze = cv2.AKAZE_create()
# BRISK
detector_brisk = cv2.BRISK_create()
# KAZE
detector_kaze = cv2.KAZE_create()
# SIFT
detector_shift = cv2.ORB_create()
# MSER
detector_mser = cv2.MSER_create()
# AgastFeatureDetector
detector_agast = cv2.AgastFeatureDetector_create()

def apt1():
    # orb
    keypoints_orb = detector_orb.detect(gray)
    # fast
    keypoints_fast = detector_fast.detect(gray)
    # akaze
    keypoints_akaze = detector_akaze.detect(gray)
    # brisk
    keypoints_brisk = detector_brisk.detect(gray)
    # kaze
    keypoints_kaze = detector_kaze.detect(gray)
    # sift
    keypoints_sift = detector_shift.detect(gray)
    # mser
    keypoints_mser = detector_mser.detect(gray)
    # agast
    keypoints_agast = detector_agast.detect(gray)

    # orb
    img_orb = cv2.drawKeypoints(gray, keypoints_orb, None)
    # fast
    img_fast = cv2.drawKeypoints(gray, keypoints_fast, None)
    # akaze
    img_akaze = cv2.drawKeypoints(gray, keypoints_akaze, None)
    # brisk
    img_brisk = cv2.drawKeypoints(gray, keypoints_brisk, None)
    # kaze
    img_kaze = cv2.drawKeypoints(gray, keypoints_kaze, None)
    # sift
    img_sift = cv2.drawKeypoints(gray, keypoints_sift, None)
    # mser
    img_mser = cv2.drawKeypoints(gray, keypoints_mser, None)
    # agast
    img_agast = cv2.drawKeypoints(gray, keypoints_agast, None)


    cv2.imshow("ORB", img_orb)
    cv2.imshow("FAST", img_fast)
    cv2.imshow("AKAZE", img_akaze)
    cv2.imshow("BRISK", img_brisk)
    cv2.imshow("KAZE", img_kaze)
    cv2.imshow("SIFT", img_sift)
    cv2.imshow("MSER", img_mser)
    cv2.imshow("AgastFeatureDetector", img_agast)

    cv2.waitKey(0)

def apt2():
    cap = cv2.VideoCapture(1)

    while True:
        key = cv2.waitKey(1)
        if key != -1:
            break
        ret, src = cap.read()


        filename = "./images/tmpmint.jpg"
        gray = cv2.imread(filename, 1)

        size = tuple([gray.shape[1], gray.shape[0]])
        center = tuple([int(size[0] / 2), int(size[1] / 2)])
        angle = 0.0
        scale = 1.0
        rot_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rot_img = cv2.warpAffine(gray, rot_matrix, size, flags=cv2.INTER_CUBIC)




        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # def orb():
        #     kp1_orb, dest1_orb = detector_orb.detectAndCompute(src, None)
        #     kp2_orb, dest2_orb = detector_orb.detectAndCompute(rot_img, None)
        #     matches_orb = bf.match(dest1_orb, dest2_orb)
        #     matches_orb = sorted(matches_orb, key=lambda x: x.distance)
        #     match_img_orb = cv2.drawMatches(src, kp1_orb, rot_img, kp2_orb, matches_orb[:20], None, flags=2)
        #     cv2.imshow("Match_orb", match_img_orb)

        def akaze():
            kp1_akaze, dest1_akaze = detector_akaze.detectAndCompute(src, None)
            kp2_akaze, dest2_akaze = detector_akaze.detectAndCompute(rot_img, None)
            matches_akaze = bf.match(dest1_akaze, dest2_akaze)
            matches_akaze = sorted(matches_akaze, key=lambda x: x.distance)
            match_img_akaze = cv2.drawMatches(src, kp1_akaze, rot_img, kp2_akaze, matches_akaze[:20], None, flags=2)
            cv2.imshow("Match_akaze", match_img_akaze)

        # def brisk():
        #     kp1_brisk, dest1_brisk = detector_brisk.detectAndCompute(src, None)
        #     kp2_brisk, dest2_brisk = detector_brisk.detectAndCompute(rot_img, None)
        #     matches_brisk = bf.match(dest1_brisk, dest2_brisk)
        #     matches_brisk = sorted(matches_brisk, key=lambda x: x.distance)
        #     match_img_brisk = cv2.drawMatches(src, kp1_brisk, rot_img, kp2_brisk, matches_brisk[:20], None, flags=2)
        #     cv2.imshow("Match_brisk", match_img_brisk)

        # def shift():
        #     kp1_shift, dest1_shift = detector_shift.detectAndCompute(src, None)
        #     kp2_shift, dest2_shift = detector_shift.detectAndCompute(rot_img, None)
        #     matches_shift = bf.match(dest1_shift, dest2_shift)
        #     matches_shift = sorted(matches_shift, key=lambda x: x.distance)
        #     match_img_shift = cv2.drawMatches(src, kp1_shift, rot_img, kp2_shift, matches_shift[:20], None, flags=2)
        #     cv2.imshow("Match_shift", match_img_shift)

        akaze()
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()

def apt3():
    return


if __name__ == "__main__":
    # apt1()
    apt2()