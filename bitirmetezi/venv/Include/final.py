import cv2
import numpy as np
import dlib
from math import hypot
import pyglet
import time

# Sesleri yükleme kısmı
ses = pyglet.media.load("ses.wav", streaming=False)
sol_ses = pyglet.media.load("sol.m4a", streaming=False)
sag_ses = pyglet.media.load("sağ.m4a", streaming=False)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #vcv2.CAP_DSHOW yazmayınca program çalışmasına rağmen hata veriyor.
yazi = np.zeros((300, 1400), np.uint8)
yazi[:] = 255

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Klavye ayarları
klavye = np.zeros((600, 1000, 3), np.uint8)
sol_harfler = {0: "A", 1: "B", 2: "C", 3: "Y", 4: "D",
              5: "E", 6: "F", 7: "G", 8: "Z", 9: "H",
              10: "I", 11: "M", 12: "J", 13: "K", 14: "<"}
sag_harfler = {0: "L", 1: "M", 2: "N", 3: "O", 4: "Q",
              5: "P", 6: "R", 7: "S", 8: "W", 9: "_",
              10: "T", 11: "U", 12: "X", 13: "V", 14: "<"}

def harf_ciz(harf_index, text, harf_tasarim):
    # harfler için -konum
    if harf_index == 0:
        x = 0
        y = 0
    elif harf_index == 1:
        x = 200
        y = 0
    elif harf_index == 2:
        x = 400
        y = 0
    elif harf_index == 3:
        x = 600
        y = 0
    elif harf_index == 4:
        x = 800
        y = 0
    elif harf_index == 5:
        x = 0
        y = 200
    elif harf_index == 6:
        x = 200
        y = 200
    elif harf_index == 7:
        x = 400
        y = 200
    elif harf_index == 8:
        x = 600
        y = 200
    elif harf_index == 9:
        x = 800
        y = 200
    elif harf_index == 10:
        x = 0
        y = 400
    elif harf_index == 11:
        x = 200
        y = 400
    elif harf_index == 12:
        x = 400
        y = 400
    elif harf_index == 13:
        x = 600
        y = 400
    elif harf_index == 14:
        x = 800
        y = 400

    width = 200
    height = 200
    th = 3 # kalınlık

    # Metin ayarları
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y

    if harf_tasarim is True:
        cv2.rectangle(klavye, (x + th, y + th), (x + width - th, y + height - th), (255, 238, 204), -1)
        cv2.putText(klavye, text, (text_x, text_y), font_letter, font_scale, (51, 102, 153), font_th)
    else:
        cv2.rectangle(klavye, (x + th, y + th), (x + width - th, y + height - th), (255, 153, 0), -1)
        cv2.putText(klavye, text, (text_x, text_y), font_letter, font_scale, (51, 102, 153), font_th)

def menu_tasarim():
    rows, cols, _ = klavye.shape
    th_lines = 4 # kalınlık çizgileri
    cv2.line(klavye, (int(cols/2) - int(th_lines/2), 0),(int(cols/2) - int(th_lines/2), rows),
             (51, 51, 51), th_lines)
    cv2.putText(klavye, "SOL", (80, 300), font, 6, (255, 255, 255), 5)
    cv2.putText(klavye, "SAG", (80 + int(cols/2), 300), font, 6, (255, 255, 255), 5)

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN
#yüz bölgesi kırpma oranları
def kirpma_orani(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio
#göz çevresi noktaları
def goz_cevresi_noktaları(facial_landmarks):
    sol_goz = []
    sag_goz = []
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        sol_goz.append([x, y])
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        sag_goz.append([x, y])
    sol_goz = np.array(sol_goz, np.int32)
    sag_goz = np.array(sag_goz, np.int32)
    return sol_goz, sag_goz
#bakış oranı alma
def bakis_orani_al(eye_points, facial_landmarks):
    sol_goz_orani = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [sol_goz_orani], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [sol_goz_orani], True, 255, 2)
    cv2.fillPoly(mask, [sol_goz_orani], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(sol_goz_orani[:, 0])
    max_x = np.max(sol_goz_orani[:, 0])
    min_y = np.min(sol_goz_orani[:, 1])
    max_y = np.max(sol_goz_orani[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    sol_kisim_threshold = threshold_eye[0: height, 0: int(width / 2)]
    sol_kisim_white = cv2.countNonZero(sol_kisim_threshold)

    sag_kisim_threshold = threshold_eye[0: height, int(width / 2): width]
    sag_kisim_white = cv2.countNonZero(sag_kisim_threshold)

    if sol_kisim_white == 0:
        bakis_orani = 1
    elif sag_kisim_white == 0:
        bakis_orani = 5
    else:
        bakis_orani = sol_kisim_white / sag_kisim_white
    return bakis_orani

# Counters
frames = 0
harf_index = 0
blinking_frames = 0
frames_to_blink = 6
frames_active_letter = 9

# Text and keyboard settings
text = ""
keyboard_selected = "left"
last_keyboard_selected = "left"
select_keyboard_menu = True
keyboard_selection_frames = 0

while True:
    _, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=0.8, fy=0.8)
    rows, cols, _ = frame.shape
    klavye[:] = (26, 26, 26)
    frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw a white space for loading bar
    frame[rows - 50: rows, 0: cols] = (255, 255, 255)

    if select_keyboard_menu is True:
        menu_tasarim()

    # Keyboard selected
    if keyboard_selected == "left":
        keys_set = sol_harfler
    else:
        keys_set = sag_harfler
    active_letter = keys_set[harf_index]

    # Face detection
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        sol_goz, sag_goz = goz_cevresi_noktaları(landmarks)

        # Detect blinking
        sol_goz_orani = kirpma_orani([36, 37, 38, 39, 40, 41], landmarks)
        sag_goz_orani = kirpma_orani([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (sol_goz_orani + sag_goz_orani) / 2

        # Eyes color
        cv2.polylines(frame, [sol_goz], True, (0, 0, 255), 2)
        cv2.polylines(frame, [sag_goz], True, (0, 0, 255), 2)


        if select_keyboard_menu is True:
            # Detecting gaze to select Left or Right keybaord
            sol_goz_bakis_orani = bakis_orani_al([36, 37, 38, 39, 40, 41], landmarks)
            sag_goz_bakis_orani = bakis_orani_al([42, 43, 44, 45, 46, 47], landmarks)
            bakis_orani = (sag_goz_bakis_orani + sol_goz_bakis_orani) / 2

            if bakis_orani <= 0.9:
                keyboard_selected = "sag"
                keyboard_selection_frames += 1
                # Bakışları bir tarafa 15 kareden fazla tutunca klavyeye gidilecek
                if keyboard_selection_frames == 15:
                    select_keyboard_menu = False
                    sag_ses.play()
                    # Klavye seçildiğinde kare sayısını 0 olarak ayarlama
                    frames = 0
                    keyboard_selection_frames = 0
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_frames = 0
            else:
                keyboard_selected = "sol"
                keyboard_selection_frames += 1
                # Bakışları bir tarafa 15 kareden fazla tutunca klavyeye gitmek için
                if keyboard_selection_frames == 15:
                    select_keyboard_menu = False
                    sol_ses.play()
                    # Klavye seçildiğinde kare sayısını 0 olarak ayarlamak için
                    frames = 0
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_frames = 0

        else:
            # Yanan tuşu seçmek için yanıp sönmeyi algılamak için
            if blinking_ratio > 5:
                # cv2.putText(frame, "KIRPMA", (50, 150), font, 4, (255, 0, 0), thickness=3)
                blinking_frames += 1
                frames -= 1

                # Kapalıyken gözleri yeşil gösterme
                cv2.polylines(frame, [sol_goz], True, (0, 255, 0), 2)
                cv2.polylines(frame, [sag_goz], True, (0, 255, 0), 2)

                # harf yaz
                if blinking_frames == frames_to_blink:
                    if active_letter != "<" and active_letter != "_":
                        text += active_letter
                    if active_letter == "_":
                        text += " "
                    ses.play()
                    select_keyboard_menu = True
                    # time.sleep(1)

            else:
                blinking_frames = 0


    # Klavyede harfleri görüntüleme
    if select_keyboard_menu is False:
        if frames == frames_active_letter:
            harf_index += 1
            frames = 0
        if harf_index == 15:
            harf_index = 0
        for i in range(15):
            if i == harf_index:
                light = True
            else:
                light = False
            harf_ciz(i, keys_set[i], light)

    # ekranda yazdığımız metni gösterme
    cv2.putText(yazi, text, (80, 100), font, 9, 0, 3)

    # Kırpma yükleme bar'ı
    percentage_blinking = blinking_frames / frames_to_blink
    loading_x = int(cols * percentage_blinking)
    cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)


    cv2.imshow("Frame", frame)
    cv2.imshow("Sanal Klavye", klavye)
    cv2.imshow("Yazi Tahtasi", yazi)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()