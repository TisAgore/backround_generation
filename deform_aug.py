import cv2
import random
import numpy as np
from Information_extraction import information_extraction
from transformation import flag_judge, identify_reference_corner
from transformation import bezier_transformation, affine_transformation, L2A_transformation



def new_local(src, times=1, stroke_radius=2, k1_control_field_corner=0.6, k2_control_field_third_bezier=0.6, segment=2):

    def deformation(list_all):
        """
        [[list_cor], [list_ske], [list_bezier_use_information]]
          /////[list_bezier_use_information] = [[pt_cor1, pt_cor2],[pt_third_bezier],[俩角点控制域],[第三控制点控制域]]
                    |
                    |
        [[cor1, cor2],[ske]]
        """

        list_already, len_list_all, list_pt_reference = [], len(list_all), []
        for a in range(len_list_all):
            list_already.append([[[], []], []])

        for index_list_already in range(len_list_all):
            list_child_already = list_already[index_list_already]
            list_child_all = list_all[index_list_already]
            flag_situation_judge = flag_judge(list_child_already)

            if flag_situation_judge == 4:
                list_already[index_list_already] = bezier_transformation(list_child_already,
                                                                         list_child_all, flag_situation_judge)
                continue
            else:
                if flag_situation_judge == 1 and index_list_already != 0:
                    flag_situation_judge = 3
                    list_reference = identify_reference_corner(list_all[index_list_already][0][0], list_pt_reference)
                    index_temp1, index_temp2 = list_reference[2], list_reference[1]
                    x_change = list_already[index_temp1][0][index_temp2][0] - list_all[index_temp1][0][index_temp2][0]
                    y_change = list_already[index_temp1][0][index_temp2][1] - list_all[index_temp1][0][index_temp2][1]
                    x_get = list_all[index_list_already][0][0][0] + x_change
                    y_get = list_all[index_list_already][0][0][1] + y_change
                    x_already = random.uniform(x_get - list_all[index_list_already][2][0][2][0][0],
                                               x_get + list_all[index_list_already][2][0][2][0][0] + 0.1)
                    y_already = random.uniform(y_get - list_all[index_list_already][2][0][2][0][1],
                                               y_get + list_all[index_list_already][2][0][2][0][1] + 0.1)
                    list_child_already[0][0] = [x_already, y_already]

                choice = random.randint(1, 3)
                if choice == 1:
                    list_already[index_list_already] = bezier_transformation(
                        list_child_already, list_child_all, flag_situation_judge)
                elif choice == 2:
                    list_already[index_list_already] = affine_transformation(
                        list_child_already, list_child_all, flag_situation_judge, stroke_radius)
                else:
                    list_already[index_list_already] = L2A_transformation(
                        list_child_already, list_child_all, flag_situation_judge, segment, stroke_radius)

            for index_already_pt in range(2):
                pt_standard = list_all[index_list_already][0][index_already_pt]
                pt_change = list_already[index_list_already][0][index_already_pt]
                list_pt_reference.append((pt_standard, index_already_pt, index_list_already))
                for index_scan in range(index_list_already + 1, len_list_all):
                    for index_scan_child in range(2):
                        if pt_standard == list_all[index_scan][0][index_scan_child]:
                            list_already[index_scan][0][index_scan_child] = pt_change

        return list_already

    def draw_src(list_all, canvas_width, canvas_height):
        list_pt = []
        for a in list_all:
            b = a[1] 
            for pt in b:
                list_pt.append(pt)

        if not list_pt:
            image_film = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            image_film[:] = 255
            return image_film

        xs = [p[0] for p in list_pt]
        ys = [p[1] for p in list_pt]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        offset_x = canvas_width / 2.0 - center_x
        offset_y = canvas_height / 2.0 - center_y

        image_film = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

        for pt in list_pt:
            x_new = int(round(pt[0] + offset_x))
            y_new = int(round(pt[1] + offset_y))
            if 0 <= x_new < canvas_width and 0 <= y_new < canvas_height:
                cv2.circle(image_film, (x_new, y_new), stroke_radius, (0, 0, 0), -1)

        return image_film
    
    list_information = information_extraction(src, k1_control_field_corner, k2_control_field_third_bezier)
    height, width = src.shape[:2]
    list_draw = []
    while len(list_draw) < times:
        list_final = deformation(list_information)
        picture = draw_src(list_final, width, height)
        if picture.shape[2] != 3:
            continue
        list_draw.append(picture)

    return list_draw