from skimage import morphology, data, color
import cv2 as cv
import numpy as np
import math
import random
import copy


def information_extraction(src, k1_control_field_corner, k2_control_field_third_bezier):
    """
    [[list_cor], [list_ske], [list_bezier_use_information]]
     /////[list_bezier_use_information] = [[pt_cor1, pt_cor2],[pt_third_bezier],[俩角点控制域],[第三控制点控制域]]
    """

    # 计算两点距离
    def two_points_distance(point1, point2):
        p1 = point1[0] - point2[0]
        p2 = point1[1] - point2[1]
        distance = math.hypot(p1, p2)
        return distance

    # 计算点到两点所组成直线距离
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        if line_point1 == line_point2:
            point_array = np.array(point)
            point1_array = np.array(line_point1)
            return np.linalg.norm(point_array - point1_array)
        A = line_point2[1] - line_point1[1]
        B = line_point1[0] - line_point2[0]
        C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
            (line_point2[0] - line_point1[0]) * line_point1[1]
        distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A ** 2 + B ** 2))
        return distance

    # 骨架索引
    def skeleton_index(skeleton_src):
        index = []
        row, col = skeleton_src.shape
        for a in range(0, row):
            for b in range(0, col):
                if skeleton_src[a][b] == 1:
                    index.append((b, a))
        return index

    # 求多分支点和端点
    def corner_index(list_ske):
        list_corner = []
        for point in list_ske:
            num_branches = 0
            for a in range(-1, 2):
                for b in range(-1, 2):
                    if(point[0] + a, point[1] + b) in list_ske:
                        num_branches = num_branches + 1
            if num_branches != 3:
                list_corner.append(point)
        return list_corner

    # 去掉骨架上所有角点,只留下骨架
    def skeleton_clean(list_ske, list_corner):
        for pt in list_corner:
            list_ske.remove(pt)
        return list_ske

    # 角点去冗余
    def list_corner_clean(image_original, list_corner):
        image_original = image_original.astype(np.uint8) * 255
        image_original_copy = np.zeros_like(image_original)
        for pt in list_corner:
            cv.circle(image_original_copy, pt, 0, (255, 255, 255), -1)
        nums_label, labels = cv.connectedComponents(image_original_copy)
        list_total = []
        for a in range(0, nums_label - 1):
            list_total.append([])
        row, col = labels.shape
        for a in range(0, row):
            for b in range(0, col):
                if labels[a][b] > 0:
                    list_total[(labels[a][b] - 1)].append((b, a))
        list_corner_new = []
        for list in list_total:
            if len(list) == 1:
                list_corner_new.append(list[0])
            else:
                list_sort_a = []
                list_sort_b = []
                list_distance = []
                for pt in list:
                    list_sort_a.append(pt[0])
                    list_sort_b.append(pt[1])
                list_sort_a.sort()
                list_sort_b.sort()
                mid_a = (list_sort_a[0] + list_sort_a[-1]) / 2
                mid_b = (list_sort_b[0] + list_sort_b[-1]) / 2
                for pt in list:
                    list_distance.append(two_points_distance((mid_a, mid_b), pt))
                list_corner_new.append(list[list_distance.index(min(list_distance))])
        return list_corner_new

    # 骨架连通域检测
    def skeleton_connected(image_original, ske_index):
        image_original = image_original.astype(np.uint8) * 255
        image_original_copy = np.zeros_like(image_original)
        for pt in ske_index:
            cv.circle(image_original_copy, pt, 0, (255, 255, 255), 0)
        nums_label, labels = cv.connectedComponents(image_original_copy)
        list_total = []
        for a in range(0, nums_label-1):
            list_total.append([])
        row, col = labels.shape
        for a in range(0, row):
            for b in range(0, col):
                if labels[a][b] > 0:
                    list_total[(labels[a][b]-1)].append((b, a))
        return list_total

    # ---------- ИСПРАВЛЕННАЯ ФУНКЦИЯ ske_cor_match ----------
    def ske_cor_match(list_ske, list_cor):
        """
        [[end_pt1, end_pt2], [cor_pt1, cor_pt2], [list_ske_pt]]
        """
        list_total = []
        for list in list_ske:
            list_integration = []
            list_endpoint = []
            list_corner = []
            # 骨架数量大于等于2的情况
            if len(list) >= 2:
                # 先数分支数，寻找端点 (8-邻域, ровно 2 соседа)
                for pt in list:
                    count_branches = 0
                    for a in range(-1, 2):
                        for b in range(-1, 2):
                            if (pt[0] + a, pt[1] + b) in list:
                                count_branches = count_branches + 1
                    if count_branches == 2:
                        list_endpoint.append(pt)

                # ЕСЛИ НЕТ ДВУХ КОНЦЕВЫХ ТОЧЕК – берём две наиболее удалённые точки
                if len(list_endpoint) < 2:
                    max_dist = -1
                    pt1, pt2 = list[0], list[0]
                    for i in range(len(list)):
                        for j in range(i+1, len(list)):
                            d = two_points_distance(list[i], list[j])
                            if d > max_dist:
                                max_dist = d
                                pt1, pt2 = list[i], list[j]
                    list_endpoint = [pt1, pt2]

                list_integration.append(list_endpoint)
                # 找到端点后，匹配距离最近的角点
                for pt_end in list_endpoint:
                    list_distance = []
                    for pt_corner in list_cor:
                        list_distance.append(two_points_distance(pt_end, pt_corner))
                    point_corner = list_cor[list_distance.index(min(list_distance))]
                    list_corner.append(point_corner)
                list_integration.append(list_corner)
                list_integration.append(list)   # сам скелет
                list_total.append(list_integration)

            # 骨架数量等于1的情况
            if len(list) == 1:
                pt = list[0]
                list_endpoint = [pt, pt]
                list_integration.append(list_endpoint)
                list_distance = []
                for pt_corner in list_cor:
                    list_distance.append(two_points_distance(pt, pt_corner))
                list_copy_distance = copy.copy(list_distance)
                list_copy_distance.sort()
                d_min, d_second_min = list_copy_distance[0], list_copy_distance[1]
                pt_cor1, pt_cor2 = list_cor[list_distance.index(d_min)], list_cor[list_distance.index(d_second_min)]
                list_corner = [pt_cor1, pt_cor2]
                list_integration.append(list_corner)
                list_integration.append(list)   # скелет из одной точки
                list_total.append(list_integration)

        # 考虑有孤立角点存在的情况
        list_cor_wait_clean = []
        for list in list_total:
            for pt in list[1]:
                list_cor_wait_clean.append(pt)
        list_cor_clean = set(list_cor_wait_clean)
        for pt in list_cor_clean:
            list_cor.remove(pt)
        if len(list_cor) != 0:
            for pt in list_cor:
                # СОЗДАЁМ ПРАВИЛЬНУЮ СТРУКТУРУ: [endpoints, corners, skeleton]
                list_integration = []
                list_endpoint = [pt, pt]
                list_corner = [pt, pt]
                list_ske = [pt]          # скелет – одна точка
                list_integration.append(list_endpoint)
                list_integration.append(list_corner)
                list_integration.append(list_ske)
                list_total.append(list_integration)

        return list_total

    # ---------- ИСПРАВЛЕННАЯ ФУНКЦИЯ ske_rearrangement (добавлена защита) ----------
    def ske_rearrangement(list_total):
        """
         [[cor1_1,cor1_2],[ske_list1]]
        """
        for a in list_total:
            # Если по какой-то причине a[0] пуст – пропускаем (хотя теперь такого быть не должно)
            if len(a[0]) < 2:
                continue
            point_temp = a[0][0]
            list_ske_new = []
            while point_temp != a[0][1]:
                list_ske_new.append(point_temp)
                a[2].remove(point_temp)
                list_temp = []
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if (point_temp[0]+x, point_temp[1]+y) in a[2]:
                            list_temp.append((point_temp[0]+x, point_temp[1]+y))
                point_temp = list_temp[0]
            list_ske_new.append(a[0][1])
            # 重构原来的list_from_ske_cor_match
            del(a[2])
            del(a[0])
            a.append(list_ske_new)

        return list_total

    # 角点增补 (без изменений)
    def cor_addition(list_total):
        """
        [[list_cor], [list_ske], [list_bezier_use_information]]
            ///////[list_bezier_use_information] = [[list_cor], [list_ske]]
        """
        for index_list_total in range(len(list_total)):
            list_bezier_information, list_bezier_cor, list_bezier_ske, list_integration = [], [], [], []
            child_list_total = list_total[index_list_total]
            len_ske = len(child_list_total[1])
            if len_ske <= 4:
                list_no_addition = copy.copy(child_list_total)
                list_bezier_information.append(list_no_addition)
                list_total[index_list_total].append(list_bezier_information)
                continue

            list_cor = child_list_total[0]
            list_ske = child_list_total[1]
            list_bezier_cor.append(list_cor[0])
            flag_first_pt = 1
            flag_need_new_direction = 1
            pt_cor = list_cor[0]
            flag_judge_s_change = 0
            target_arc_x, target_arc_y = 0, 0
            flag_judge_integration = 0
            list_judge_three_direction = []
            x, y = 0, 0
            index_ske = 0

            for index_ske in range(len_ske-3):
                if flag_need_new_direction == 1:
                    flag_need_new_direction = 0
                    if flag_first_pt == 1:
                        flag_first_pt = 0
                        pt_cor = list_cor[0]
                        flag_judge_near = 0
                        for a in range(-1, 2):
                            for b in range(-1, 2):
                                if list_ske[index_ske][0] + a == pt_cor[0] and list_ske[index_ske][1] + b == pt_cor[1]:
                                    flag_judge_near = 1
                                    x, y = list_ske[0][0] - pt_cor[0], list_ske[0][1] - pt_cor[1]
                        if flag_judge_near == 0:
                            x, y = list_ske[1][0] - list_ske[0][0], list_ske[1][1] - list_ske[0][1]
                    else:
                        x, y = list_ske[index_ske][0] - pt_cor[0], list_ske[index_ske][1] - pt_cor[1]

                    target_arc_x, target_arc_y = -x, -y
                    if (x, y) == (-1, -1):
                        list_judge_three_direction = [(-1, -1), (-1, 0), (0, -1)]
                    elif (x, y) == (-1, 0):
                        list_judge_three_direction = [(-1, 0), (-1, 1), (-1, -1)]
                    elif (x, y) == (-1, 1):
                        list_judge_three_direction = [(-1, 1), (-1, 0), (0, 1)]
                    elif (x, y) == (0, 1):
                        list_judge_three_direction = [(0, 1), (-1, 1), (1, 1)]
                    elif (x, y) == (1, 1):
                        list_judge_three_direction = [(1, 1), (0, 1), (1, 0)]
                    elif (x, y) == (1, 0):
                        list_judge_three_direction = [(1, 0), (1, 1), (1, -1)]
                    elif (x, y) == (1, -1):
                        list_judge_three_direction = [(1, -1), (1, 0), (0, -1)]
                    elif (x, y) == (0, -1):
                        list_judge_three_direction = [(0, -1), (-1, -1), (1, -1)]

                direction_ske_x = list_ske[index_ske+1][0] - list_ske[index_ske][0]
                direction_ske_y = list_ske[index_ske+1][1] - list_ske[index_ske][1]

                if target_arc_x == direction_ske_x and target_arc_y == direction_ske_y:
                    flag_judge_integration = 1
                elif flag_judge_s_change == 1:
                    if direction_ske_x == x and direction_ske_y == y:
                        flag_judge_integration = 1
                elif flag_judge_s_change == 0:
                    if (direction_ske_x, direction_ske_y) not in list_judge_three_direction:
                        flag_judge_s_change = 1

                if flag_judge_integration == 0:
                    list_bezier_ske.append(list_ske[index_ske])
                elif flag_judge_integration == 1:
                    list_bezier_cor.append(list_ske[index_ske])
                    list_integration.append(list_bezier_cor), list_integration.append(list_bezier_ske)
                    list_bezier_information.append(list_integration)
                    list_bezier_cor, list_bezier_ske, list_integration = [], [], []
                    pt_cor = list_ske[index_ske]
                    list_bezier_cor.append(pt_cor)
                    flag_judge_s_change, flag_judge_integration, flag_need_new_direction = 0, 0, 1

            list_bezier_cor.append(list_cor[1])
            list_bezier_ske.extend(list_ske[index_ske + 1:])
            list_integration.append(list_bezier_cor), list_integration.append(list_bezier_ske)
            list_bezier_information.append(list_integration)
            list_total[index_list_total].append(list_bezier_information)

        return list_total

    # 找骨架最突出点和第三贝塞尔曲线点 (без изменений)
    def third_bezier_curve_point(list_total):
        for index_list_total in range(len(list_total)):
            list_bezier_information = list_total[index_list_total][2]
            for index_child_bezier in range(len(list_bezier_information)):
                list_child_bezier = list_bezier_information[index_child_bezier]
                list_ske, list_distance_ske = list_child_bezier[1], []
                pt_cor1, pt_cor2 = list_child_bezier[0][0], list_child_bezier[0][1]
                for pt_ske in list_ske:
                    list_distance_ske.append(get_distance_from_point_to_line(pt_ske, pt_cor1, pt_cor2))
                pt_bulge = list_ske[list_distance_ske.index(max(list_distance_ske))]
                del(list_total[index_list_total][2][index_child_bezier][1])
                pt_center_x = 0.5 * float(pt_cor1[0]) + 0.5 * float(pt_cor2[0])
                pt_center_y = 0.5 * float(pt_cor1[1]) + 0.5 * float(pt_cor2[1])
                pt_third_bezier_x = int((float(pt_bulge[0]) - pt_center_x) * 2 + pt_center_x)
                pt_third_bezier_y = int((float(pt_bulge[1]) - pt_center_y) * 2 + pt_center_y)
                pt_third_bezier = [(pt_third_bezier_x, pt_third_bezier_y)]
                list_total[index_list_total][2][index_child_bezier].append(pt_third_bezier)
        return list_total

    # 控制域计算 (без изменений)
    def control_field(list_total, k1, k2):
        list_cor_all = []
        for p in list_total:
            for q in p[2]:
                for pt in q[0]:
                    list_cor_all.append(pt)
        list_cor_all = list(set(list_cor_all))

        for index_list_total in range(len(list_total)):
            list_bezier_information = list_total[index_list_total][2]
            for index_child_bezier in range(len(list_bezier_information)):
                list_child_bezier = list_bezier_information[index_child_bezier]
                list_cor, pt_third_bezier = list_child_bezier[0], list_child_bezier[1][0]
                pt_cor1, pt_cor2 = list_cor[0], list_cor[1]
                list_control_field_corner, list_control_field_third_bezier = [], []

                for pt_cor in list_cor:
                    list_temp = copy.copy(list_cor_all)
                    list_temp.remove(pt_cor)
                    list_distance = []
                    for pt_temp in list_temp:
                        list_distance.append(two_points_distance(pt_temp, pt_cor))
                    pt_corner_min_distance = list_temp[list_distance.index(min(list_distance))]
                    width_control_field = float(abs(pt_cor[0] - pt_corner_min_distance[0]) * k1)
                    height_control_field = float(abs(pt_cor[1] - pt_corner_min_distance[1]) * k1)
                    list_control_field_corner.append([width_control_field, height_control_field])

                len_rectangular = float(get_distance_from_point_to_line(pt_third_bezier, pt_cor1, pt_cor2))
                list_control_field_third_bezier.append(float(len_rectangular * k2))

                list_total[index_list_total][2][index_child_bezier].append(list_control_field_corner)
                list_total[index_list_total][2][index_child_bezier].append(list_control_field_third_bezier)

        return list_total

    # 画出来看用的 (без изменений)
    def draw_temp(skeleton_src, list_total):
        skeleton_image = skeleton_src.astype(np.uint8) * 255
        skeleton_image = np.zeros_like(skeleton_image)
        skeleton_image = cv.cvtColor(skeleton_image, cv.COLOR_GRAY2BGR)
        list_color = [(60, 230, 150), (230, 150, 150), (255, 80, 10), (60, 60, 60), (10, 70, 250),
                      (50, 255, 190)]
        for k in list_total:
            col = random.randint(0, 5)
            for pt in k[1]:
                cv.circle(skeleton_image, pt, 0, list_color[col], cv.FILLED)
            for g in k[2]:
                for pt in g[0]:
                    cv.circle(skeleton_image, pt, 0, (0, 255, 0), cv.FILLED)
                pt_third_bezier = g[1][0]
                cv.circle(skeleton_image, pt_third_bezier, 0, (255, 255, 255), cv.FILLED)
                pt_cor1, pt_cor2 = g[0][0], g[0][1]
                list_control_field_corner,  list_control_field_third_bezier = g[2], g[3]
                cv.rectangle(img=skeleton_image,
                             pt1=(int(pt_cor1[0])-int(list_control_field_corner[0][0]),
                                  int(pt_cor1[1])+int(list_control_field_corner[0][1])),
                             pt2=(int(pt_cor1[0])+int(list_control_field_corner[0][0]),
                                  int(pt_cor1[1])-int(list_control_field_corner[0][1])),
                             color=(255, 255, 0), thickness=1)
                cv.rectangle(img=skeleton_image,
                             pt1=(int(pt_cor2[0]) - int(list_control_field_corner[1][0]),
                                  int(pt_cor2[1]) + int(list_control_field_corner[1][1])),
                             pt2=(int(pt_cor2[0]) + int(list_control_field_corner[1][0]),
                                  int(pt_cor2[1]) - int(list_control_field_corner[1][1])),
                             color=(255, 255, 0), thickness=1)
                cv.rectangle(img=skeleton_image,
                             pt1=(int(pt_third_bezier[0]) - int(list_control_field_third_bezier[0]),
                                  int(pt_third_bezier[1]) + int(list_control_field_third_bezier[0])),
                             pt2=(int(pt_third_bezier[0]) + int(list_control_field_third_bezier[0]),
                                  int(pt_third_bezier[1]) - int(list_control_field_third_bezier[0])),
                             color=(0, 255, 0), thickness=1)
        return skeleton_image

    # Основной процесс
    img = src
    img = cv.blur(img, (3, 3))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = (gray < 240)
    skeleton_image = morphology.skeletonize(gray)

    list_skeleton = skeleton_index(skeleton_image)
    list_corner_original = corner_index(list_skeleton)
    list_skeleton = skeleton_clean(list_skeleton, list_corner_original)
    list_corner_new = list_corner_clean(skeleton_image, list_corner_original)
    list_skeleton = skeleton_connected(skeleton_image, list_skeleton)
    list_match = ske_cor_match(list_skeleton, list_corner_new)

    list_match = ske_rearrangement(list_match)
    list_match = cor_addition(list_match)
    list_match = third_bezier_curve_point(list_match)
    image_information = control_field(list_match, k1_control_field_corner, k2_control_field_third_bezier)

    return image_information