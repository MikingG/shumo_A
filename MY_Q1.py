import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon
from mpl_toolkits.mplot3d import Axes3D
import shapely as spl
import scipy as sp

month_len = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
D_values = [(sum(month_len[2:]) + sum(month_len[:i])) % 365 for i in range(12)]
print(D_values)
ST_values = [9, 10.5, 12, 13.5, 15]
H = 3
Latitude = 39.4
SUN_HALF_SPREAD = 0.00465


def sun_position(ST, D, Latitude):
    '''input:
    ST: solar time
    D: day of the year
    Latitude: latitude of the place

    output:
    sin_as, cos_as: the sin and cos of the angle between the sun and the horizontal plane
    sin_gamma_s, cos_gamma_s: the sin and cos of the angle between the projection of the sun on the horizontal plane and the south
    gamma_s: the angle between the projection of the sun on the horizontal plane and the south
    p.s.
    as：alpha_s: the angle between the sun and the horizontal plane
    gamma_s: the angle between the projection of the sun on the horizontal plane and the south

    function:
    calculate the sun position
    '''
    phi = np.deg2rad(39.4)
    delta = np.arcsin(np.sin(2 * np.pi * D / 365) * np.sin(np.deg2rad(23.45)))
    omega = np.pi / 12 * (ST - 12)
    alpha_s_value = np.arcsin(np.cos(delta) * np.cos(phi) * np.cos(omega) + np.sin(delta) * np.sin(phi))
    gamma_s_numerator = np.sin(delta) - np.sin(alpha_s_value) * np.sin(phi)
    gamma_s_denominator = np.cos(alpha_s_value) * np.cos(phi)

    gamma_s_argument = np.clip(gamma_s_numerator / gamma_s_denominator, -1, 1)
    gamma_s_value = np.arccos(gamma_s_argument)
    return alpha_s_value, gamma_s_value
# D_values[0]

# D_values[0]


# based on the sun position(sin_as, cos_as, cos_gamma_s)
# and mirror position(x_m, y_m, z_m) and tower position(x_t, y_t, z_t)
# calculate the angle of mirror(A_H,E_H) and the plane of mirror

def mirror_angle(p_m,p_t,sun_pos):
    '''input:
    p_m: position of the mirror
    p_t: position of the tower
    sun_pos: the position of the sun
    output:
    sin_A_H, cos_A_H: the sin and cos of the angle between the mirror and the north
    sin_E_H, cos_E_H: the sin and cos of the angle between the mirror and horizontal plane
    T: the transformation matrix from the global coordinate to the local coordinate
    '''
    delta_x = p_t[0] - p_m[0]
    delta_y = p_t[1] - p_m[1]
    delta_z = p_t[2] - p_m[2]

    distance = np.sqrt(delta_x**2+delta_y**2+delta_z**2)
    n_at = 0.99321-0.0001176*distance+1.97*10**(-8)*distance**2
    alpha_s, gamma_s = sun_pos

    vector_m2t = np.array([delta_x, delta_y, delta_z])
    # normalization
    vector_m2t = vector_m2t / np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
    # vector_s2m = np.array([cos_as * np.cos(np.pi - gamma_s), cos_as * np.sin(np.pi - gamma_s), sin_as])
    vector_s2m = np.array([np.cos(alpha_s) * np.cos(np.pi/2 - gamma_s), np.cos(alpha_s) * np.sin(np.pi/2 - gamma_s), np.sin(alpha_s)])

    # normalization
    vector_s2m = vector_s2m / np.sqrt(vector_s2m[0] ** 2 + vector_s2m[1] ** 2 + vector_s2m[2] ** 2)

    n_vector = vector_m2t + vector_s2m
    n_vector = n_vector / np.sqrt(n_vector[0] ** 2 + n_vector[1] ** 2 + n_vector[2] ** 2)

    n_cos = np.dot(n_vector, vector_m2t)

    # calculate the angle of mirror(A_H,E_H)
    cos_A_H = n_vector[0] / np.sqrt(n_vector[0] ** 2 + n_vector[1] ** 2)
    sin_A_H = n_vector[1] / np.sqrt(n_vector[0] ** 2 + n_vector[1] ** 2)
    sin_E_H = n_vector[2] / np.sqrt(n_vector[0] ** 2 + n_vector[1] ** 2 + n_vector[2] ** 2)
    cos_E_H = np.sqrt(1 - sin_E_H ** 2)
    T = np.array([[-sin_E_H, -sin_A_H * cos_E_H, cos_A_H * cos_E_H],
                  [cos_E_H, -sin_A_H * sin_E_H, cos_A_H * sin_E_H],
                  [0, cos_A_H, sin_A_H]])

    return sin_A_H, cos_A_H, sin_E_H, cos_E_H, T, vector_s2m, vector_m2t, n_cos, n_vector, n_at
# the local coordinate is defined as the plane of mirror
# the x axis is the horizontal to ground along the mirror and to the right
# the y axis is the along the mirror and vertical to x axis towards the top
# the z axis is the normal vector of the mirror
# the origin point is in the center of the mirror
# the transformation matrix is from global coordinate to local coordinate
# the inverse transformation matrix is from local coordinate to global coordinate



def local2global(p, T):
    '''input:
    p: the position in local coordinate
    T: the transformation matrix from global coordinate to local coordinate
    output:
    the position in global coordinate
    fucntion:
    the T_inv is the inverse matrix of T
    and it equal to the transpose of T
    and it transform the position in local coordinate to global coordinate
    '''
    T_inv = T.T
    vector = p
    vector = np.dot(T_inv, vector)
    return np.array([vector[0], vector[1], vector[2]])


def global2local(p, T):
    '''input:
    p: the position in global coordinate
    T: the transformation matrix from global coordinate to local coordinate
    output:
    the position in local coordinate
    '''
    vector = p
    vector = np.dot(T, vector)
    return np.array([vector[0], vector[1], vector[2]])


def shadow(main_mirror, covering_mirror):

    '''input:
    main_mirror: the information of mirror1
    this is a list contain the position(1 arug), size(2 arugs:h, w), transformation matrix T, inlight (a vector inL)
    and outlight(a vector outL) of mirror1
    covering_mirror: the information of mirror2
    output:
    the shadow rate of mirror2 on mirror1'''
    #x,y,z is the position of the center of mirror
    #T is the transformation matrix from global coordinate to local coordinate
    #inL is a vector represent inlight of the mirror
    #outL is a vector represent outlight of the mirror
    position1, h1, w1, T1, inL1, outL1 = main_mirror
    position2, h2, w2, T2, inL2, outL2 = covering_mirror
    # the shadow is the inlight and outlight of mirror1 that hit the mirror2
    #the corner of mirror1

    p1 = position2+local2global(np.array([-w2/2,-h2/2,0]),T2)
    p2 = position2+local2global(np.array([w2/2,-h2/2,0]),T2)
    p3 = position2+local2global(np.array([w2/2,h2/2,0]),T2)
    p4 = position2+local2global(np.array([-w2/2,h2/2,0]),T2)

    #the corner of mirror1 in the local coordinate of mirror2
    p1_2 = global2local(p1-position1,T1)
    p2_2 = global2local(p2-position1,T1)
    p3_2 = global2local(p3-position1,T1)
    p4_2 = global2local(p4-position1,T1)

    inL1_2 = global2local(inL1,T1)
    outL1_2 = global2local(outL1,T1)

    #check the edge
    if (p1_2[2]<=0):
        p1_shadow = p1_2 + inL1_2 * (-p1_2[2] / inL1_2[2])
        p2_shadow = p2_2 + inL1_2 * (-p2_2[2] / inL1_2[2])
        p3_shadow = p3_2 + inL1_2 * (-p3_2[2] / inL1_2[2])
        p4_shadow = p4_2 + inL1_2 * (-p4_2[2] / inL1_2[2])

        p1_block = p1_2 + outL1_2 * (-p1_2[2] / outL1_2[2])
        p2_block = p2_2 + outL1_2 * (-p2_2[2] / outL1_2[2])
        p3_block = p3_2 + outL1_2 * (-p3_2[2] / outL1_2[2])
        p4_block = p4_2 + outL1_2 * (-p4_2[2] / outL1_2[2])
    else:
        p1_shadow = p2_shadow = p3_shadow = p4_shadow = np.array([0,0,0])
        p1_block = p2_block = p3_block = p4_block = np.array([0,0,0])

    plot_mirror = spl.Polygon([(w1/2,h1/2),(w1/2,-h1/2),(-w1/2,-h1/2),(-w1/2,h1/2)])
    plot_shadow = spl.Polygon([p1_shadow, p2_shadow, p3_shadow, p4_shadow])
    plot_block = spl.Polygon([p1_block, p2_block, p3_block, p4_block])
    plot_shadow = plot_shadow.intersection(plot_mirror)
    plot_block = plot_block.intersection(plot_mirror)

    # plt.plot(*plot_mirror.exterior.xy,color='r')
    # plt.plot(*plot_shadow.exterior.xy,color='g')
    # plt.plot(*plot_block.exterior.xy,color='b')
    #cacluate the cross point of 2 segment(p1_shadow,p3_shadow) and (p2_shadow,p4_shadow)
    #cacluate the cross point of 2 segment(p1_block,p3_block) and (p2_block,p4_block)

    cover_area = plot_shadow.area+plot_block.area-plot_shadow.intersection(plot_block).area
    plt.show()
    res = 1-cover_area/plot_mirror.area
    res = round(res, 6)
    return res


def tower_shadow(tower, mirror):
    '''input:
    tower: the position of the tower
    mirror: the position of the mirror
    output:
    the shadow of the tower on the mirror'''
    position2 = tower[0]
    h2 = tower[1]
    w2 = tower[2]

    position1, h1, w1, T1, inL1, outL1 = mirror

    #合理近似，太阳角度不变
    # 认为塔的阴影是一个矩形
    #计算塔的阴影在镜子上的投影

    #light from tower is the projection mirror_inlight on the horizontal plane
    light_from_tower = inL1-np.array([0,0,inL1[2]])
    light_from_tower = -light_from_tower/np.linalg.norm(light_from_tower)
    cos_A_H = light_from_tower[0] / np.sqrt(light_from_tower[0] ** 2 + light_from_tower[1] ** 2)
    sin_A_H = light_from_tower[1] / np.sqrt(light_from_tower[0] ** 2 + light_from_tower[1] ** 2)
    sin_E_H = 1
    cos_E_H = 0

    T2 = np.array([[-sin_E_H, -sin_A_H * cos_E_H, cos_A_H * cos_E_H],
                  [cos_E_H, -sin_A_H * sin_E_H, cos_A_H * sin_E_H],
                  [0, cos_A_H, sin_A_H]])

    position2 = position2 + np.array([0,0,h2/2])

    p1 = position2 + local2global(np.array([-w2 / 2, -h2/2, 0]), T2)
    p2 = position2 + local2global(np.array([w2 / 2, -h2/2, 0]), T2)
    p3 = position2 + local2global(np.array([w2 / 2, h2/2, 0]), T2)
    p4 = position2 + local2global(np.array([-w2 / 2, h2/2, 0]), T2)

    # the corner of mirror1 in the local coordinate of mirror2
    p1_2 = global2local(p1 - position1, T1)
    p2_2 = global2local(p2 - position1, T1)
    p3_2 = global2local(p3 - position1, T1)
    p4_2 = global2local(p4 - position1, T1)

    inL1_2 = global2local(inL1, T1)

    # check the edge
    if (p1_2[2] <= 0):
        p1_shadow = p1_2 + inL1_2 * (-p1_2[2] / inL1_2[2])
        p2_shadow = p2_2 + inL1_2 * (-p2_2[2] / inL1_2[2])
        p3_shadow = p3_2 + inL1_2 * (-p3_2[2] / inL1_2[2])
        p4_shadow = p4_2 + inL1_2 * (-p4_2[2] / inL1_2[2])
    else:
        p1_shadow = p2_shadow = p3_shadow = p4_shadow = np.array([0, 0, 0])

    plot_mirror = spl.Polygon([(w1 / 2, h1 / 2), (w1 / 2, -h1 / 2), (-w1 / 2, -h1 / 2), (-w1 / 2, h1 / 2)])
    plot_shadow = spl.Polygon([p1_shadow, p2_shadow, p3_shadow, p4_shadow])
    # plt.plot(*plot_shadow.exterior.xy, color='b')
    plot_cover = plot_shadow.intersection(plot_mirror)
    # plt.plot(*plot_mirror.exterior.xy, color='r')
    # plt.plot(*plot_cover.exterior.xy, color='g')
    # cacluate the cross point of 2 segment(p1_shadow,p3_shadow) and (p2_shadow,p4_shadow)
    # cacluate the cross point of 2 segment(p1_block,p3_block) and (p2_block,p4_block)

    cover_area = plot_cover.area
    plt.show()
    res = 1 - cover_area / plot_mirror.area
    res = round(res, 6)
    return res

def trunc(collector, mirror):
    '''input:
        tower: the position of the tower
        mirror: the position of the mirror
        output:
        the shadow of the tower on the mirror'''

    collector_height = collector[1]
    collector_width = collector[2]
    collector_position = collector[0] + np.array([0, 0, collector_height / 2])
    mirrorPosition, mirrorHeight, mirrorWeight, mirrorTrans, mirrorInLight, mirrorOutLight = mirror

    # light from tower is the projection mirror_inlight on the horizontal plane
    light2tower = mirrorOutLight - np.array([0, 0, mirrorOutLight[2]])
    light2tower = light2tower / np.linalg.norm(light2tower)

    # the coordinate of tower towards the light source
    cos_rotate = light2tower[0] / np.sqrt(light2tower[0] ** 2 + light2tower[1] ** 2)
    sin_rotate = light2tower[1] / np.sqrt(light2tower[0] ** 2 + light2tower[1] ** 2)
    # print(light2tower)
    # print(cos_rotate, sin_rotate)
    collectorTrans = np.array([[-sin_rotate, 0, cos_rotate],
                               [cos_rotate, 0, sin_rotate],
                               [0, 1, 0]])
    collectorTrans = collectorTrans.T


    a = np.array([0, 0, 1])
    # caculate the corner of mirror in the coordinate of mirror
    mirrorCorner = []
    distance = np.linalg.norm(mirrorPosition - collector_position)
    spread_len = distance * np.tan(SUN_HALF_SPREAD)

    change_width = spread_len + mirrorWeight / 2
    change_height = spread_len + mirrorHeight / 2
    change_list = [(change_width, change_height), (change_width, -change_height), (-change_width, -change_height), (-change_width, change_height)]
    change_vector = np.array(change_list)
    mirror_poly = spl.Polygon(change_vector)

    mirrorCorner.append(mirrorPosition + local2global(np.array(change_list[0]+tuple([0])), mirrorTrans))
    mirrorCorner.append(mirrorPosition + local2global(np.array(change_list[1]+tuple([0])), mirrorTrans))
    mirrorCorner.append(mirrorPosition + local2global(np.array(change_list[2]+tuple([0])), mirrorTrans))
    mirrorCorner.append(mirrorPosition + local2global(np.array(change_list[3]+tuple([0])), mirrorTrans))

    # caculate the corner of mirror in the coordinate of tower
    mirrorCorner_collector = []
    mirrorOutLight_collector = global2local(mirrorOutLight, collectorTrans)
    mirrorOutLight_mirror = global2local(mirrorOutLight, mirrorTrans)

    for i in range(len(mirrorCorner)):
        mirrorCorner_collector.append(global2local(mirrorCorner[i] - collector_position, collectorTrans))

    for i in range(len(mirrorCorner_collector)):
        mirrorCorner_collector[i] = mirrorCorner_collector[i] + mirrorOutLight_collector * (
                    -mirrorCorner_collector[i][2] / mirrorOutLight_collector[2])

    plot_collector = spl.Polygon(
        [(collector_width / 2, collector_height / 2), (collector_width / 2, -collector_height / 2),
         (-collector_width / 2, -collector_height / 2), (-collector_width / 2, collector_height / 2)])
    plot_mirror_project = spl.Polygon(mirrorCorner_collector)
    mirrorProject_outer = list([*plot_mirror_project.exterior.coords])

    plot_trunc = plot_collector.intersection(plot_mirror_project)


    trunc_outer = list(plot_trunc.exterior.coords)
    mirrorProject_outer_mirror = []
    trunc_outer_mirror = []
    for i in range(len(trunc_outer)):
        trunc_outer[i] = local2global(np.array(trunc_outer[i]), collectorTrans) + collector_position
        trunc_outer_mirror.append(global2local(trunc_outer[i] - mirrorPosition, mirrorTrans))
        trunc_outer_mirror[i] = trunc_outer_mirror[i] + mirrorOutLight_mirror * (
                    -trunc_outer_mirror[i][2] / mirrorOutLight_mirror[2])
    for i in range(len(mirrorProject_outer)):
        mirrorProject_outer[i] = local2global(np.array(mirrorProject_outer[i]), collectorTrans) + collector_position
        mirrorProject_outer_mirror.append(global2local(np.array(mirrorProject_outer[i]) - mirrorPosition, mirrorTrans))
        mirrorProject_outer_mirror[i] = mirrorProject_outer_mirror[i] + mirrorOutLight_mirror * (
                    -mirrorProject_outer_mirror[i][2] / mirrorOutLight_mirror[2])


    ploy_trunc_mirror = spl.Polygon(trunc_outer_mirror)
    ploy_mirrorProject_mirror = spl.Polygon(mirrorProject_outer_mirror)


    trunc_mirror_area = ploy_trunc_mirror.area
    mirrorProject_mirror_area = ploy_mirrorProject_mirror.area

    # fig = plt.figure(1, dpi=90)
    #
    # ax = fig.add_subplot(121)
    #3d plot draw the polygon
    # plot_polygon(mirror_poly,ax=ax, add_points=False,color='r')
    # plot_polygon(plot_collector)
    # plot_polygon(plot_mirror_project,ax=ax, add_points=False,color='g')
    # plot_polygon(ploy_mirrorProject_mirror,ax=ax, add_points=False,color='b')
    #
    # plt.show()

    res2 = trunc_mirror_area / mirrorProject_mirror_area

    res = res2

    return res

def D_N_I(sin_as):
    H = 3
    a = 0.4237 - 0.00821 * (6 - H) * (6 - H)
    b = 0.5055 + 0.00595 * (6.5 - H) * (6.5 - H)
    c = 0.2711 + 0.01858 * (2.5 - H) * (2.5 - H)
    G = 1.366
    DNI = G * (a + b * np.exp(-c / sin_as))
    return DNI

def f(mirrors, D, ST, p_t, Latitude):
     # ((x,y,z),w,h)
    n_at = {}#大气折射率的字典
    n_cos = {}#余弦效率的字典
    tow_shadow = {}#塔的阴影的字典
    n_shadow = {}#镜子之间的阴影的字典
    n_trunc = {}#截断效率的字典
    n = {}#总效率的字典
    for d in D:
        for st in ST:#对于每一天的每一个时刻
            n[(d, st)] = []
            sun_p = sun_position(d, st, Latitude)#太阳的位置
            n_cos[(d, st)] = []
            n_shadow[(d,st)] = []
            n_at[(d, st)] = []
            tow_shadow[(d, st)] = []
            n_trunc[(d, st)] = []
            for i in range(len(mirrors)):
                n_shadow[(d,st)].append([])
                # mirror_main = mirror_angle(mirrors[i][0], p_t, sun_p)
                # mirror1 = mirrors[i] + (mirror_main[4], mirror_main[5], mirror_main[6])

                mirror_main = mirror_angle(mirrors[i][0], p_t+np.array([0,0,80+4]), sun_p)
                mirror1 = mirrors[i] + (mirror_main[4], mirror_main[5], mirror_main[6])
                n_at[(d, st)].append(mirror_main[9])
                t = trunc([p_t+np.array([0, 0, 80]), 8, 7], mirror1)
                if t<0.1:
                    print("d:",d)
                    print("st:",st)
                    print("t:",t)
                    print("mirror1:",mirror1)
                n_trunc[(d, st)].append(t)
                n_cos[(d, st)].append(mirror_main[7])

                tow_shadow[(d, st)].append(tower_shadow([p_t, 8, 7], mirror1))
                for j in range(len(mirrors)):
                    if (i != j):
                        if (np.linalg.norm(mirrors[i][0] - mirrors[j][0]) < 35):
                            mirror_cover = mirror_angle(mirrors[j][0], p_t, sun_p)
                            mirror2 = mirrors[j] + (mirror_cover[4], mirror_cover[5], mirror_cover[6])
                            s = shadow(mirror1, mirror2)
                        else:
                            s = 1
                        # test:
                        # print(s)

                        n_shadow[(d,st)][-1].append(s)
                    if i == j:

                        n_shadow[(d,st)][-1].append(1)

            shadows = []  # 阴影效率

            for i in range(len(n_shadow[d, st])):
                product = 1
                for j in range(len(n_shadow[d, st])):
                    product *= n_shadow[d, st][i][j]
                product*=tow_shadow[d,st][i]#阴影效率
                shadows.append(product)
                product *= n_at[d,st][i] * n_cos[d,st][i] * 0.92  * n_trunc[d,st][i] #总效率
                n[d,st].append(product)#第i面镜子的光学效率
            n_shadow[(d, st)] = shadows

            # products 是 [每块定日镜的光学效率]
            # 输出每天每时每块镜子的光学效率，阴影效率，大气透射率，余弦效率,截断效率
    return n, n_shadow, n_at, n_cos, n_trunc  # 字典{(d,st):效率值}




def aver_all_n(mirrors, D_values, ST_values, tow_pos, Latitude):
    n, n_shadow, n_at, n_cos, n_trunc = f(mirrors, D_values, ST_values, tow_pos, Latitude)
    aver_n = 0
    aver_n_shadow = 0
    aver_n_cos = 0
    aver_n_trunc = 0
    # 每天的平均效率
    d_n = []
    d_n_shadow = []
    d_n_cos = []
    d_n_trunc = []

    for d in D_values:
        aver_n = 0
        aver_n_shadow = 0
        aver_n_trunc = 0
        aver_n_cos = 0
        for ST in ST_values:
            for i in range(len(n[(d, ST)])):
                aver_n += n[(d, ST)][i]  # 求每个时刻镜子的平均光学效率
                aver_n_trunc += n_trunc[(d, ST)][i]  # 求每个时刻镜子的平均截断效率
                aver_n_cos += n_cos[(d, ST)][i]  # 求每个时刻镜子的平均余弦效率
                aver_n_shadow += n_shadow[(d, ST)][i]  # 求每个时刻镜子的平均阴影效率


        d_n.append(aver_n)
        d_n_shadow.append(aver_n_shadow)
        d_n_cos.append(aver_n_cos)
        d_n_trunc.append(aver_n_trunc)

    # 每个特定day的平均光学效率
    # 镜子数
    count = len(mirrors)
    aver_n = 0
    aver_n_shadow = 0
    aver_n_trunc = 0
    aver_n_cos = 0
    for i in range(len(d_n)):
        d_n[i] /=  5* count  # 平均日光学效率
        d_n_shadow[i] /= 5 * count  # 平均日阴影效率
        d_n_cos[i] /= 5 * count  # 平均日余弦效率
        d_n_trunc[i] /= 5 * count  # 平均日截断效率效率

        aver_n_trunc += d_n_trunc[i]
        aver_n += d_n[i]
        aver_n_cos += d_n_cos[i]
        aver_n_shadow += d_n_shadow[i]
    aver_n /= 12  # 平均年光学效率
    aver_n_cos /= 12  # 平均年余弦效率
    aver_n_trunc /= 12  # 平均年截断效率
    aver_n_shadow /= 12  # 平均年阴影效率
    # 输出热功率
    DNI = {}
    area = 0
    d_sum_E = []
    aver_E = 0

    for i in range(len(mirrors)):
        area += mirrors[i][1] * mirrors[i][2]  # 求所有镜面的总面积

    for d in D_values:
        sum = 0
        for st in ST_values:
            a_s,gamma_s = sun_position(st, d, Latitude)
            DNI[(d, st)] = D_N_I(np.sin(a_s))
            p = 1
            for i in range(len(mirrors)):
                p += mirrors[i][1] * mirrors[i][2] * n[d,st][i]  # 面积×光学效率

            p *= DNI[(d, st)]

            sum += p
        d_sum_E.append(sum)
        # 每日定日场输出热功率
        d_sum_E[-1] /= area*5
        # 每日单位面积镜面平均输出热功率
        aver_E += d_sum_E[-1]#年平均单位面积镜面平均输出热功率

    unit_E = aver_E
    aver_E *= area


    print("************************************")
    print("平均光学效率：")
    print(d_n)
    print("************************************")
    print("平均余弦效率：")
    print(d_n_cos)
    print("************************************")
    print("平均阴影遮挡效率：")
    print(d_n_shadow)
    print("************************************")
    print("平均截断效率：")
    print(d_n_trunc)
    print("************************************")
    print("单位面积镜面平均输出热功率：")
    print(d_sum_E)
    print("************************************")
    print("年平均光学效率：")
    print(aver_n)
    print("************************************")
    print("年平均余弦效率：")
    print(aver_n_cos)
    print("************************************")
    print("年平均阴影遮挡效率：")
    print(aver_n_shadow)
    print("************************************")
    print("年平均截断效率：")
    print(aver_n_trunc)
    print("************************************")
    print("年平均输出热功率：")
    print(aver_E)
    print("************************************")
    print("年单位面积镜面平均输出热功率：")
    print(unit_E)
    return unit_E


data1 = pd.read_excel('附件.xlsx')
data1 = pd.DataFrame(data1)
data = []
data1.insert(loc=2, column='z', value=4)
data1.insert(loc=3, column='h', value=6)
data1.insert(loc=4, column='w', value=6)
data1 = data1.values
for i in data1:
    data.append((np.array([i[0], i[1], i[2]]), i[3], i[4]))

ST_values = [9, 10.5, 12, 13.5, 15]
H = 3
Latitude = 39.4

aver_all_n(data[:300], D_values, ST_values, np.array([0, 0, 0]), Latitude)