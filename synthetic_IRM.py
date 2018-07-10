import numpy as np
import numpy.random as npr

def type_name(id):
    return "t_" + str(id)

def obj_name(id):
    return "o_" + str(id)

def generate(num_objects, num_types, beta):
    types = [type_name(i) for i in range(num_types)]
    objects = np.array([obj_name(i) for i in range(num_objects)])
    # npr.shuffle(objects)

    split_objs = np.split(objects, num_types)

    true_z = {}
    item_type_mem = {}
    for idx, cluster in enumerate(split_objs):
        item_type_mem[types[idx]] = cluster
        for obj in cluster:
            true_z[obj] = types[idx]

    eta = {}
    for i in types:
        for j in types:
            eta[(i,j)] = npr.beta(beta, beta)

    data = []

    print(item_type_mem)

    for obj_i in objects:
        for obj_j in objects:
            type_i = true_z[obj_i]
            type_j = true_z[obj_j]
            b = int(eta[(type_i,type_j)] > npr.rand())
            data.append(('r', obj_i, obj_j, b))

    print(data)



generate(40, 2, 5)





