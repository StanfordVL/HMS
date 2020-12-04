import numpy as np
import random
import yaml
import pickle
import os
from hms import sim
import matplotlib
import matplotlib.pyplot as plt
import logging

# Load global word vecs dict upon import
WORD_VECS = {}
WORD_VEC_SIZE = 300
WORD_VEC_FILE = 'data/word_vecs.pkl'
if not os.path.isfile(WORD_VEC_FILE):
    with open("data/embeddings/glove.840B.300d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                vector = np.asarray(values[1:], "float32")
                WORD_VECS[word] = vector
            except:
                pass
    with open(WORD_VEC_FILE, 'wb') as to_f:
        pickle.dump(WORD_VECS, to_f)
else:
    with open(WORD_VEC_FILE, 'rb') as from_f:
        WORD_VECS = pickle.load(from_f)

# Load global test objects set upon import
TEST_OBJECTS = set()
with open('cfg/test_objects.txt', 'r') as f:
    for line in f:
        TEST_OBJECTS.add(line.lower().replace('_',' ').strip())

def get_word_vec(string):
    if string in WORD_VECS:
        word_vec = WORD_VECS[string]
    else:
        count = 1
        words = string.lower().replace('_',' ').split(' ')
        word_vec=np.zeros([WORD_VEC_SIZE])
        for word in words:
            if word in WORD_VECS:
                #if count==1:
                #    word_vec=WORD_VECS[word]
                #else:
                #    word_vec*=WORD_VECS[word]
                word_vec+=WORD_VECS[word]
                count+=1
        word_vec/=count
    return word_vec

class Container(object):
    """
    A class to represent containers in a 3D scene graph.
    """
    def __init__(self,
                 label,
                 volume,
                 parent,
                 has_target=False,
                 sim_obj = None):
        self.label = label
        self.volume = volume
        #self.parent = parent #this causes CUDA overflow somehow
        self.embedding_vec = None
        self.has_target = has_target
        self.children = []
        self.sim_obj = sim_obj

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def set_vec(self, vec):
        self.embedding_vec = vec

    def get_vec(self):
        return self.embedding_vec

    def set_second_vec(self, vec):
        self.second_embedding_vec = vec

    def get_second_vec(self):
        return self.second_embedding_vec

    def get_word_vec(self):
        if 'shelf' in self.label:
            return get_word_vec('shelf')
        return get_word_vec(self.label)

    def get_features_vec(self):
        volume = np.array([self.volume])
        word_vec = self.get_word_vec()
        feature_vec = np.concatenate([word_vec, volume])
        return feature_vec

    def set_sim_obj(self, sim_object):
        self.sim_object = sim_object

    def get_sim_obj(self):
        return self.sim_object

    def __str__(self):
        return self.label

class Object(object):
    """
    A class to represent objects in a 3D Scene Graph.
    """
    def __init__(self,
                 category,
                 description,
                 dimensions,
                 location,
                 parent,
                 location_img=None,
                 orientation=None,
                 image_path=None,
                 aabb=None,
                 occludes_target=False,
                 is_target=False,
                 sim_obj=None):
        self.category = category
        self.description = description
        self.dimensions = dimensions
        self.location = location
        self.location_img = location_img
        self.orientation = orientation
        # self.parent = parent this causes CUDA overflow somehow
        self.embedding_vec = None
        self.second_embedding_vec = None
        self.occludes_target = occludes_target
        self.is_target = is_target
        self.aabb = aabb
        self.image_path = image_path
        self.children = None
        self.image = None
        self.same_type_as_target = False
        self.is_occluded = False
        self.occludes_object = False
        self.sim_obj = sim_obj

    def read_image(self):
        if self.image_path == None or not os.path.isfile(self.image_path):
            return None
        with open(self.image_path, 'rb') as f:
            return np.load(f, allow_pickle=True)

    def set_vec(self, vec):
        self.embedding_vec = vec

    def get_vec(self):
        return self.embedding_vec

    def set_second_vec(self, vec):
        self.second_embedding_vec = vec

    def get_second_vec(self):
        return self.second_embedding_vec

    def get_word_vec(self):
        return get_word_vec(self.description)

    def get_features_vec(self,
                         include_word_vec=False,
                         target_word_vec=None):
        dimensions = self.normalized_dim
        location = self.normalized_loc
        if include_word_vec:
            word_vec = get_word_vec(self.description)
            feature_vec = np.concatenate([word_vec, dimensions, location])
        elif target_word_vec is not None:
            label = self.category.lower()
            #if label[-1]=='s':
            #    label = label[0:-1]
            #if label=='Veggie':
            #    label = 'Vegetable'
            word_vec = get_word_vec(label)
            word_vec_dot = np.dot(word_vec, target_word_vec)
            word_vec_sim = word_vec_dot / (np.linalg.norm(word_vec)*np.linalg.norm(target_word_vec))
            feature_vec = np.concatenate([dimensions, location, np.array([word_vec_sim])])
        else:
            feature_vec = np.concatenate([dimensions, location])
        return feature_vec

    def set_sim_obj(self, sim_object):
        self.sim_object = sim_object

    def get_sim_obj(self):
        return self.sim_object

    def __str__(self):
        return self.description

def nodes_to_str(parents, indent=0):
    nodes_str=''
    for parent in parents:
      nodes_str+='\t' * indent + str(parent)+'\n'
      if parent.children is not None:
          nodes_str+=nodes_to_str(parent.children, indent+1)
    return nodes_str

class SceneGraph(object):
    """
    Little class to bundle functionality for reasoning over scene graphs
    """
    def __init__(self, rooms, target_object, all_objs=None):
        self.rooms = rooms
        self.target_object = target_object
        self.all_objs = all_objs

    def __str__(self):
        return nodes_to_str(self.rooms)

def sample_containers(options, num_sample=1):
    """
    Sample containers having and not having the target.
    """
    sampled = []
    incorrect_options = []
    for option in options:
        if option.has_target and len(sampled)<num_sample:
            sampled.append(option)
        else:
            incorrect_options.append(option)
    for i in range(num_sample):
        sampled.append(random.choice(incorrect_options))
        incorrect_options.remove(sampled[-1])
    return sampled

def sample_objects(options, num_sample=1, require_image=True):
    """
    Sample objects occluding and not occluding the target.
    """
    sampled_occludes = []
    incorrect_options_occludes = []
    for option in options:
        if option.is_occluded:
            continue
        if require_image and str(option.image)=='None':
            continue

        if option.occludes_target:
            if len(sampled_occludes)<num_sample:
                sampled_occludes.append(option)
        else:
            incorrect_options_occludes.append(option)
    for i in range(len(sampled_occludes)):
        if len(incorrect_options_occludes) == 0:
            break
        sampled_occludes.append(random.choice(incorrect_options_occludes))
        incorrect_options_occludes.remove(sampled_occludes[-1])
    return sampled_occludes

def make_obj(object_label, parent, object_info=None, is_target=False, load_sim=False):
    """
    Utility function to creat an Object from loaded yaml file.
    """
    occludes_target = False
    obj = None

    description = object_info['description']
    dimensions = object_info['dimensions']
    location = object_info['location']
    if 'location_img' in object_info:
        location_img = object_info['location_img']
    else:
        location_img = np.array([0.0,0.0])
    orientation = object_info['orientation']
    aabb = object_info['aabb']
    image_path = None
    if 'image_path' in object_info:
        image_path = object_info['image_path']
    scale = 1.0
    if 'scale' in object_info:
        scale = object_info['scale']

    sim_obj = None
    if load_sim:
        sim_obj = sim.StaticObject(object_info['path'], scale)
    obj = Object(object_label,
                 description,
                 dimensions,
                 location,
                 parent,
                 location_img=location_img,
                 orientation=orientation,
                 image_path=image_path,
                 aabb=aabb,
                 occludes_target=occludes_target,
                 is_target=is_target,
                 sim_obj=sim_obj)

    return obj

def aabb_yz_overlap(a, b):
    """
    Overlap of b into a, as a fraction.
    """
    if b[1][1] >= a[0][1]:
        return 0

    dx = min(a[1][0], b[1][0]) - max(a[0][0], b[0][0])
    dy = min(a[1][2], b[1][2]) - max(a[0][2], b[0][2])
    a_size = (a[1][0]-a[0][0])*(a[1][2]-a[0][2])
    if (dx>=0) and (dy>=0):
        return dx*dy / a_size

    return 0

OCCLUSION_THRESH = 0.3 # global threshold for whether to count object as occluded
def eval_occlusions(objs, target_object=None):
    """
    Update info on which objects occlude which, based on ground truth
    bounding boxes.
    """
    for obj_1 in objs:
        for obj_2 in objs:
            if obj_1==obj_2:
                continue
            overlap = aabb_yz_overlap(obj_2.aabb, obj_1.aabb)
            if overlap > OCCLUSION_THRESH:
                obj_2.is_occluded = True
                obj_1.occludes_object = True
                if obj_2 == target_object:
                    obj_1.occludes_target = True

def create_scene_graph_with_target(scene_graph_dir,
                                   scene_graph_hierarchy,
                                   require_target_occluded = False,
                                   load_sim=False,
                                   exclude_test_objects=False):
    """
    Create a scene graph by loading stored info from files and sampling a valid
    target object.
    """
    target_object = None
    while target_object is None:# in case
        rooms = []
        containers = []
        room_with_target = random.choice(list(scene_graph_hierarchy.keys()))
        all_objs = []
        for room_label in scene_graph_hierarchy:
            room_has_target = room_label == room_with_target
            room = Container(room_label, 1.0, None, room_has_target)
            rooms.append(room)

            room_containers = scene_graph_hierarchy[room_label]
            container_with_target = random.choice(list(room_containers.keys()))
            for container_key in room_containers:
                container_has_target = room_has_target and container_key == container_with_target
                container_info = room_containers[container_key]
                container_label = container_info['label']
                container_path = container_info['path']
                shelves = container_info['shelves']
                if load_sim:
                    container_file = next(filter(lambda x: 'urdf' in x, os.listdir(container_path)), None)
                    container_urdf_path = os.path.join(container_path,container_file)
                    container_sim = sim.ObjectContainer(container_urdf_path)
                    volume = container_sim.size[0]*container_sim.size[1]*container_sim.size[2]
                else:
                    info_file = os.path.join(container_path, 'info.yaml')
                    with open(info_file,'r') as f:
                        container_info = yaml.load(f)
                    size = container_info['size']
                    volume = size[0]*size[1]*size[2]*(1.0+(random.random()-0.5)*0.2)
                    container_sim = None
                container = Container(container_label, volume, room, container_has_target, sim_obj=container_sim)
                containers.append(container)
                room.add_child(container)

                shelf_with_target = random.choice(list(shelves.keys()))

                for shelf_label in shelves:
                    shelf_has_target = container_has_target and shelf_label == shelf_with_target
                    shelf = Container(shelf_label, 1.0, container, shelf_has_target)
                    container.add_child(shelf)

                    shelf_path = shelves[shelf_label]['path']
                    object_types = shelves[shelf_label]['object_types']
                    target_object_type = random.choice(object_types)

                    objs = []
                    min_locs = [-100, -100, -100]
                    max_locs = [100, 100, 100]
                    min_dims = [-100, -100, -100]
                    max_dims = [100, 100, 100]
                    for object_type in object_types:
                        correct_type = shelf_has_target and object_type == target_object_type

                        target_object_label = None
                        placements_file_path = os.path.join(scene_graph_dir, container_key, shelf_label, '%s_placements.yaml'%object_type)
                        with open(placements_file_path,'r') as placements_file:
                            shelf_objects = yaml.load(placements_file)

                        if correct_type and not require_target_occluded:
                            target_object_label = random.choice(list(shelf_objects.keys()))
                        else:
                            target_object_label = None

                        for shelf_object in shelf_objects:
                            object_label = shelf_object
                            is_target = correct_type and object_label == target_object_label
                            object_info = None
                            object_info = shelf_objects[shelf_object]

                            if 'location_img' in object_info and object_info['location_img'][0]!=0:
                                loc = object_info['location_img']
                                min_locs = [min(min_locs[i],loc[i]) for i in range(2)]
                                max_locs = [max(max_locs[i],loc[i]) for i in range(2)]
                                dim = object_info['dimensions']
                                min_dim = [min(min_dims[i],dim[i]) for i in range(2)]
                                max_dim = [max(max_dims[i],dim[i]) for i in range(2)]

                            obj = make_obj(object_type, shelf, object_info, is_target=is_target, load_sim=load_sim)
                            objs.append(obj)

                            if is_target:
                                target_object = obj
                            if correct_type:
                                obj.same_type_as_target = True
                            shelf.add_child(obj)
                        all_objs+=objs

                        eval_occlusions(objs, target_object)
                        for obj in objs:
                            if require_target_occluded and correct_type and target_object is None and obj.is_occluded\
                               and (not exclude_test_objects or not obj.description in TEST_OBJECTS):
                                target_object = obj
                                obj.is_target = True
                                for obj in objs:
                                    obj.occludes_target = False
                                eval_occlusions(objs, target_object)
                    for obj in objs:
                        obj.normalized_loc = [(obj.location_img[i]-min_locs[i])/(max_locs[i]-min_locs[i]) for i in range(2)]
                        obj.normalized_dim = [(obj.dimensions[i]-min_dims[i])/(max_dims[i]-min_dims[i]) for i in range(2)]

        max_volume = max([container.volume for container in containers])
        for container in containers:
            container.volume/=max_volume
    return SceneGraph(rooms, target_object, all_objs=all_objs)

def sample_scene_graph(scene_graphs_path='data/scene_graphs/training',
                       require_target_occluded=False,
                       load_sim=False):
    """
    Sample a random scene graph.
    """
    scene_graph_dirs = os.listdir(scene_graphs_path)
    chosen_scene_graph = random.choice(scene_graph_dirs)
    scene_graph_dir = os.path.join(scene_graphs_path,chosen_scene_graph)
    scene_graph_file = os.path.join(scene_graph_dir,'scene_graph.yaml')
    with open(scene_graph_file , 'r') as f:
        scene_graph_hierarchy = yaml.load(f)
    return create_scene_graph_with_target(scene_graph_dir, scene_graph_hierarchy, require_target_occluded,
                                          load_sim=load_sim, exclude_test_objects=True)

def get_test_scene_graphs(scene_graphs_path='data/scene_graphs/testing',
                          require_target_occluded=False,
                          load_sim=False):
    """
    Iterate through all scene grahps in a given directory.
    """
    scene_graph_dirs = os.listdir(scene_graphs_path)
    for i,scene_graph_dir in enumerate(sorted(scene_graph_dirs)):
        random.seed(i)
        np.random.seed(i)
        scene_graph_file = os.path.join(scene_graphs_path, scene_graph_dir,'scene_graph.yaml')
        with open(scene_graph_file , 'r') as f:
            scene_graph_hierarchy = yaml.load(f)
        yield create_scene_graph_with_target(os.path.join(scene_graphs_path,scene_graph_dir),
                                             scene_graph_hierarchy,
                                             require_target_occluded,
                                             load_sim=load_sim,
                                             exclude_test_objects=False)
