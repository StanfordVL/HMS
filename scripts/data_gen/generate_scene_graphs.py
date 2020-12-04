import os
import copy
import yaml
import random
import pprint
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Generate randomized scene graphs.')
parser.add_argument('num_generate', type=int)
parser.add_argument('--num_start', type=int, default=0)
parser.add_argument('--save', action='store_true')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
pp = pprint.PrettyPrinter(indent=4, compact=True)

if args.test:
    with open('cfg/containers.yaml', 'r') as to_read:
        container_options = yaml.load(to_read)
else:
    with open('cfg/containers_test.yaml', 'r') as to_read:
        container_options = yaml.load(to_read)

if args.test:
    path_save_in = 'data/scene_graphs/testing'
else:
    path_save_in = 'data/scene_graphs/training'

for generate_num in range(args.num_start, args.num_start + args.num_generate):
    scene_graph = {}
    dest_dir = os.path.join(path_save_in, 'scene_graph_%d'%generate_num)
    if args.save and not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    used_object_types = set()
    for container in sorted(container_options.keys()):
        container_info = container_options[container]
        if 'exclude_prob' in container_info:
            if random.random() < container_info['exclude_prob']:
                continue
        container_path = container_info['path']
        container_room_probs = container_info['rooms']
        container_room_options = list(container_room_probs.keys())
        container_room_probs = [container_room_probs[option] for option in container_room_options]
        room_choice = np.random.choice(container_room_options, p=container_room_probs)
        container_info_copy = copy.deepcopy(container_info)
        del container_info_copy['rooms']
        if 'exclude_prob' in container_info_copy:
            del container_info_copy['exclude_prob']
        shelves_info = {}
        for shelf in container_info_copy['shelves']:
            shelf_info = container_info_copy['shelves'][shelf]
            shelf_path = shelf_info['path']
            object_type_list = []
            for object_type in shelf_info['object_types']:
                if object_type in used_object_types:
                    continue
                used_object_types.add(object_type)
                object_type_path = shelf_info['object_types'][object_type]
                path_to_placements = os.path.join(container_path, 'placements', shelf_path, object_type_path)
                placement_files = os.listdir(path_to_placements)
                print(path_to_placements)
                chosen_placement_file = sorted(placement_files)[0]
                path_to_placement_file = os.path.join(path_to_placements, chosen_placement_file)
                new_placements_file_dir = os.path.join(dest_dir, container, shelf)
                pickle_file = '%s_placements.pkl'%object_type
                new_path_to_placement_file = os.path.join(new_placements_file_dir, pickle_file)
                if args.save:
                    if not os.path.isdir(new_placements_file_dir):
                        os.makedirs(new_placements_file_dir)
                    os.replace(path_to_placement_file, new_path_to_placement_file)
                    path_to_pickle_file = new_path_to_placement_file
                else:
                    path_to_pickle_file = path_to_placement_file

                with open(path_to_pickle_file, 'rb') as f:
                    pickle_data = pickle.load(f)
                yaml_info = {}
                images = {}
                for obj_key in pickle_data:
                    yaml_info[obj_key] = {}
                    for key in pickle_data[obj_key]:
                        if key!='image':
                            yaml_info[obj_key][key] = pickle_data[obj_key][key]
                        elif str(pickle_data[obj_key][key])!='None':
                            images[obj_key] = pickle_data[obj_key][key]
                for obj_key in images:
                    placement_file = pickle_file.replace('.pkl','')
                    path_to_image = os.path.join(new_placements_file_dir, '%s_%s_image.npy'%(placement_file,str(obj_key)))
                    if args.save:
                        with open(path_to_image, 'wb') as f:
                            np.save(f,images[obj_key])
                    yaml_info[obj_key]['image_path'] = path_to_image
                yaml_path = new_path_to_placement_file.replace('pkl', 'yaml')
                if args.save:
                    with open(yaml_path,'w') as f:
                        yaml.dump(yaml_info, f)
                object_type_list.append(object_type)
            if len(object_type_list) > 0:
                shelf_info['object_types'] = object_type_list
                shelves_info[shelf] = shelf_info
        if len(shelves_info.keys()) > 0:
            if room_choice not in scene_graph:
                scene_graph[str(room_choice)] = {}
            container_info_copy['shelves'] = shelves_info
            scene_graph[str(room_choice)][container] = container_info_copy

    if args.save:
        dest_file = os.path.join(dest_dir, 'scene_graph.yaml')
        with open(dest_file,'w') as file_to_write:
            yaml.dump(scene_graph, file_to_write)
    pp.pprint(scene_graph)
    for i in range(3):
        print('-----------------------------------------------------------')

