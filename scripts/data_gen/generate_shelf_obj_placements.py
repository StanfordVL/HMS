from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys
import os
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import numpy as np
import argparse
import random

import pybullet as pb
import pybullet_data
import time

import pickle
from PIL import Image
import gibson2
from hms.sim import *

def generate_shelf_placements(objects_path,
                              container_file,
                              shelf_num,
                              num_generate,
                              num_objects,
                              count_start=0,
                              side=None,
                              rot_randomization=0,
                              obj_scale=1.0,
                              num_place_attempts=500,
                              show_gui=False):

    container_dir = os.path.dirname(container_file)
    placements_dir = os.path.join(container_dir,'placements')
    object_type = objects_path.split('/')[-1]
    if object_type == '':
        object_type = objects_path.split('/')[-2]
    gen_save_dir = os.path.join(placements_dir, 'shelf_%d'%shelf_num, object_type)
    if side is not None:
        gen_save_dir = os.path.join(gen_save_dir, side)
    if not os.path.exists(gen_save_dir):
        os.makedirs(gen_save_dir)

    container = ObjectContainer(args.container_file)
    env = ContainerObjectsEnv(show_gui=show_gui)
    print(f'container attributes: {dir(container)}')
    print(f'filename: {container.filename}')
    shelf_height = container.shelf_heights[shelf_num]
    num_placements_finished = 0
    print(f'num generate: {num_generate}')
    while num_placements_finished < num_generate:
        print('Generating placements %d'%num_placements_finished)
        gen_num = count_start + num_placements_finished
        container = ObjectContainer(args.container_file)
        env.reset(container)
        objects = os.listdir(objects_path)
        print(f'objects dir: {objects}')
        result_dict = {}
        error_flag = False
        target_obj_left = True

        start_time = time.time()

        # Drop Objects
        for i in range(num_objects):
            error_count = 0

            # if error break
            if error_flag:
                break

            item_to_add = np.random.choice(objects)
            print('Attempting to place %s'%item_to_add)
            obj_dir = os.path.join(objects_path, item_to_add, 'meshes')
            print(f'object dir path: {obj_dir}')
            obj_file = next(filter(lambda x: 'urdf' in x, os.listdir(obj_dir)), None)
            obj_fname = obj_dir + '/' + obj_file
            obj = ShelfObject(obj_fname, obj_scale)
            body_id = obj.load()
            found_placement = False
            while not found_placement:
                if error_count > 25:
                    pb.removeBody(bodyUniqueId=body_id)
                    print('Could not find placements!')
                    error_flag = True
                    error_count = 0
                    break
                for attempt in range(num_place_attempts):
                    orientation = obj.sample_orientation(rot_randomization)
                    obj.set_orientation(orientation)
                    aabb = pb.getAABB(body_id)
                    obj_width = (aabb[1][0] - aabb[0][0])
                    obj_length = (aabb[1][1] - aabb[0][1])
                    obj_height = (aabb[1][2] - aabb[0][2])

                    if side == 'right':
                        obj_x = np.random.uniform(obj_width/2.0, container.aabb[1][0] -obj_width/2.0)
                    elif side == 'left':
                        obj_x = np.random.uniform(container.aabb[0][0] + obj_width/2.0, -obj_width/2.0)
                    else:
                        obj_x = np.random.uniform(container.aabb[0][0]+obj_width/2,
                                                  container.aabb[1][0]-obj_width/2)
                    attempt_ratio = float(attempt+1)/num_place_attempts
                    y_start = container.aabb[0][1] + obj_length/2*1.1
                    y_end = container.aabb[1][1] - obj_length/2*1.1
                    y_len = container.aabb[1][1] - container.aabb[0][1] - obj_length
                    obj_y = np.random.uniform(y_start + (1-attempt_ratio)*y_len,
                                              y_end)
                    obj_z = np.random.uniform(shelf_height+obj_height/2,
                                              shelf_height+obj_height/2.0+0.1*attempt_ratio)
                    obj.set_position([obj_x, obj_y, obj_z])
                    pb.stepSimulation()
                    contact_points = pb.getContactPoints(body_id)
                    # if our new object overlaps with other objects, try again
                    # if this occurs too many times, move on
                    if len(contact_points) == 0:
                        found_placement = True
                        break
                if not found_placement:
                    error_count += 1
                else:
                    success_drop = env.gentle_drop(body_id)
                    for i in range(100):
                        pb.stepSimulation()

                    aabb = pb.getAABB(body_id)
                    pos = obj.get_position()
                    orint = obj.get_orientation()

                    if not success_drop or pos[2] < 0:
                        error_count += 1
                        found_placement = False
                    else:
                        pb.removeBody(bodyUniqueId=body_id)
                        obj = StaticObject(obj_fname, obj_scale)
                        obj_id = env.add_object(obj)
                        obj.set_position_orientation(pos,orint)
                        dimensions = [(aabb[1][i]-aabb[0][i]) for i in range(3)]

                        obj_dict = {'path':obj_fname,
                                    'description':item_to_add.lower().replace('_',' '),
                                    'aabb': aabb,
                                    'scale': obj_scale,
                                    'orientation':orint,
                                    'dimensions': dimensions,
                                    'location': pos}
                        result_dict[obj_id] = obj_dict
                        objects.remove(item_to_add)
        if not error_flag:
            mean_loc = np.array([0.0,0.0,0.0])
            for obj_id in result_dict:
                obj_dict = result_dict[obj_id]
                mean_loc+=obj_dict['location']
            mean_loc/=len(result_dict)

            rgb, depth, im3d, depth_im3d = env.get_observation()
            print(f"obj_ids: {result_dict.keys()}")
            for obj_id in result_dict:
                obj_dict = result_dict[obj_id]

                env.set_camera_point_at(mean_loc, dist=0.45)
                _, _, segmask, _, _ = env.get_observation(obj_id)
                location = np.array([0.0,0.0,0.0])
                if not np.any(segmask):
                    obj_dict['location_img'] = [0.0,0.0]
                    obj_dict['dimension'] = [0.0,0.0]
                    continue
                rows = np.any(segmask, axis=0)
                cols = np.any(segmask, axis=1)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                rmean = (rmin+rmax)/2
                cmean = (cmin+cmax)/2
                rlen = rmax-rmin
                clen = cmax-cmin
                obj_dict['location_img'] = [rmean,cmean]
                obj_dict['dimension'] = [rlen,clen]

                env.set_camera_point_at(obj_dict['location'])
                rgb, depth, segmask, im3d, depth_im3d = env.get_observation(obj_id, visualize=False, save=True)
                obj_image = env.get_obj_img(rgb, segmask, save=True)
                obj_dict[obj_id+'_image'] = obj_image
                # Add the rgb and depth images
                obj_dict['rgb'] = rgb
                obj_dict['depth'] = depth
                obj_dict[obj_id+'_segmask'] = segmask
                obj_dict['im3d'] = im3d
                obj_dict['depth_im3d'] = depth_im3d
                # Get camera intrinsics
                K = env.get_camera_intrinsics()
                obj_dict['K'] = K
                # Get the projection matrix
                P = env.get_projection_matrix()
                obj_dict['P'] = P
                V = env.get_V()
                obj_dict['V'] = V
                lightP = env.get_lightP()
                obj_dict['lightP'] = lightP
                lightV = env.get_lightV()
                obj_dict['lightV'] = lightV

                # Try different camera positions (hopefully)
                env.set_camera_point_at(obj_dict['location'], randomize=True)
                rgb, depth, segmask, im3d, depth_im3d = env.get_observation(obj_id, visualize=False, save=True)
                obj_image = env.get_obj_img(rgb, segmask, save=True)
                obj_dict['image_rand'] = obj_image
                # Add the rgb and depth images
                obj_dict['rgb_rand'] = rgb
                obj_dict['depth_rand'] = depth
                obj_dict[obj_id+'_segmask_rand'] = segmask
                obj_dict['im3d_rand'] = im3d
                obj_dict['depth_im3d_rand'] = depth_im3d
                # Get camera intrinsics
                K_rand = env.get_camera_intrinsics()
                obj_dict['K_rand'] = K_rand
                # get projection matrix
                P_rand = env.get_projection_matrix()
                obj_dict['P_rand'] = P_rand
                V_rand = env.get_V()
                obj_dict['V_rand'] = V_rand
                lightP_rand = env.get_lightP()
                obj_dict['lightP_rand'] = lightP_rand
                lightV_rand = env.get_lightV()
                obj_dict['lightV_rand'] = lightV_rand


            filename = os.path.join(gen_save_dir, 'shelf_setup_%d.pkl'%gen_num)
            print(f'save pickle at: {filename}')
            with open(filename,'wb') as f:
                pickle.dump(result_dict,f)
            num_placements_finished+=1

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Instantiate placements of objects on a shelf.')
    parser.add_argument('objects_folder', type=str, help='Path to folder with objects to place on shelf')
    parser.add_argument('container_file', type=str, help='Path to urdf file of container with shelves')
    parser.add_argument('--shelf_num', type=int, default=0, help='Shelf number to place on (0 being lowest)')
    parser.add_argument('--side', type=str, default=None, help='Whether to put objects on only the left or right side of the shelf.')
    parser.add_argument('--num_objects', type=int, default=5, help='How many objects to place on shelf.')
    parser.add_argument('--num_generate', type=int, default=5, help='How many different configurations of objects to create.')
    parser.add_argument('--count_start', type=int, default=0, help='First value to use for naming shelf placements.')
    parser.add_argument('--rot_randomization', type=float, default=0, help='How much to randomize object rotations.')
    parser.add_argument('--obj_scale', type=float, default=1.0, help='How much to scale objects.')
    parser.add_argument('--show_gui', action='store_true', help='Whether to show GUI.')
    args = parser.parse_args()
    generate_shelf_placements(args.objects_folder,
                              args.container_file,
                              args.shelf_num,
                              args.num_generate,
                              args.num_objects,
                              args.count_start,
                              args.side,
                              args.rot_randomization,
                              args.obj_scale,
                              show_gui=args.show_gui)
