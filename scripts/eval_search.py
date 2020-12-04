import os
import yaml
import random
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from hms.sim import ContainerObjectsEnv
from hms.models import ObjectSearchModel, WordVecModel, RandomModel, OracleModel, MostLikelyModel
from hms.scene_graph import  get_test_scene_graphs, eval_occlusions
from hms.utils import configure_logging

def test_scene_graph_obj_search(model_path,
                                run_name=None,
                                start_give_up_thresh=0.1,
                                multi_pass=True,
                                save_images=True,
                                container_selection='ours',
                                object_selection='ours'):
    if run_name is None:
        log_dir = 'logs/search_eval/%s_%s'%(container_selection,object_selection)
    else:
        log_dir = 'logs/search_eval/%s'%(run_name)
    configure_logging(log_dir)
    no_pass_container_selection_model = container_selection=='ours_no_pass' or object_selection=='ours_no_pass'
    obj_context_vecs = container_selection=='ours_context_vec' or object_selection=='ours_context_vec'
    obj_word_vecs = container_selection!='ours_no_labels' and object_selection!='ours_no_labels'
    no_images = container_selection=='ours_no_images' or object_selection=='ours_no_images'

    container_selection_model = None
    object_selection_model = None
    if 'ours' in container_selection:
        no_pass_container_selection_model = container_selection=='ours_no_pass'
        obj_context_vecs = container_selection=='ours_context_vec'
        obj_word_vecs = container_selection!='ours_no_labels'
        no_images = container_selection=='ours_no_images'

        model = ObjectSearchModel(no_pass=no_pass_container_selection_model,
                                  obj_context_vecs=obj_context_vecs,
                                  obj_word_vecs=obj_word_vecs,
                                  no_images=no_images)
        model_path = 'data/models/%s.pt'%container_selection
        model.load_state_dict(model_path)
        model.set_eval()
        container_selection_model = model
    elif container_selection=='word_vec':
        container_selection_model = WordVecModel()
    elif container_selection=='random':
        container_selection_model = RandomModel()
    elif container_selection=='oracle':
        container_selection_model = OracleModel()
    elif container_selection=='most_likely':
        container_selection_model = MostLikelyModel()

    if 'ours' in object_selection:
        if container_selection==object_selection:
            object_selection_model = container_selection_model
        else:
            no_pass_container_selection_model =  object_selection=='ours_no_pass'
            obj_context_vecs = object_selection=='ours_context_vec'
            obj_word_vecs = object_selection!='ours_no_labels'
            no_images = object_selection=='ours_no_images'

            model = ObjectSearchModel(no_pass=no_pass_container_selection_model,
                                      obj_context_vecs=obj_context_vecs,
                                      obj_word_vecs=obj_word_vecs,
                                      no_images=no_images)
            model_path = 'data/models/%s.pt'%object_selection
            model.load_state_dict(model_path)
            model.set_eval()
            object_selection_model = model

    give_up_thresh = [start_give_up_thresh for i in range(4)]
    scene_graphs = get_test_scene_graphs(load_sim=True, require_target_occluded=True)
    total_actions, prev_total_actions = 0, 0
    total_targets_found = 0
    total_scene_graphs = 0
    env = ContainerObjectsEnv()
    for scene_graph in scene_graphs:
        random.seed(total_scene_graphs)
        np.random.seed(total_scene_graphs)
        logging.info('Starting search %d '%total_scene_graphs)
        logging.info('Searching for object %s'%scene_graph.target_object.description)
        logging.info('Searching in scene graph:\n%s---'%str(scene_graph))
        container_selection_model.set_scene_graph(scene_graph)
        if object_selection_model is not None and object_selection_model!=container_selection_model:
            object_selection_model.set_scene_graph(scene_graph)

        total_scene_graphs+=1
        found_target = False
        pass_num = 0
        seen_objects = set()
        removed_objects = set()

        shelf_count=0
        removal_count = 0

        last_give_up_room_val = None
        while not found_target:
            if last_give_up_room_val is not None:
                give_up_thresh[0] = last_give_up_room_val
            pass_num+=1
            rooms_dict = {}
            for room in scene_graph.rooms:
                room_classification = container_selection_model.classify_container(room, 0)
                rooms_dict[room] = float(room_classification)
                if object_selection_model is not None and object_selection_model!=container_selection_model:
                    object_selection_model.classify_container(room, 0, second_vec=True)

            logging.info('Room values:')
            for room in rooms_dict:
                logging.info('%s (%s) = %f'%(room.label, str(room.has_target), rooms_dict[room]))

            for room, room_score in sorted(rooms_dict.items(), key=lambda x: x[1], reverse=True):
                if found_target:
                    break
                if room_score < give_up_thresh[0]:
                    logging.info('%s has low score, breaking'%room.label)
                    last_give_up_room_val = room_score
                    break
                logging.info('\nExploring room %s\n'%room.label)
                total_actions+=1

                container_dict = {}
                for container in room.children:
                    container_classification = container_selection_model.classify_container(container, 1)
                    container_dict[container] = float(container_classification)
                    if object_selection_model is not None and object_selection_model!=container_selection_model:
                        object_selection_model.classify_container(container, 1, second_vec=True)

                logging.info('Container values:')
                for container in container_dict:
                    logging.info('%s (%s) = %f'%(container.label, str(container.has_target), container_dict[container]))

                for container, container_score in sorted(container_dict.items(), key=lambda x: x[1], reverse=True):
                    if found_target:
                        break
                    if container_score < give_up_thresh[1]:
                        logging.info('%s has low score, breaking'%room.label)
                        give_up_thresh[1] = container_score
                        break
                    logging.info('\nExploring container %s\n'%container.label)
                    total_actions+=1
                    env.reset(container.sim_obj)

                    shelf_dict = {}
                    for shelf in container.children:
                        shelf_classification = container_selection_model.classify_container(shelf, 2)
                        shelf_dict[shelf] = float(shelf_classification)
                        if object_selection_model is not None and object_selection_model!=container_selection_model:
                            object_selection_model.classify_container(shelf, 2, second_vec=True)

                    logging.info('Shelf values:')
                    for container in shelf_dict:
                        logging.info('%s (%s) = %f'%(container.label, str(container.has_target), shelf_dict[container]))

                    for shelf, shelf_score in sorted(shelf_dict.items(), key=lambda x: x[1], reverse=True):
                        removal_count = 0
                        if found_target:
                            break
                        if shelf_score < give_up_thresh[2]:
                            logging.info('Shelf have low score, breaking\n')
                            give_up_thresh[2] = shelf_score
                            break
                        logging.info('\nExploring shelf %s\n'%shelf.label)
                        total_actions+=1
                        shelf_count+=1

                        obj_dict = {}
                        for obj in shelf.children:
                            if obj in removed_objects:
                                continue
                            obj_dict[obj] = -1.0
                            if 'ours' in object_selection:
                                obj.image = obj.read_image()
                            env.add_object(obj.sim_obj)
                            obj.sim_obj.set_position_orientation(obj.location, obj.orientation)

                        start_num_objs = len(obj_dict.keys())
                        mean_loc_set = False
                        mean_loc = np.array([0.0,0.0,0.0])
                        for obj in obj_dict:
                            if not mean_loc_set:
                                mean_loc+=np.array(obj.location)
                        if not mean_loc_set:
                            mean_loc/=len(obj_dict.keys())
                            mean_loc[2]+=0.05
                            mean_loc_set = True
                        while len(obj_dict.keys()) >= 1:
                            if save_images:
                                env.set_camera_point_at(mean_loc, dist=0.45)
                                img_path = os.path.join(log_dir,'traversal_images')
                                if not os.path.exists(img_path):
                                    os.makedirs(img_path)

                                rgb, depth = env.get_observation()
                                '''
                                min_val = min([float(v) for v in obj_dict.values()])
                                max_val = max([float(v) for v in obj_dict.values()])
                                if max_val > min_val:
                                    for obj in obj_dict:
                                        _, _, segmask = env.get_observation(obj.sim_obj.body_id)
                                        if not np.any(segmask):
                                            continue
                                        rows = np.any(segmask, axis=0)
                                        cols = np.any(segmask, axis=1)
                                        rmin, rmax = np.where(rows)[0][[0, -1]]
                                        cmin, cmax = np.where(cols)[0][[0, -1]]
                                        rmean = (rmin+rmax)/2
                                        cmean = (cmin+cmax)/2
                                        rlen = rmax - rmin
                                        clen = cmax - cmin
                                        maxlen = max([rlen, clen])
                                        rmin = int((rmean - maxlen/2))
                                        rmax = int((rmean + maxlen/2))
                                        cmin = int((cmean - maxlen/2))
                                        cmax = int((cmean + maxlen/2))
                                        #rgb[cmin:cmax, rmin:rmin+5, 1]+= float(obj_dict[obj])
                                        #rgb[cmin:cmax, rmax-5:rmax, 1]+= float(obj_dict[obj])
                                        #rgb[cmin:cmin+5, rmin:rmax, 1]+= float(obj_dict[obj])
                                        #rgb[cmax-5:cmax, rmin:rmax, 1]+= float(obj_dict[obj])
                                '''
                                plt.imshow(rgb)
                                plt.title(scene_graph.target_object.description)
                                plt.axis('off')

                                plt.savefig(img_path +"/%d_%d_%d.png"%(total_scene_graphs, shelf_count, removal_count),
                                            bbox_inches='tight')
                                plt.close()

                            for obj in obj_dict:
                                obj.is_occluded = False
                            eval_occlusions(obj_dict.keys(), scene_graph.target_object)
                            target_occluded = scene_graph.target_object.is_occluded
                            if not scene_graph.target_object.is_occluded:
                                logging.info('\nObject found!!!')
                                found_target = True
                                total_targets_found+=1
                                break
                            if object_selection_model is not None:
                                object_selection_model.classify_container(shelf, 2)

                            max_obj_score = 0.0
                            min_val = 100
                            max_val = -100
                            for obj in obj_dict:
                                if obj.is_occluded:
                                    obj_dict[obj] = -1.0
                                    continue
                                if object_selection=='random':
                                    obj_dict[obj] = np.random.random()
                                elif object_selection=='oracle':
                                    obj_dict[obj] = int(obj.occludes_target)
                                elif object_selection=='nearest':
                                    obj_dict[obj] = -obj.location[1]
                                    min_val = min(obj_dict[obj], min_val)
                                    max_val = max(obj_dict[obj], max_val)
                                elif object_selection=='largest':
                                    obj_dict[obj] = obj.dimensions[2]*obj.dimensions[2]
                                    min_val = min(obj_dict[obj], min_val)
                                    max_val = max(obj_dict[obj], max_val)
                                else:
                                    if object_selection_model!=container_selection_model:
                                        if obj.get_second_vec() is None:
                                            obj_dict[obj] = 0.0
                                            continue
                                        occlusion_classification = object_selection_model.classify_object_occlusion(obj, second_vec=True)
                                    else:
                                        if obj.get_vec() is None:
                                            obj_dict[obj] = 0.0
                                            continue
                                        occlusion_classification = object_selection_model.classify_object_occlusion(obj)
                                    obj_dict[obj] = occlusion_classification

                            if object_selection=='nearest' or object_selection=='largest':
                                for obj in obj_dict:
                                    if obj.is_occluded and not obj.occludes_target:
                                        continue
                                    if max_val!=min_val:
                                        obj_dict[obj] = (obj_dict[obj]-min_val)/(max_val-min_val)
                                    else:
                                        obj_dict[obj] = 1.0

                            logging.info('Object values:')
                            for obj in obj_dict:
                                logging.info('%s (%s) = %f'%(obj.description, str(obj.occludes_target), obj_dict[obj]))

                            max_score = max(obj_dict.values())
                            if max_score < give_up_thresh[3]\
                               and (object_selection=='word_vec' or object_selection=='ours'):
                                logging.info('Objects have low score, breaking\n')
                                if max_score > -1:
                                    give_up_thresh[3] = max_score
                                break

                            max_obj = max(obj_dict, key=obj_dict.get)
                            logging.info('\nRemoving object %s\n'%max_obj.description)
                            max_obj.sim_obj.set_position([0,0,-1])

                            removal_count+=1
                            total_actions+=1
                            del obj_dict[max_obj]
                            removed_objects.add(max_obj)

                            if len(obj_dict.keys())==1 and scene_graph.target_object in obj_dict:
                                logging.info('\nObject found!!!')
                                found_target = True
                                total_targets_found+=1
                                break

                            if object_selection=='ours':
                                for obj_i, obj in enumerate(obj_dict):
                                    env.set_camera_point_at(obj.location)
                                    rgb, depth, segmask = env.get_observation(obj.sim_obj.body_id)
                                    obj_image = env.get_obj_img(rgb, segmask, save=False)
                                    obj.image = obj_image

                                    shelf_classification = container_selection_model.classify_container(shelf, 2)

        actions = total_actions - prev_total_actions
        prev_total_actions = total_actions
        logging.info('At scene graph %d pass %d: actions=%d, total_actions=%d'%(total_scene_graphs, pass_num, actions, total_actions))
        logging.info('-----------------------------------------------------------------\n')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Instantiate placements of objects on a shelf.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to container_selection_model')
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--give_up_thresh', type=float, default=0.1)
    parser.add_argument('--container_selection', type=str, default='ours')
    parser.add_argument('--object_selection', type=str, default='ours')
    args = parser.parse_args()
    test_scene_graph_obj_search(args.model_path,
                                run_name=args.run_name,
                                save_images=args.save_images,
                                start_give_up_thresh=args.give_up_thresh,
                                container_selection=args.container_selection,
                                object_selection=args.object_selection)
