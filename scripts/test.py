import os
import yaml
import random
import numpy as np
import argparse
import multiprocessing
from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from hms.models import *
from hms.scene_graph import sample_scene_graph, sample_containers, sample_objects, get_test_scene_graphs
from hms.utils import configure_logging


def test_scene_graph_obj_search(model_path,
                                container_selection='ours',
                                object_selection='ours'):
    random.seed(0)
    np.random.seed(0)
    obj_word_vecs = container_selection!='ours_no_labels'
    no_pass_model = container_selection=='ours_no_pass'
    obj_context_vecs = container_selection=='ours_context_vec' or object_selection=='ours_context_vec'
    no_images = container_selection=='ours_no_images'
    if 'ours' in container_selection:
        model = ObjectSearchModel(no_pass=no_pass_model,
                                  obj_context_vecs=obj_context_vecs,
                                  obj_word_vecs=obj_word_vecs,
                                  no_images=no_images)
        if model_path is None:
            model_path = 'data/models/%s.pt'%container_selection
        model.load_state_dict(model_path)
    if container_selection=='word_vec':
        model = WordVecModel()
    if container_selection=='most_likely':
        model = MostLikelyModel()

    if object_selection is None:
        object_selection = container_selection

    scene_graphs = get_test_scene_graphs(require_target_occluded=True)
    scene_graph_count = 0
    test_container_total_count,test_object_occlusion_total_count = 0, 0
    test_container_correct_count,test_occlusion_correct_count = 0, 0
    for scene_graph in scene_graphs:
        model.set_scene_graph(scene_graph)
        scene_graph_count+=1
        all_children = []
        for level in range (4):
            if level < 3:# If container
                if level == 0:
                    parents = sample_containers(scene_graph.rooms, num_sample=1)
                else:
                    parents = sample_containers(all_children, num_sample=1)
                    all_children = []
                if container_selection=='word_vec':
                    classifications = model.classify_containers(parents)
                for parent_i, parent in enumerate(parents):
                    test_container_total_count+=1
                    if 'ours' in container_selection or container_selection=='most_likely':
                        try:
                            classification = model.classify_container(parent,level)
                        except:
                            pass
                    elif container_selection=='random':
                        classification = random.random() > 0.5
                    else:
                        classification = classifications[parent_i]

                    # Calculate losses, check if correct
                    if level < 2:
                        parent_children = parent.children
                    else:
                        if 'ours' in object_selection and 'no_image' not in object_selection:
                            parent_children = [child for child in parent.children if child.image is not None]
                        else:
                            parent_children = parent.children
                    all_children+=parent_children

                    label = parent.has_target

                    is_correct = (classification > 0.5) == label
                    if is_correct:
                        test_container_correct_count+=1
            else:#At object level
                objects = sample_objects(all_children,
                                         num_sample=1,
                                         require_image=object_selection=='ours')

                if object_selection=='nearest':
                    nearness_vals = [-obj.location[1] for obj in objects]
                    max_val = max(nearness_vals)
                    min_val = min(nearness_vals)
                    classifications = [(n-min_val)/(max_val-min_val) for n in nearness_vals]
                elif object_selection=='largest':
                    size_vals = [obj.dimensions[0]*obj.dimensions[2] for obj in objects]
                    max_val = max(size_vals)
                    min_val = min(size_vals)
                    classifications = [(n-min_val)/(max_val-min_val) for n in size_vals]
                for obj_i, obj in enumerate(objects):
                    if 'ours' in object_selection or object_selection=='word_vec':
                        occlusion_classification = model.classify_object_occlusion(obj)
                    elif object_selection=='random':
                        occlusion_classification = random.random() > 0.5
                    else:
                        occlusion_classification = classifications[obj_i]

                    label = obj.occludes_target
                    is_correct = (occlusion_classification > 0.5) == label
                    if is_correct:
                        test_occlusion_correct_count+=1
                    test_object_occlusion_total_count+=1

        if test_object_occlusion_total_count > 0:
            container_accuracy = float(test_container_correct_count)/test_container_total_count*100
            occlusion_accuracy = float(test_occlusion_correct_count)/test_object_occlusion_total_count*100
            print('Scene graph count %d - accuracies: container=%.2f , occlusion=%.2f' %
                          (scene_graph_count, container_accuracy, occlusion_accuracy))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Instantiate placements of objects on a shelf.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model')
    parser.add_argument('--container_selection', type=str, default='ours')
    parser.add_argument('--object_selection', type=str, default=None)
    args = parser.parse_args()
    test_scene_graph_obj_search(args.model_path,
                                container_selection=args.container_selection,
                                object_selection=args.object_selection)
