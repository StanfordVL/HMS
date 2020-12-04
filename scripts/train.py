import os
import random
import numpy as np
import argparse
from tqdm import tqdm
import multiprocessing
import logging
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from hms.models import *
from hms.scene_graph import sample_scene_graph, sample_containers, sample_objects
from hms.utils import configure_logging

def read_scene_graphs(queue, exit_queue, require_target_occlusion):
    while not exit_queue.full():
        if not queue.full():
            scene_graph = sample_scene_graph(require_target_occluded=require_target_occlusion)
            queue.put(scene_graph)

def read_obj_image(obj):
    obj.image = obj.read_image()
    return obj

def train_scene_graph_obj_search(batch_size, epoch_len, num_epochs, run_name=None,
                                 model_type='ours'):
    if run_name is None:
        run_name = model_type
    criterion = nn.BCELoss()
    no_pass_model = model_type=='ours_no_pass'
    obj_context_vecs = model_type=='ours_context_vec'
    obj_word_vecs = model_type!='ours_no_labels'
    no_images = model_type=='ours_no_images'
    model = ObjectSearchModel(no_pass=no_pass_model,
                              obj_context_vecs=obj_context_vecs,
                              obj_word_vecs=obj_word_vecs,
                              no_images=no_images)

    queue = multiprocessing.Queue(50)
    exit_queue = multiprocessing.Queue(1)
    processes = []
    num_read_processes = 4 if model_type!='context_vec' else 8
    for n in range(num_read_processes):
        p = multiprocessing.Process(target=read_scene_graphs, args=(queue, exit_queue, True))
        processes.append(p)
        p.start()

    if not obj_context_vecs and not no_images:
        read_image_pool = multiprocessing.Pool(6)

    optimizer = optim.Adam(model.parameters, lr=0.0001)
    model.set_train()
    configure_logging('logs/training/%s'%run_name)
    writer = SummaryWriter('logs/training/%s'%run_name)
    for epoch in tqdm(range(num_epochs+1), desc='Epoch'):
        train_container_total_count,train_object_occlusion_total_count = 0, 0
        train_container_correct_count,train_occlusion_correct_count = 0, 0
        running_container_loss, running_occlusion_loss = 0, 0
        for batch_num in tqdm(range(epoch_len), desc='Training'):
            scene_graphs = []
            for i in range(batch_size):
                scene_graphs.append(queue.get())
            all_children = [[] for i in range(len(scene_graphs))]
            container_losses = []
            occlusion_losses = []
            optimizer.zero_grad()
            for level in range (4):
                for batch_i, scene_graph in enumerate(scene_graphs):
                    model.set_scene_graph(scene_graph)
                    if level < 3:# If container
                        if level == 0:
                            parents = sample_containers(scene_graph.rooms, num_sample=1)
                        else:
                            parents = sample_containers(all_children[batch_i], num_sample=1)
                            all_children[batch_i] = []
                        for parent_i,parent in enumerate(parents):
                            train_container_total_count+=1
                            if level == 2 and not obj_context_vecs and not no_images:
                                parent.children = read_image_pool.map(read_obj_image, parent.children)
                            try:
                                classification = model.classify_container(parent, level)
                            except:
                                continue

                            # Calculate losses, check if correct
                            if level < 2 or obj_context_vecs or no_images:
                                parent_children = parent.children
                            else:
                                parent_children = [child for child in parent.children if child.image is not None]
                            all_children[batch_i]+=parent_children

                            label = parent.has_target
                            label_torch = Variable(torch.from_numpy(np.array([label])).float()).cuda()
                            loss = criterion(classification, label_torch)
                            container_losses.append(loss)

                            is_correct = (classification > 0.5) == label
                            if is_correct:
                                train_container_correct_count+=1
                    else:#At object level
                        objects = sample_objects(all_children[batch_i],
                                                 num_sample=1,
                                                 require_image=not obj_context_vecs and not no_images)
                        for obj_i,obj in enumerate(objects):
                            occlusion_classification = model.classify_object_occlusion(obj)

                            label = obj.occludes_target
                            label_torch = Variable(torch.from_numpy(np.array([label])).float()).cuda()
                            loss = criterion(occlusion_classification, label_torch)
                            occlusion_losses.append(loss)

                            is_correct = (occlusion_classification > 0.5) == label
                            if is_correct:
                                train_occlusion_correct_count+=1
                            train_object_occlusion_total_count+=1

            total_loss = 0
            if len(container_losses) > 0:
                container_loss = sum(container_losses)/len(container_losses)
                running_container_loss+=container_loss.item()
                total_loss+= container_loss
            if len(occlusion_losses) > 0:
                occlusion_loss = sum(occlusion_losses)/len(occlusion_losses)
                running_occlusion_loss+=occlusion_loss.item()
                total_loss+= occlusion_loss
            if epoch!=0:
                total_loss.backward()

                optimizer.step()

        running_occlusion_loss/=epoch_len
        running_container_loss/=epoch_len
        container_accuracy = int(float(train_container_correct_count)/train_container_total_count*100)
        occlusion_accuracy = int(float(train_occlusion_correct_count)/train_object_occlusion_total_count*100)
        logging.info('Epoch %d - losses: container=%.3f , occlusion=%.3f' %
                      (epoch, running_container_loss, running_occlusion_loss))
        logging.info('Epoch %d - accuracies: container=%d , occlusion=%d' %
                      (epoch, container_accuracy, occlusion_accuracy))
        writer.add_scalar('container_loss', running_container_loss, global_step=epoch)
        writer.add_scalar('occlusion_loss', running_occlusion_loss, global_step=epoch)
        writer.add_scalar('container_accuracy', container_accuracy, global_step=epoch)
        writer.add_scalar('occlusion_accuracy', occlusion_accuracy, global_step=epoch)
        writer.flush()

        save_path = os.path.join('data/models/%s.pt'%run_name)
        model.save_state_dict(save_path)
    writer.close()
    exit_queue.put('exit')
    for p in processes:
        p.join()
    pool.close()
    pool.join()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Instantiate placements of objects on a shelf.')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=10, help='How many scene graph instances to use at a time.')
    parser.add_argument('--epoch_len', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--model_type', type=str, default='ours')
    args = parser.parse_args()
    train_scene_graph_obj_search(args.batch_size,
                                 args.epoch_len,
                                 args.num_epochs,
                                 args.run_name,
                                 model_type=args.model_type)
