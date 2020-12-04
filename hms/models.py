import yaml
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import torchvision.transforms as transforms
import copy
import math
import numpy as np
from PIL import Image

class NodeEmbedNet(nn.Module):
    """
    A set of FC layers to take features of a node and create an embedding
    """
    def __init__(self, node_features_size, embed_size, num_layers=2):
        super(NodeEmbedNet, self).__init__()
        self.embed_size = embed_size

        self.input_layer = nn.Linear(node_features_size, embed_size)

        self.layers = [self.input_layer]
        for n in range(num_layers-1):
            self.layer = nn.Linear(embed_size, embed_size)
            self.layers.append(self.layer)

    def forward(self, node_features):
        if type(node_features)==list:
            embedding = torch.stack(node_features)
            for layer in self.layers:
                embedding = F.relu(layer(embedding))
            return torch.unbind(embedding)
        embedding = node_features
        for layer in self.layers:
            embedding = F.relu(layer(embedding))
        return embedding

class ObjectEmbedNet(nn.Module):
    """
    A set of FC layers to take features of a node and create an embedding
    """
    def __init__(self, node_features_size, embed_size, num_layers=2):
        super(ObjectEmbedNet, self).__init__()
        self.embed_size = embed_size
        if node_features_size > 100:# if embedding images
            self.img_model = models.resnet18(pretrained=True).cuda()

            self.img_model.fc = nn.Sequential(nn.Dropout(0.1),
                                              nn.Linear(in_features=512, out_features=256, bias=True),
                                              nn.ReLU(),
                                              nn.Dropout(0.1),
                                              nn.Linear(in_features=256, out_features=256, bias=True))

        self.input_layer = nn.Linear(node_features_size, embed_size)

        self.layers = [self.input_layer]
        for n in range(num_layers-1):
            self.layer = nn.Linear(embed_size, embed_size)
            self.layers.append(self.layer)

    def compute_image_features(self, img_features):
        return torch.unbind(self.img_model(torch.stack(img_features)))#.detach())

    def forward(self, children_features):
        embedding = torch.stack(children_features)
        for layer in self.layers:
            embedding = F.relu(layer(embedding))
        return torch.unbind(embedding)

class EdgeNet(nn.Module):
    """
    A set of FC layers to take vectors of two nodes and create an edge embedding
    """
    def __init__(self, parent_size,
                       child_size,
                       output_size,
                       edge_feature_combine_method='mul',
                       num_layers=2):
        super(EdgeNet, self).__init__()
        self.output_size = output_size

        self.edge_feature_combine_method = edge_feature_combine_method
        if edge_feature_combine_method == 'concat':
            self.edge_input_size = parent_size + child_size
        elif edge_feature_combine_method == 'mul':
            self.edge_input_size = parent_size
            assert parent_size == child_size
        else:
            raise ValueError('%s is not a valid way to combine node features'%str(edge_feature_combine_method))

        self.input_layer = nn.Linear(self.edge_input_size, output_size)
        self.layers = [self.input_layer]
        for n in range(num_layers-1):
            self.layer = nn.Linear(output_size, output_size)
            self.layers.append(self.layer)

    def forward(self, parent_vec, child_vecs):
        if self.edge_feature_combine_method == 'concat':
            out = torch.stack([torch.cat((parent_vec, child_vec)) for child_vec in child_vecs])
        elif self.edge_feature_combine_method == 'mul':
            out = torch.stack([parent_vec*child_vec for child_vec in child_vecs])
        else:
            raise ValueError('%s is not a valid way to combine node features'%str(self.edge_feature_combine_method))
        for layer in self.layers:
            out = F.relu(layer(out))
        return torch.unbind(out)

class ClassifyNet(nn.Module):
    """
    A set of FC layers to take features of a node a classify some output
    """
    def __init__(self, input_size,
                       hidden_layer_size,
                       target_embed_net=None,
                       features_merge_method='mul',
                       num_layers=2):
        super(ClassifyNet, self).__init__()

        self.features_merge_method = features_merge_method
        if self.features_merge_method=='concat':
            input_size*=2

        self.input_layer = nn.Linear(input_size, hidden_layer_size)
        self.layers = [self.input_layer]
        for n in range(num_layers-2):
            self.linear_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
            self.layers.append(self.linear_layer)
        self.output_layer = nn.Linear(hidden_layer_size, 1)
        self.layers.append(self.output_layer)

        self.target_embed_net = target_embed_net

    def forward(self, node_vec, target_features):
        self.features_merge_method = 'mul'
        if self.target_embed_net is not None:
            target_vec = self.target_embed_net(target_features)
            if self.features_merge_method=='concat':
                out = torch.cat((node_vec, target_vec))
            else:
                out = node_vec * target_vec
        else:
            out = node_vec
        for layer in self.layers:
            if layer!=self.output_layer:
                out = F.relu(layer(out))
            else:
                out = layer(out)
                out = nn.Sigmoid()(out)
        return out

class CombinedNet(nn.Module):

    """
    A neural net architecure that assumes one parent node with multiple child nodes,
    with starting feature vectors, that aggregates info about children into the parent,
    and then outputs prediction for the parent as well as updated representations for
    child nodes.
    """
    def __init__(self, container_embed_net,
                       object_embed_net,
                       target_embed_net,
                       edge_embed_net,
                       classification_net,
                       no_pass=False):
        super(CombinedNet, self).__init__()
        self.container_embed_net = container_embed_net
        self.object_embed_net = object_embed_net
        self.target_embed_net = target_embed_net
        self.edge_embed_net = edge_embed_net
        self.classification_net = classification_net
        self.no_pass = no_pass

    def forward_top_level(self, parent_features, children_features, target_features):
        parent_vec = self.container_embed_net(parent_features)
        self.child_embed_net = self.container_embed_net
        return self.forward(parent_vec, children_features, target_features)

    def forward_child_containers(self, parent_vec, children_features, target_features):
        self.child_embed_net = self.container_embed_net
        return self.forward(parent_vec, children_features, target_features)

    def forward_child_objects(self, parent_vec, children_features, target_features):
        self.child_embed_net = self.object_embed_net
        return self.forward(parent_vec, children_features, target_features)

    def forward(self, parent_vec, children_features, target_features):
        children_vecs = self.child_embed_net(children_features)
        if self.no_pass:
            classification = self.classification_net(parent_vec, target_features)
            return children_vecs, classification
        new_children_vecs = self.edge_embed_net(parent_vec, children_vecs)
        new_parent_vec = torch.sum(torch.stack(new_children_vecs),0)/len(new_children_vecs)*0.5 + parent_vec*0.5
        classification = self.classification_net(new_parent_vec, target_features)

        return new_children_vecs, classification

class ObjectSearchModel(object):

    """
    The combined Hierarchical Mechanical Search model, with options for ablations and utility
    functions for saving and loading.
    """

    def __init__(self, no_pass=False, obj_context_vecs=False, obj_word_vecs=False, no_images=False):
        super(ObjectSearchModel, self).__init__()
        self.container_embed_net = NodeEmbedNet(301, 100, num_layers=2).cuda()
        self.obj_context_vecs = obj_context_vecs
        if obj_context_vecs:
            self.object_embed_net = ObjectEmbedNet(5, 100, num_layers=2).cuda()
        else:
            if obj_word_vecs and not no_images:
                self.object_embed_net = ObjectEmbedNet(560, 100, num_layers=2).cuda()
            elif no_images:
                self.object_embed_net = ObjectEmbedNet(304, 100, num_layers=2).cuda()
            else:
                self.object_embed_net = ObjectEmbedNet(260, 100, num_layers=2).cuda()
        self.obj_word_vecs = obj_word_vecs
        self.no_images = no_images
        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.552716,0.498705,0.447810],
                                         std=[0.225595,0.217791,0.229298])
        self.to_tensor = transforms.ToTensor()
        self.target_embed_net = NodeEmbedNet(300, 100, num_layers=2).cuda()
        self.edge_net = EdgeNet(100, 100, 100, edge_feature_combine_method='concat').cuda()
        self.container_classify_net = ClassifyNet(100, 100, self.target_embed_net, num_layers=3).cuda()
        self.object_occlusion_classify_net = ClassifyNet(100, 100, self.target_embed_net, num_layers=3).cuda()

        self.combined_net = CombinedNet(self.container_embed_net,
                                        self.object_embed_net,
                                        self.target_embed_net,
                                        self.edge_net,
                                        self.container_classify_net,
                                        no_pass=no_pass).cuda()
        self.parameters = list(self.combined_net.parameters())
        self.parameters+=list(self.object_occlusion_classify_net.parameters())
        self.parameters+=list(self.container_classify_net.parameters())
        self.parameters+=list(self.edge_net.parameters())
        self.parameters+=list(self.target_embed_net.parameters())
        self.parameters+=list(self.object_embed_net.parameters())
        self.parameters+=list(self.container_embed_net.parameters())

    def set_scene_graph(self, scene_graph):
        self.scene_graph = scene_graph
        self.target_features = scene_graph.target_object.get_word_vec()
        self.target_features_torch = Variable(torch.from_numpy(self.target_features).float()).cuda()

    def classify_container(self, parent, level, second_vec=False):
        children_features_torch = [Variable(torch.from_numpy(child.get_features_vec()).float()).cuda()\
                                                             for child in parent.children]
        model = self.combined_net
        children = parent.children
        if level == 0:
            parent_features = parent.get_features_vec()
            parent_features_torch = Variable(torch.from_numpy(parent_features).float()).cuda()
            children_vecs, classification = model.forward_top_level(parent_features_torch,
                                                                    children_features_torch,
                                                                    self.target_features_torch)
        elif level == 1:
            vec = parent.get_second_vec() if second_vec else parent.get_vec()
            children_vecs, classification = model.forward_child_containers(vec,
                                                                           children_features_torch,
                                                                           self.target_features_torch)
        elif level == 2:
            children_features_torch = []
            torch_images = []
            unnocluded_objects = []
            for i in range(len(parent.children)):
                child = parent.children[i]
                if child.is_occluded:
                    continue
                if not self.obj_context_vecs and not self.no_images:
                    if str(child.image) == 'None' and child.image_path is not None:
                        child.image = child.read_image()
                    if str(child.image) == 'None':
                        continue
                    img = Image.fromarray((child.image*255).astype(np.uint8)).convert('RGB')
                    torch_img = Variable(self.normalize(self.to_tensor(self.scaler(img)))).cuda()
                    torch_images.append(torch_img)
                unnocluded_objects.append(child)

            children = unnocluded_objects
            if not self.obj_context_vecs and not self.no_images:
                object_img_fc = self.object_embed_net.compute_image_features(torch_images)
                for i in range(len(unnocluded_objects)):
                    child = unnocluded_objects[i]
                    non_img_features = Variable(torch.from_numpy(child.get_features_vec(include_word_vec=self.obj_word_vecs)).float()).cuda()
                    features = torch.cat((non_img_features,object_img_fc[i]))
                    children_features_torch.append(features)
                    child.set_vec(features)
            elif self.no_images:
                for i in range(len(unnocluded_objects)):
                    context_vec = Variable(torch.from_numpy(child.get_features_vec(include_word_vec=True)).float()).cuda()
                    children_features_torch.append(context_vec)
                    child.set_vec(context_vec)
            else:
                for i in range(len(unnocluded_objects)):
                    context_vec = Variable(torch.from_numpy(child.get_features_vec(target_word_vec=self.target_features)).float()).cuda()
                    children_features_torch.append(context_vec)
                    child.set_vec(context_vec)
            children_vecs, classification = model.forward_child_objects(parent.get_vec(),
                                                                        children_features_torch,
                                                                        self.target_features_torch)

        for i,child in enumerate(children):
            if second_vec:
                child.set_second_vec(children_vecs[i])
            else:
                child.set_vec(children_vecs[i])
        return classification

    def classify_object_occlusion(self, obj, second_vec=False):
        if second_vec:
            return self.object_occlusion_classify_net(obj.get_second_vec(), self.target_features_torch)
        else:
            return self.object_occlusion_classify_net(obj.get_vec(), self.target_features_torch)

    def set_train(self):
        self.target_embed_net.train()
        self.edge_net.train()
        self.container_classify_net.train()
        self.object_occlusion_classify_net.train()
        self.combined_net.train()

    def set_eval(self):
        self.target_embed_net.eval()
        self.edge_net.eval()
        self.container_classify_net.eval()
        self.object_occlusion_classify_net.eval()
        self.combined_net.eval()

    def save_state_dict(self, path):
        torch.save({
            'combined_net': self.combined_net,
            'object_occlusion_classify_net': self.object_occlusion_classify_net,
            'target_embed_net': self.target_embed_net,
            'object_embed_net': self.object_embed_net,
        },
            path)

    def load_state_dict(self, path):
        checkpoint = torch.load(path)
        self.combined_net = checkpoint['combined_net']
        self.object_occlusion_classify_net = checkpoint['object_occlusion_classify_net']
        self.target_embed_net = checkpoint['target_embed_net']
        self.object_embed_net = checkpoint['object_embed_net']

# Baseline models
class RandomModel(object):

    def __init__(self):
        pass

    def set_scene_graph(self, scene_graph):
        pass

    def classify_container(self, container, level):
        return np.random.random()

class OracleModel(object):

    def __init__(self):
        pass

    def set_scene_graph(self, scene_graph):
        pass

    def classify_container(self, container, level):
        return int(container.has_target)

class WordVecModel(object):

    def __init__(self):
        super(WordVecModel, self).__init__()

    def set_scene_graph(self, scene_graph):
        self.scene_graph = scene_graph
        self.target_features = scene_graph.target_object.get_word_vec()

    def classify_containers(self, containers):
        similarities = [self.classify_container(container) for container in containers]
        max_index = similarities.index(max(similarities))
        classifications = [int(i==max_index) for i in range(len(similarities))]
        return classifications

    def classify_objects(self, objects):
        similarities = [self.classify_object_occlusion(obj) for obj in objects]
        max_index = similarities.index(max(similarities))
        classifications = [int(i==max_index) for i in range(len(similarities))]
        return classifications

    def classify_container(self, container, level=None):
        dot = np.dot(container.get_word_vec(), self.target_features)
        mag = np.linalg.norm(self.target_features)*np.linalg.norm(container.get_word_vec())
        return dot/mag

    def classify_object_occlusion(self, obj):
        dot = np.dot(obj.get_word_vec(), self.target_features)
        mag = np.linalg.norm(self.target_features)*np.linalg.norm(obj.get_word_vec())
        return dot/mag

    def load_state_dict(self, path):
        pass

class MostLikelyModel(object):

    def __init__(self):
        super(MostLikelyModel, self).__init__()
        with open('cfg/containers.yaml', 'r') as to_read:
            container_options = yaml.load(to_read, yaml.FullLoader)
        object_type_room_probs = {}
        object_type_container_probs = {}
        object_type_shelf_probs = {}
        for container in sorted(container_options.keys()):
            container_info = container_options[container]
            container_label = container_info['label']
            if 'exclude_prob' in container_info:
                exclude_prob = container_info['exclude_prob']
            else:
                exclude_prob = 0.0

            container_room_probs = container_info['rooms']
            for shelf in container_info['shelves']:
                shelf_info = container_info['shelves'][shelf]
                object_type_list = []
                for object_type in shelf_info['object_types']:
                    if object_type not in object_type_room_probs:
                        object_type_room_probs[object_type] = {}
                        object_type_container_probs[object_type] = {}
                        object_type_shelf_probs[object_type] = {}
                    if container_label not in object_type_container_probs[object_type]:
                        object_type_container_probs[object_type][container_label] = 1.0-exclude_prob
                    else:
                        object_type_container_probs[object_type][container_label]+= 1.0-exclude_prob
                    if shelf not in object_type_shelf_probs[object_type]:
                        object_type_shelf_probs[object_type][shelf] = 1.0
                    else:
                        object_type_shelf_probs[object_type][shelf]+= 1.0
                    for room in container_room_probs:
                        if room not in object_type_room_probs[object_type]:
                            object_type_room_probs[object_type][room] = 0
                        object_type_room_probs[object_type][room]+=container_room_probs[room]*(1-exclude_prob)
        for object_type in object_type_room_probs:
            room_sum = sum(list(object_type_room_probs[object_type].values()))
            for room in object_type_room_probs[object_type]:
                object_type_room_probs[object_type][room]/=room_sum
            container_sum = sum(list(object_type_container_probs[object_type].values()))
            for container in object_type_container_probs[object_type]:
                object_type_container_probs[object_type][container]/=container_sum
            shelf_sum = sum(list(object_type_shelf_probs[object_type].values()))
            for shelf in object_type_container_probs[object_type]:
                object_type_container_probs[object_type][shelf]/=shelf_sum
        self.object_type_room_probs = object_type_room_probs
        self.object_type_container_probs = object_type_container_probs
        self.object_type_shelf_probs = object_type_shelf_probs

    def set_scene_graph(self, scene_graph):
        self.scene_graph = scene_graph
        self.target_obj = scene_graph.target_object

    def classify_container(self, container, level):
        if level==0:
            if container.label not in self.object_type_room_probs[self.target_obj.category]:
                return 0.0
            return self.object_type_room_probs[self.target_obj.category][container.label]
        elif level==1:
            if container.label not in self.object_type_container_probs[self.target_obj.category]:
                return 0.0
            return self.object_type_container_probs[self.target_obj.category][container.label]
        elif level==2:
            return 1.0

    def set_scene_graph(self, scene_graph):
        self.scene_graph = scene_graph
        self.target_obj = scene_graph.target_object

