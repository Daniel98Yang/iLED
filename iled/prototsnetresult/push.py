import os
import time

import numpy as np
import torch
from receptive_field import ReceptiveFieldInfo, compute_rf_prototype


def find_high_activation_crop(activation_map, ts_len, proto_rf_info: ReceptiveFieldInfo):
    max_activation_idx = np.argmax(activation_map)
    rf_start = max_activation_idx - proto_rf_info.size // 2
    rf_start = 0 if rf_start < 0 else rf_start
    rf_end = rf_start + proto_rf_info.size
    rf_end = ts_len if rf_end > ts_len else rf_end

    return rf_start, rf_end+1


# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader,
                    prototype_network,
                    class_specific=True,
                    preprocess_input_function=None,
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    proto_ts_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype serie comes from
                    prototype_activation_function_in_numpy=None,
                    device=None):

    prototype_network.eval()

    start = time.time()
    proto_layer_shape = prototype_network.proto_layer_shape
    n_prototypes = prototype_network.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         proto_layer_shape.latent_features,
         proto_layer_shape.latent_proto_len])

    '''
    proto_rf_boxes and proto_bounds column:
    0: example index in the entire dataset
    1: proto start index in the input serie
    2: proto end index in the input serie
    3: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 4],
                                    fill_value=-1)
        proto_bounds = np.full(shape=[n_prototypes, 4],
                                            fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 3],
                                    fill_value=-1)
        proto_bounds = np.full(shape=[n_prototypes, 3],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            os.makedirs(proto_epoch_dir, exist_ok=True)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network.num_classes

    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the serie
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input,
                                    start_index_of_search_batch,
                                   prototype_network,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bounds,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   proto_ts_filename_prefix=proto_ts_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
                                   device=device)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bounds)

    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(proto_layer_shape))
    prototype_network.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).to(device))
    end = time.time()


# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bounds, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               proto_ts_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None,
                               device=None):

    prototype_network.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.to(device)
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = prototype_network.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    tses_proto_dists = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_ts_idx_dict = {key: [] for key in range(num_classes)}
        # ts_y is the time serie's integer label
        for ts_idx, ts_y in enumerate(search_y):
            label = ts_y.item()
            class_to_ts_idx_dict[label].append(ts_idx)

    proto_layer_shape = prototype_network.proto_layer_shape
    n_prototypes = proto_layer_shape.num_prototypes
    proto_len = proto_layer_shape.latent_proto_len
    max_dist = proto_layer_shape.latent_features * proto_layer_shape.latent_proto_len

    for proto_idx in range(n_prototypes):
        #if n_prototypes_per_class != None:
        if class_specific:
            # get a class of a particular prototype
            target_class = torch.argmax(prototype_network.prototype_class_identity[proto_idx]).item()
            # if there are no time series of the target_class in this batch
            # we go on to the next prototype
            if len(class_to_ts_idx_dict[target_class]) == 0:
                continue
            # get distances of time series of current class to the current proto
            tses_to_curr_proto_dists = tses_proto_dists[class_to_ts_idx_dict[target_class]][:,proto_idx,:]
        else:
            # if it is not class specific, then we will search through every example
            tses_to_curr_proto_dists = tses_proto_dists[:,proto_idx,:]

        # get the minimal distance to the current proto from the whole batch
        batch_min_proto_dist = np.amin(tses_to_curr_proto_dists)
        if batch_min_proto_dist < global_min_proto_dist[proto_idx]:
            # get index of time serie with smallest distance to the current proto
            batch_argmin_proto_dist = \
                list(np.unravel_index(np.argmin(tses_to_curr_proto_dists, axis=None),
                                      tses_to_curr_proto_dists.shape))
            if class_specific:
                # time series in tses_to_curr_proto_dists are coming from only one class
                # so we have to translate the index to the global index in the whole batch 
                batch_argmin_proto_dist[0] = class_to_ts_idx_dict[target_class][batch_argmin_proto_dist[0]]

            # retrieve the corresponding feature map patch
            ts_idx_in_batch = batch_argmin_proto_dist[0]
            fmap_proto_start_ind = batch_argmin_proto_dist[1] * prototype_layer_stride
            fmap_proto_end_ind = fmap_proto_start_ind + proto_len

            batch_min_fmap_patch_j = protoL_input_[ts_idx_in_batch,
                                                   :,
                                                   fmap_proto_start_ind:fmap_proto_end_ind]

            global_min_proto_dist[proto_idx] = batch_min_proto_dist
            # print(batch_min_fmap_patch_j.shape)
            global_min_fmap_patches[proto_idx] = batch_min_fmap_patch_j

            # get the receptive field boundary of the serie patch
            # that generates the representation
            protoL_rf_info = prototype_network.proto_layer_rf_info
            receptive_field_info = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist, protoL_rf_info)

            # get the whole serie
            original_closest_ts = search_batch_input[receptive_field_info[0]]
            original_closest_ts = original_closest_ts.numpy()
            original_closest_ts = np.transpose(original_closest_ts, (1, 0))
            original_ts_len = original_closest_ts.shape[0]

            # crop out the receptive field
            rf_from_ts = original_closest_ts[receptive_field_info[1]:receptive_field_info[2], :]

            # save the prototype receptive field information
            proto_rf_boxes[proto_idx, 0] = receptive_field_info[0] + start_index_of_search_batch
            proto_rf_boxes[proto_idx, 1] = receptive_field_info[1]
            proto_rf_boxes[proto_idx, 2] = receptive_field_info[2]
            if proto_rf_boxes.shape[1] == 4 and search_y is not None:
                proto_rf_boxes[proto_idx, 3] = search_y[receptive_field_info[0]].item()

            # find the highly activated region of the original time serie
            closest_ts_proto_dist = tses_proto_dists[ts_idx_in_batch, proto_idx, :]
            if prototype_network.prototype_activation_function == 'log':
                proto_act_closest_ts = np.log((closest_ts_proto_dist + 1) / (closest_ts_proto_dist + prototype_network.epsilon))
            elif prototype_network.prototype_activation_function == 'linear':
                proto_act_closest_ts = max_dist - closest_ts_proto_dist
            else:
                proto_act_closest_ts = prototype_activation_function_in_numpy(closest_ts_proto_dist)
            upsampled_proto_act = np.interp(
                np.linspace(0, proto_act_closest_ts.shape[0] - 1, original_ts_len),
                np.arange(proto_act_closest_ts.shape[0]),
                proto_act_closest_ts)
            proto_bound = find_high_activation_crop(upsampled_proto_act, ts_len=original_ts_len, proto_rf_info=protoL_rf_info)
            # crop out the fragemtn with high activation as prototypical part
            cropped_closest_ts = original_closest_ts[proto_bound[0]:proto_bound[1], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bounds[proto_idx, 0] = proto_rf_boxes[proto_idx, 0]
            proto_bounds[proto_idx, 1] = proto_bound[0]
            proto_bounds[proto_idx, 2] = proto_bound[1]
            if proto_bounds.shape[1] == 4 and search_y is not None:
                proto_bounds[proto_idx, 3] = search_y[receptive_field_info[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(proto_idx) + '.npy'),
                            proto_act_closest_ts)
                if proto_ts_filename_prefix is not None:
                    # save the whole serie containing the prototype as npy
                    np.save(os.path.join(dir_for_saving_prototypes,
                                            proto_ts_filename_prefix + '-original' + str(proto_idx) + '.npy'),
                            original_closest_ts)
                    orig_ts_with_add_feature = np.hstack((original_closest_ts, upsampled_proto_act.reshape((-1, 1))))
                    np.save(os.path.join(dir_for_saving_prototypes,
                                            f'{proto_ts_filename_prefix}-original_with_self_act{proto_idx:03d}.npy'),
                            orig_ts_with_add_feature)

                    # if different from the original (whole) serie, save the serie receptive field as npy
                    if rf_from_ts.shape[0] != original_ts_len or rf_from_ts.shape[1] != original_ts_len:
                        np.save(
                            os.path.join(
                                dir_for_saving_prototypes,
                                f'{proto_ts_filename_prefix}-receptive_field{proto_idx:03d}.npy'),
                            rf_from_ts)

                    np.save(
                        os.path.join(
                            dir_for_saving_prototypes, proto_ts_filename_prefix + str(proto_idx) + '.npy'),
                        cropped_closest_ts)

    if class_specific:
        del class_to_ts_idx_dict
