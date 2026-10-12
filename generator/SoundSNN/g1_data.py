from itertools import product
import sys
import os
import torch
from torch.nn import ZeroPad2d

sys.path.append(os.getcwd())
from generator.SoundSNN.data_handler import SNNDataHandler
import numpy as np
from generator.mapping_utils.data_handler import DataHandler


def generate_g1_data(handler, size_y, size_x, sequence_length):
    data = {}
    data['fc1'] = {}
    data['fc2'] = {}
    data['fc3'] = {}
    data['fc4'] = {}
    
    fc1_input_raw_data = np.array(handler.parameters['fc1']['input']).astype(np.int8)
    fc1_weight_raw_data = np.array(handler.parameters['fc1']['weight']).astype(np.int8)
    fc1_bias_raw_data = np.array(handler.parameters['fc1']['bias']).astype(np.int32)
    fc1_output_raw_data = np.array(handler.parameters['fc1']['output']).astype(np.int8)
    fc1_input_split_dict = {}
    fc1_weight_split_dict = {}
    fc1_bias_split_dict = {}
    fc1_output_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        fc1_input_split_dict[(core_x, core_y)] = ((0, fc1_input_raw_data.shape[1]), )
        fc1_weight_split_dict[(core_x, core_y)] = ((0, fc1_weight_raw_data.shape[0]), 
                                                   (0, fc1_weight_raw_data.shape[1]))
        fc1_bias_split_dict[(core_x, core_y)] = ((0, fc1_bias_raw_data.shape[0]), )
        fc1_output_split_dict[(core_x, core_y)] = ((0, fc1_output_raw_data.shape[1]), )                           

    for i in range(sequence_length):
        data['fc1'][i] = {}
        data['fc1'][i]['input'] = DataHandler.tensor_split(
            raw_data=fc1_input_raw_data[i], split_dict=fc1_input_split_dict, data_type=1,
            alignment=(16, ), dims=[0])
        data['fc1'][i]['weight'] = DataHandler.tensor_split(
            raw_data=fc1_weight_raw_data, split_dict=fc1_weight_split_dict, data_type=1,
            alignment=(32, None), dims=[0, 1])
        data['fc1'][i]['bias'] = DataHandler.tensor_split(
            raw_data=fc1_bias_raw_data, split_dict=fc1_bias_split_dict, data_type=0,
            alignment=(32, ), dims=[0])
        data['fc1'][i]['output'] = DataHandler.tensor_split(
            raw_data=fc1_output_raw_data[i], split_dict=fc1_output_split_dict, data_type=1,
            alignment=(32, ), dims=[0])

    fc2_input_raw_data = np.array(handler.parameters['fc2']['input']).astype(np.int8)
    fc2_weight_raw_data = np.array(handler.parameters['fc2']['weight']).astype(np.int8)
    fc2_bias_raw_data = np.array(handler.parameters['fc2']['bias']).astype(np.int32)
    fc2_output_raw_data = np.array(handler.parameters['fc2']['output']).astype(np.int8)
    fc2_input_split_dict = {}
    fc2_weight_split_dict = {}
    fc2_bias_split_dict = {}
    fc2_output_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        fc2_input_split_dict[(core_x, core_y)] = ((0, fc2_input_raw_data.shape[1]), )
        fc2_weight_split_dict[(core_x, core_y)] = ((0, fc2_weight_raw_data.shape[0]), 
                                                   (0, fc2_weight_raw_data.shape[1]))
        fc2_bias_split_dict[(core_x, core_y)] = ((0, fc2_bias_raw_data.shape[0]), )
        fc2_output_split_dict[(core_x, core_y)] = ((0, fc2_output_raw_data.shape[1]), )                           

    for i in range(sequence_length):
        data['fc2'][i] = {}
        data['fc2'][i]['input'] = DataHandler.tensor_split(
            raw_data=fc2_input_raw_data[i], split_dict=fc2_input_split_dict, data_type=1,
            alignment=(16, ), dims=[0])
        data['fc2'][i]['weight'] = DataHandler.tensor_split(
            raw_data=fc2_weight_raw_data, split_dict=fc2_weight_split_dict, data_type=1,
            alignment=(32, None), dims=[0, 1])
        data['fc2'][i]['bias'] = DataHandler.tensor_split(
            raw_data=fc2_bias_raw_data, split_dict=fc2_bias_split_dict, data_type=0,
            alignment=(32, ), dims=[0])
        data['fc2'][i]['output'] = DataHandler.tensor_split(
            raw_data=fc2_output_raw_data[i], split_dict=fc2_output_split_dict, data_type=1,
            alignment=(32, ), dims=[0])

    fc3_input_raw_data = np.array(handler.parameters['fc3']['input']).astype(np.int8)
    fc3_weight_raw_data = np.array(handler.parameters['fc3']['weight']).astype(np.int8)
    fc3_bias_raw_data = np.array(handler.parameters['fc3']['bias']).astype(np.int32)
    fc3_output_raw_data = np.array(handler.parameters['fc3']['output']).astype(np.int8)
    fc3_input_split_dict = {}
    fc3_weight_split_dict = {}
    fc3_bias_split_dict = {}
    fc3_output_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        fc3_input_split_dict[(core_x, core_y)] = ((0, fc3_input_raw_data.shape[1]), )
        fc3_weight_split_dict[(core_x, core_y)] = ((0, fc3_weight_raw_data.shape[0]), 
                                                   (0, fc3_weight_raw_data.shape[1]))
        fc3_bias_split_dict[(core_x, core_y)] = ((0, fc3_bias_raw_data.shape[0]), )
        fc3_output_split_dict[(core_x, core_y)] = ((0, fc3_output_raw_data.shape[1]), )                           

    for i in range(sequence_length):
        data['fc3'][i] = {}
        data['fc3'][i]['input'] = DataHandler.tensor_split(
            raw_data=fc3_input_raw_data[i], split_dict=fc3_input_split_dict, data_type=1,
            alignment=(16, ), dims=[0])
        data['fc3'][i]['weight'] = DataHandler.tensor_split(
            raw_data=fc3_weight_raw_data, split_dict=fc3_weight_split_dict, data_type=1,
            alignment=(32, None), dims=[0, 1])
        data['fc3'][i]['bias'] = DataHandler.tensor_split(
            raw_data=fc3_bias_raw_data, split_dict=fc3_bias_split_dict, data_type=0,
            alignment=(32, ), dims=[0])
        data['fc3'][i]['output'] = DataHandler.tensor_split(
            raw_data=fc3_output_raw_data[i], split_dict=fc3_output_split_dict, data_type=1,
            alignment=(32, ), dims=[0])

    fc4_input_raw_data = np.array(handler.parameters['fc4']['input']).astype(np.int8)
    fc4_weight_raw_data = np.array(handler.parameters['fc4']['weight']).astype(np.int8)
    fc4_bias_raw_data = np.array(handler.parameters['fc4']['bias']).astype(np.int32)
    fc4_output_raw_data = np.array(handler.parameters['fc4']['output']).astype(np.int8)
    fc4_input_split_dict = {}
    fc4_weight_split_dict = {}
    fc4_bias_split_dict = {}
    fc4_output_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        fc4_input_split_dict[(core_x, core_y)] = ((0, fc4_input_raw_data.shape[1]), )
        fc4_weight_split_dict[(core_x, core_y)] = ((0, fc4_weight_raw_data.shape[0]), 
                                                   (0, fc4_weight_raw_data.shape[1]))
        fc4_bias_split_dict[(core_x, core_y)] = ((0, fc4_bias_raw_data.shape[0]), )
        fc4_output_split_dict[(core_x, core_y)] = ((0, fc4_output_raw_data.shape[1]), )                           

    for i in range(sequence_length):
        data['fc4'][i] = {}
        data['fc4'][i]['input'] = DataHandler.tensor_split(
            raw_data=fc4_input_raw_data[i], split_dict=fc4_input_split_dict, data_type=1,
            alignment=(16, ), dims=[0])
        data['fc4'][i]['weight'] = DataHandler.tensor_split(
            raw_data=fc4_weight_raw_data, split_dict=fc4_weight_split_dict, data_type=1,
            alignment=(32, None), dims=[0, 1])
        data['fc4'][i]['bias'] = DataHandler.tensor_split(
            raw_data=fc4_bias_raw_data, split_dict=fc4_bias_split_dict, data_type=0,
            alignment=(32, ), dims=[0])
        data['fc4'][i]['output'] = DataHandler.tensor_split(
            raw_data=fc4_output_raw_data[i], split_dict=fc4_output_split_dict, data_type=1,
            alignment=(32, ), dims=[0])

    return data


if __name__ == '__main__':
    handler = SNNDataHandler()
    data = generate_g1_data(handler, size_y=1, size_x=1, sequence_length=39)
