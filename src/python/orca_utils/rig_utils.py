import copy
import json
import re
import os,yaml
import logging
import time
import random

import numpy as np


#dictionaries to convert context items to numbers
import sys

from orca_utils import utils, rig_utils, nlu_utils, state_utils
# from orca_utils.gazebo_utils import loop_wait_gazebo

emergency_status_converter = {'out':0, 'identified': 1, 'solved': 2}
emergency_converter = {'fire':0, 'gas leakage': 1}
severity_converter = {'minor':0, 'average': 1, 'major': 2}
symptom_converter = {'smoke':0, 'high pressure': 1}
battery_level_converter = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
robot_type_converter = {'flying':0, 'ground': 1}
wind_speed_converter = {'low': 1, 'average': 5, 'high': 10}
gazebo_status_converter = {'Standby':0,
                    'In_Mission_Moving': 1,
                    'In_Mission_Takeoff':2,
                    'In_Mission_Landing':3,
                    'Critical_Failure':4}

status_to_int_converter = {'is in standby':0,
                    'is flying': 1,
                    'is moving': 1,
                    'is taking off':2,
                    'is landing':3,
                    'has a critical failure': 4}



class DialogueConfig():

    def __init__(self,dialogue_config_file):

        config_file = yaml.safe_load(open(dialogue_config_file).read())
        self.furhat_ip = config_file['furhat']['ip']
        self.farmi_dir = config_file['farmi_dir']
        self.furhat_left = config_file['furhat']['left']
        self.furhat_right = config_file['furhat']['right']
        self.furhat_all = config_file['furhat']['all']
        self.pre_defined_utt = []
        for utt in config_file['predefined']:
            self.pre_defined_utt.append(utt)

class EntityTacker:

    def __init__(self,initial_location):
        self.robots = []
        self.target_location = initial_location
        self.usr_act_name = 'inspect_fire' #sets user act to default value in the beginning of the dialogue
        self.usr_act_type = None
        self.target_model_name = self.set_target_model()

    def reset_tracker(self, target_location):

        self.usr_act_name = 'inspect_fire'
        self.target_location = target_location
        self.robots = []
        self.usr_act_type = None
        self.target_model_name = self.set_target_model()

    def to_json(self,subtask='inspect',mode='all'):

        json_entity_dicts = []

        if len(self.robots) == 0:
            logging.warning('No active robot')
            json_entity_dict = {'robot':None,
                                'target_location': self.target_location,
                                'usr_act_name': self.usr_act_name,
                                'usr_act_type': self.usr_act_type}
            json_entity_dicts.append(json_entity_dict)
        else:
            for r in self.robots:
                if not r['active']:
                    continue
                if subtask == 'extinguish':
                    if 'extinguish fire' in r['skill']:
                        json_entity_dict = {'robot': r['gazebo_id'],
                                            'target_location': self.target_location,
                                            'usr_act_name': self.usr_act_name,
                                            'usr_act_type': self.usr_act_type,
                                            'target_model_name': self.target_model_name
                                            }
                        json_entity_dicts.append(json_entity_dict)
                else:
                    if 'inspect' in r['skill']:
                        json_entity_dict = {'robot': r['gazebo_id'],
                                            'target_location': self.target_location,
                                            'usr_act_name': self.usr_act_name,
                                            'usr_act_type': self.usr_act_type,
                                            'target_model_name': self.target_model_name
                                            }
                        json_entity_dicts.append(json_entity_dict)

        if len(json_entity_dicts) == 0:
            logging.error('No robot available to perform the task')
            return None

        if mode == '1-best':
            return random.choice(json_entity_dicts)

        return json_entity_dicts

    def update_robots(self, tokens, nlu, robot_db, clear = True):
        '''
        based on the nlu input updates robots in use
        :param tokens:
        :param nlu:
        :return:
        '''

        if clear:
            self.clear_robots()

        d_acts = nlu_utils.extract_user_dialogue_acts(nlu)

        device_name = nlu_utils.get_surface_tokens(tokens, nlu['tokens'])
        # sometimes users may say the "<robot_name> robot"
        if len(device_name.split()) > 1:
            if 'ner' in nlu and nlu['ner'] != []:
                robot_tokens = []
                for ne in nlu['ner']:
                    if ne['ne'] == 'PERSON':
                        robot_tokens += utils.intersect_lists(ne['tokens'],tokens)
                    if ne['ne'] == 'B-CARDINAL:':
                        robot_tokens += utils.intersect_lists(ne['tokens'],tokens)
                device_name = nlu_utils.get_surface_tokens(robot_tokens, nlu['tokens'])
        #if len(device_name.split()) > 2:
        #    logging.warning('robot name still too long: {}'.format(device_name))
        #    input()
        robot_db_entry = get_robot_by_name(device_name, robot_db)
        if robot_db_entry != None:
            if robot_db_entry not in self.robots:
                robot_db_entry['active'] = True
                # since user has chosen a specific robot, clear all robots that the wizard selected
                self.robots.append(robot_db_entry)
                # utils.print_dict(entity_tracker.robots)
        elif 'INSTRUCTION' not in d_acts and 'REQUEST_ACTION' not in d_acts:
            # dealing with the case where the user has provided the type of the robot
            # if there is an instruction waits until the wizard picked a robot
            robot_db_entries = get_robot_by_type(device_name, robot_db)
            if robot_db_entries != []:
                for rbt in robot_db_entries:
                    if rbt not in self.robots:
                        rbt['active'] = True
                        self.robots.append(rbt)
            else:
                if device_name != '':
                    logging.warning('{} not found'.format(device_name))
                    utils.print_dict(self.robots)
                else:
                    logging.warning('No robot is this dialogue act')

            # input(device_name)

    def clear_robots(self):

        self.robots = []

    def reset_target_location(self,location):
        '''
        In case the location is changed in the course of the dialogue
        :param location:
        :return:
        '''

        self.target_location = location

    def set_target_model(self):
        '''
        Converts location names to the gazebo dictionary of locations
        '''

        if 'ne' in self.target_location.split('_'):
            return 'ne_tower_deck_fire'
        elif 'se' in self.target_location.split('_'):
            return 'se_tower_deck_fire'
        elif 'nw' in self.target_location.split('_'):
            return 'nw_tower_deck_fire'
        elif 'sw' in self.target_location.split('_'):
            return 'sw_tower_deck_fire'
        else:
            logging.error('Please enter a valid location [{} NOT VALID]'.format(self.target_location))
            sys.exc_info()

    def update_robot_list(self,robot_list,gazebo_state=False):
        #clearing the existing
        self.clear_robots()
        #adding those mentioned
        logging.info('updating robot list')
        for r in robot_list:
            if gazebo_state:
                rb_info = gazebo_utils.convert_state_from_gazebo_to_dm(
                    gazebo_state.get_robot_status(r['gazebo_id']))
                for item in rb_info:
                    r[item] = rb_info[item]

            self.robots.append(r)

    def return_robots_in_location(self, location=None):

        #return objects is given location only
        if location:
            if self.target_location != location:
                return []

        json_return_instructions = []
        for rb in self.robots:
            if not rb['active']:
                continue
            json_entity_dict = {'robot': rb['gazebo_id'],
                                'target_location': rb['base_location'],
                                'usr_act_name': 'return_to_base',
                                'usr_act_type': self.usr_act_type
                                }
            json_return_instructions.append(json_entity_dict)

        return json_return_instructions

    def send_confirmation_gazebo(self, d_act, system_prompt, subtask, robots, gazebo_state=None):

        if d_act in ['inform_moving', 'inform_emergency_action']:

            # XL robots_in_prompt = rig_utils.get_robot_in_prompt(robots, system_prompt)
            robots_in_prompt = get_robot_in_prompt(robots, system_prompt)
            self.update_robot_list(robots_in_prompt)
            for message in self.to_json(subtask):
                if message['robot']:
                    logging.info('Sending message from confirmation to gazebo:')
                    utils.print_dict(message)
                    if gazebo_state:
                        # if check robot current status in gazebo, for huksy if it is not near recovery point, do not send it.
                        # i.e. husky is already there
                        robot_name = message['robot']
                        logging.info(
                            "Check if send cmd to gazebo. First checking status of " + robot_name + " for d_act: " + d_act)
                        g_status = gazebo_state.get_robot_status(robot_name)

                        if g_status['near_recovery_point'] and g_status['veh_status'] == 'Standby':
                            wait_signal = "reach_path_goal"
                            logging.info(
                                'Sending cmd to gazebo for ' + robot_name + ' ' + wait_signal + '. Just send, Do Not wait !')
                            gazebo_state.initialise_robot_moving_status(robot_name)
                            dm_req = gazebo_utils.convert_dm_request_to_gazebo(message)
                            gazebo_state.send_to_gazebo(dm_req)

                        else:
                            logging.info(
                                "Did not send the robot because " + robot_name + " is either not near_recovery_point or not Standby !")

                    else:
                        # DEBUG mode
                        input()

                    # TODO: check status of the robot before sending the message and send only if robot is available
                    # input()

    def send_confirmation_gazebo(self, d_act, system_prompt, subtask, robots, gazebo_state=None):

        if d_act in ['inform_moving', 'inform_emergency_action']:

            robots_in_prompt = rig_utils.get_robot_in_prompt(robots, system_prompt)
            self.update_robot_list(robots_in_prompt)
            for message in self.to_json(subtask):
                if message['robot']:
                    logging.info('Sending message from confirmation to gazebo:')
                    if gazebo_state:
                        g_status = gazebo_state.get_robot_status(message['robot'])
                        if g_status['near_recovery_point'] and g_status['veh_status'] == 'Standby':
                            wait_signal = "reach_path_goal"
                            logging.info(
                                'Sending cmd to gazebo for ' + message['robot'] + ' ' + wait_signal + '. Just send, Do Not wait !')
                            gazebo_state.initialise_robot_moving_status(message['robot'])
                            dm_req = gazebo_utils.convert_dm_request_to_gazebo(message)
                            gazebo_state.send_to_gazebo(dm_req)

                        else:
                            logging.info(
                                "Did not send the robot because " + message['robot'] + " is either not near_recovery_point or not Standby !")

                    else:
                        #DEBUG mode
                        utils.print_dict(dm_req)
                        input()
                    # TODO: check status of the robot before sending the message and send only if robot is available
                    #input()

class rig_db():

    def __init__(self, req_dict):

        self.robots = load_robots(req_dict['robots'])
        self.environment = load_environment(req_dict['environment'])
        self.mission = req_dict['mission']

        self.mission_db = load_mission(self.robots, self.environment, self.mission)
        self.dynamic_mission_db = self.initialise_kb(self.mission_db)

    def convert_robot_status(self, gazebo_status, vehicle):

        if gazebo_status == 'In_Mission_Takeoff' or gazebo_status == 2:
            return 'is taking off'
        if gazebo_status == 'In_Mission_Moving' or gazebo_status == 1:
            if vehicle.startswith('uav'):
                return 'is flying'
            else:
                return 'is moving'
        if gazebo_status == 'Standby' or gazebo_status == 0:
            return 'is in standby'
        if gazebo_status == 'In_Mission_Landing' or gazebo_status == 3:
            return 'is landing'
        if gazebo_status == 'Critical_Failure' or gazebo_status == 4:
            return 'has a critical failure'


    def initialise_kb(self, situation_kb={}, mode='live', gazebo_state = False):
        '''

        :param mission_db: information provided in the mission file
        :param situation_kb: current situation_kb
        :param mode: 'live' if working with a live system/'wizard' if wizarded interaction/'off-line' for testing dialogue models
        :return:
        '''

        for field in self.mission_db:
            situation_kb[field] = self.mission_db[field]

        # inactivating robots
        if 'robot' in situation_kb:
            for robot in situation_kb['robot']:
                robot['active'] = False
                if mode == 'live':
                    robot['name'] = robot['names'][0]
                    if 'status' not in robot:
                        robot['status'] = 'is in standby'
                    if gazebo_state:
                        rb_info = gazebo_utils.convert_state_from_gazebo_to_dm(
                            gazebo_state.get_robot_status(robot['gazebo_id']))
                        for item in rb_info:
                            robot[item] = rb_info[item]

        if mode == 'live':
            situation_kb['mission_start'] = time.time()
            # situation_kb['mission_time'] = 0.0

        situation_kb['emergency_shutdown'] = True

        return situation_kb

    def get_objet_area_in_prompt(self,utterance):

        logging.debug('Parsing \'{}\' for object'.format(utterance))

        if self.mission_db['object']['name'] in utterance:
            return self.mission_db['object']['name']

        if self.mission_db['area']['name'] in utterance:
            return self.mission_db['area']['name']

        return False

    def get_rbt_gazebo_id(self,rbt_id):

        for rbt in self.dynamic_mission_db['robot']:
            if rbt['id'] == rbt_id:
                return rbt['gazebo_id']

    def gazebo_id_2_rbt_id(self,gazebo_id):

        logging.debug(f'getting id for {gazebo_id}')

        for rbt in self.dynamic_mission_db['robot']:
            if rbt['gazebo_id'] == gazebo_id:
                return rbt['id']

        input('robot not found')

    def deactivate_robot(self,gzb_rbt_id):

        for rbt in self.dynamic_mission_db['robot']:
            if rbt['gazebo_id'] == gzb_rbt_id:
                rbt['active'] = False

    def activate_robot(self,gzb_rbt_id):
        for rbt in self.dynamic_mission_db['robot']:
            if rbt['gazebo_id'] == gzb_rbt_id:
                rbt['active'] = True

    def check_robot_skills(self, subtask):
        '''
        Checks if there is any active robot which is able to the task
        :param subtask:
        :return:
        '''

        for rbt in self.dynamic_mission_db['robot']:
            if not rbt['active']:
                continue

            if subtask == 'extinguish':
                if 'extinguish fire' in rbt['skill']:
                    return True
            else:
                if 'inspect' in rbt['skill']:
                    return True

        return False

    def update_robot_status(self, gazebo_msg):

        for robot in self.dynamic_mission_db['robot']:
            if robot['gazebo_id'] == gazebo_msg['veh_name']:
                robot['battery_level'] = '{} percent'.format(int(100.0*float(gazebo_msg['battery'])))
                robot['speed'] = '{} meters per second'.format(gazebo_msg['velocity'])
                robot['near_recovery_point'] = gazebo_msg['near_recovery_point']
                robot['progress'] = '{} percent'.format(int(100*float(gazebo_msg['progress'])))
                robot['location'] = gazebo_msg['veh_location']
                robot['status'] = self.convert_robot_status(gazebo_msg['veh_status'], gazebo_msg['veh_name'])
                robot['eta'] = gazebo_msg['eta']

    def update_location(self, gazebo_msg, location_type):

        # this is just for a single location, for more locations a loop would be needed
        self.dynamic_mission_db[location_type]['location'] = gazebo_msg['target_locations'][0]['target_loc']

    def update_wind(self, gazebo_msg):

        self.dynamic_mission_db['weather']['wind']['coordinates'] = gazebo_msg['wind_info']

    def check_available_robots(self, subtask):
        '''
        Checks if there are robots active, and appropriate for the current subtask (used by wizard interface)
        :return:
        '''

        is_robot_active = False #controlling if any robot is active, since in that case the question of skill is not relevant

        for r in self.dynamic_mission_db['robot']:
            if r['active']:
                is_robot_active = True
                if subtask in ['inspect','assess_damage'] and \
                        'inspect' in r['skill']:
                    if 'waiting' in r and r['waiting']:
                        return False
                    else:
                        return True
                elif subtask == 'extinguish' and 'extinguish fire' in r['skill']:
                    if 'waiting' in r and r['waiting']:
                        return False
                    else:
                        return True

        if is_robot_active:
            return False
        else:
            return True

    def check_eligible_robots(self, subtask):
        '''
        Checks if the active robot is suitable for the task (used by DM)
        :param subtask:
        :return:
        '''

        for r in self.dynamic_mission_db['robot']:
            if r['active']:
                is_robot_active = True
                if subtask in ['inspect','assess_damage'] and \
                        'inspect' in r['skill']:
                    if 'waiting' in r and r['waiting']:
                        return False
                    else:
                        return True
                elif subtask == 'extinguish' and 'extinguish fire' in r['skill']:
                    if 'waiting' in r and r['waiting']:
                        return False
                    else:
                        return True

        return False

    def check_robot_moving(self):

        for r in self.dynamic_mission_db['robot']:
            if r['status'] != 'is in standby':
                return r['gazebo_id']

        return False


    def check_waiting_robots(self):

        for r in self.dynamic_mission_db['robot']:
            if 'waiting' in r and r['waiting']:
                return r['gazebo_id']
        return False

    def check_robot_speed(self):

        for r in self.dynamic_mission_db['robot']:
            if r['status'] != 'is in standby' and r['active']:
                return r['speed']

        return ''

    def force_robot_to_standby(self, gazebo_id):

        for r in self.dynamic_mission_db['robot']:
            if r['gazebo_id'] == gazebo_id:
                r['status'] = 'is in standby'
                logging.debug(f"forcing {r['name']} to standby")



    def change_waiting_status(self, robot_id, generate_utterance = None, turn = None):
        '''

        :param robot_id:
        :param generate_utterance: when we have the wizard
        :param turn: when we have
        :return:
        '''

        for r in self.dynamic_mission_db['robot']:
            if r['gazebo_id'] == robot_id:
                if 'waiting' in r and r['waiting']:
                    r['waiting'] = False
                    logging.debug(f"Removing {r['id']} from waiting")
                else:
                    logging.debug(f"Setting {r['id']} to wait")
                    r['waiting'] = True
                    if generate_utterance is not None:
                        generate_utterance('inform_robot_wait')
                    if turn is not None:
                        turn['current_state'] = 'inform_robot_wait'

def check_item_db(id, db):

    for i,item in enumerate(db):
        if item['id'] == id:
            return item,i

    return None,-1

def flat_context_vector(context_vector):

    features_dict = {}
    for item in context_vector:
        if isinstance(context_vector[item],list):
            for c,coord in enumerate(['x','y','z']):
                features_dict['{}_{}'.format(item,coord)] = context_vector[item][c]
        else:
            features_dict[item] = context_vector[item]

    return features_dict



def create_orca_context_vector(mission_db, object_db, gazebo_state=None, skip_common=False):
    '''
    Creates initial context vector for orca missions
    :param mission_db:
    :return:
    '''

    cntx_vector = {}
    if gazebo_state:

        for item in mission_db:
            if item == 'emergency_status':
                cntx_vector[item] = emergency_status_converter[mission_db[item]]
            elif item == 'object' or item == 'area':
                if mission_db[item]['location'] != None:
                    cntx_vector['location_x_{}'.format(mission_db[item]['id'])] = mission_db[item]['location'][0]
                    cntx_vector['location_y_{}'.format(mission_db[item]['id'])] = mission_db[item]['location'][1]
                    cntx_vector['location_z_{}'.format(mission_db[item]['id'])] = mission_db[item]['location'][2]
                else:
                    cntx_vector['location_x_{}'.format(mission_db[item]['id'])] = -1.0
                    cntx_vector['location_y_{}'.format(mission_db[item]['id'])] = -1.0
                    cntx_vector['location_z_{}'.format(mission_db[item]['id'])] = -1.0
            elif item == 'robot':

                rb_info = {'battery_level': 1.0,
                           'eta': -1.0,
                           'progress': -1.0,
                           'x': -1.0,
                           'y': -1.0,
                           'z': -1.0,
                           'status': 0,
                           'speed': 0.0,
                           'active': False,
                           'busy': False,
                           'activation_time': -1.0,
                           #'near_recovery_point': True
                           }

                for rb in mission_db[item]:
                    #else:
                    #    gazebo_state.initialise_robot_status()
                    #    rb_info = gazebo_state.convert_state_from_gazebo_to_dm(rb['gazebo_id'])

                    for inf in rb_info:
                        cntx_vector['{}_{}'.format(inf,rb['gazebo_id'])] = rb_info[inf]
            elif item == 'weather':
                continue #wind is not changing at the moment
                if 'wind' in mission_db[item]:
                    cntx_vector['wind_x'] = 1
                    cntx_vector['wind_y'] = 1
                    cntx_vector['wind_z'] = 0
                    cntx_vector['wind_speed'] = 5
                    cntx_vector['wind_duration'] = 999
                    #cntx_vector['wind'] = copy.deepcopy(gazebo_state.get_wind_status()) #XL


            elif item == 'mission_time':
                cntx_vector[item] = mission_db[item]

        cntx_vector['emergency_shutdown'] = False
        cntx_vector['emergency_severity'] = severity_converter[mission_db['emergency_severity']]


    else:
        for item in mission_db:
            if item == 'emergency_status' and not skip_common:
                cntx_vector[item] = emergency_status_converter[mission_db[item]]
            elif item == 'emergency':
                cntx_vector[item] = emergency_converter[mission_db[item]]
            elif item == 'emergency_severity':
                cntx_vector[item] = severity_converter[mission_db[item]]
                #cntx_vector[item] = emergency_status_converter[mission_db[item]]
            elif item == 'area' and not skip_common:
                object = object_db[get_obj_index(mission_db[item], object_db)]
                cntx_vector['{}_id'.format(item)] = get_obj_index(mission_db[item],object_db)
                # get coordinates for the area affected
                for c in ['x','y','z']:
                    if 'location' in object and object['location']:
                        cntx_vector['{}.{}'.format(c,item)] = object['location'][c]
                    else:
                        cntx_vector['{}.{}'.format(c,item)] = -1.0
            elif item == 'object' and not skip_common:
                if isinstance(mission_db[item],list):
                    for o,obj in enumerate(mission_db[item]):
                        cntx_vector['{}_{}'.format(item,o)] = object_db.index(obj['name'])
                        for c in ['x', 'y', 'z']:
                            cntx_vector['{}.{}_{}'.format(c,item,obj)] = obj[c]
                elif isinstance(mission_db[item],dict):
                    object = object_db[get_obj_index(mission_db[item]['name'],object_db)]
                    cntx_vector[object['id']] = get_obj_index(mission_db[item]['name'],object_db)
                    for c in ['x','y','z']:
                        if 'location' in object and object['location']:
                            cntx_vector['{}.{}'.format(c, item)] = object['location'][c]
                        else:
                            cntx_vector['{}.{}'.format(c, item)] = -1.0
            elif item == 'robot' and not skip_common:
                for rb in mission_db[item]:
                    cntx_vector['{}.battery'.format(rb['id'])] = battery_level_converter[rb['battery_level']]
                    # initial status is always idle
                    cntx_vector['{}.status'.format(rb['id'])] = 0
                    # eta is only available once the target is defined
                    cntx_vector['{}.eta'.format(rb['id'])] = get_eta(rb['eta'])
                    # robot possitiong
            elif item == 'mission_time' and not skip_common:
                cntx_vector[item] = mission_db[item]

    return cntx_vector

def get_obj_index(object_name,obj_db):

    for o,obj in enumerate(obj_db):
        if obj['name'] == object_name:
            return o

    return -1

def get_wind_status():

    gazebo_msg = {'act_name': 'wind_status'}

    #send to gazebo

    return [1,2,0,3,99] # for dev purposes

def get_robot_status(id):

    gazebo_msg = {'veh_name': id,
                  'act_name': 'query_status'}

    #send message to gazebo

    #for demonstation purposes
    return {'battery': 0.95,
            'eta': -1,
            'progress': 0.19,
            'veh_location': [-10.0,-30.0,0.7,],
            'velocity': 2.0,
            'veh_status': gazebo_status_converter['In_Mission_Moving']}

def get_eta(eta_string):
    '''
    Assumes string in format "<number_seconds> seconds"
    :param eta_string:
    :return:
    '''

    return int(re.findall('(\d+) seconds',eta_string)[0])

def get_orca_context_features(cntx_vector,time,situation_kb,d_act=None,content=None,gazebo_state=None,skip_common=False,init_mission_db=None):

    if skip_common:
        return cntx_vector

    #update mission time
    cntx_vector['mission_time'] = time

    if isinstance(situation_kb, str):
        situation_kb = json.loads(situation_kb)

    #gets the location for all the locations
    if gazebo_state:
        for f in cntx_vector:
            if f == 'emergency_status':
                try:
                    if isinstance(situation_kb['emergency_status'], int):
                        cntx_vector[f] = situation_kb['emergency_status']
                    else:
                        cntx_vector[f] = emergency_status_converter[situation_kb['emergency_status']]
                except:
                    if 'emergency_status' in situation_kb:
                        print(situation_kb['emergency_status'])
                        input(type(situation_kb['emergency_status']))
                    else:
                        utils.print_dict(situation_kb)
                        input(type(situation_kb))
            elif f.split('_')[0] == 'location':
                if f.split('_',2)[1] == situation_kb['area']['id']:
                    if 'location' in situation_kb['area']:
                        cntx_vector[f] = situation_kb['area']['location']
                    else:
                        cntx_vector[f] = None

                if f.split('_',1)[1] == situation_kb['object']['id']:
                    if 'location' in situation_kb['object']:
                        cntx_vector[f] = situation_kb['object']['location']
                    else:
                        cntx_vector[f] = None

            elif len(f.split('_')) > 1:
                try:
                    field, robot_id = f.rsplit('_',1)
                except:
                    print(f)
                    input()

                for r in situation_kb['robot']:
                    if r['gazebo_id'] == robot_id:
                        robot_init = get_robot_by_id(robot_id,
                                                     init_mission_db['robot'],
                                                     id_type='gazebo_id')
                        if field in ['x','y','z']:
                            if 'location' in r:
                                if field == 'x':
                                    cntx_vector[f] = r['location'][0]
                                elif field == 'y':
                                    cntx_vector[f] = r['location'][1]
                                elif field == 'z':
                                    cntx_vector[f] = r['location'][2]
                            else:
                                cntx_vector[f] = -1.0
                        elif field not in r:
                            #utils.print_dict(r)
                            #input(field)
                            cntx_vector[f] = -1.0
                        elif isinstance(r[field],str):
                            if field == 'battery_level':
                                battery_pattern = r'(\d+) percent'
                                battery = re.findall(battery_pattern, r[field])
                                if len(battery) > 0:
                                    #try:
                                    cntx_vector[f] = float(battery[0])/100.0
                                    #except:
                                    #    print(battery, r[field])
                                    #    input()
                                else:
                                    cntx_vector[f] = battery_level_converter[r[field]]

                            elif field == 'eta':
                                if r[field] == robot_init[field] and status_to_int_converter[r['status']] == 0:
                                    cntx_vector[f] = -1.0
                                else:
                                    time_pattern = r'(\d+) seconds'
                                    time = re.findall(time_pattern, r[field])
                                    if len(time) > 1:
                                        input(time)
                                    cntx_vector[f] = float(time[0])

                            elif field == 'progress':
                                progress_pattern = r'(\d+) percent'
                                progress = re.findall(progress_pattern, r[field])
                                if len(progress) > 0:
                                    cntx_vector[f] = float(progress[0])/100.0
                                else:
                                    cntx_vector[f] = -1.0
                            elif field == 'status':
                                cntx_vector[f] = status_to_int_converter[r[field]]
                            elif field == 'speed':
                                speed_pattern = r'(\d+) metres per second'
                                speed = re.findall(speed_pattern, r[field])
                                if len(speed) > 0:
                                    cntx_vector[f] = float(speed[0])
                                else:
                                    cntx_vector[f] = -1.0
                            else:
                                print(field, r[field])
                                input()
                        else:
                            cntx_vector[f] = r[field]
            elif f == 'wind':
                if 'weather' in situation_kb:
                    if 'wind' in situation_kb['weather']:
                        if situation_kb['weather']['wind']['direction'] == 'south east':
                            cntx_vector['wind_x'] = 1
                            cntx_vector['wind_y'] = -1
                        cntx_vector['wind_speed'] = wind_speed_converter[situation_kb['weather']['wind']['speed']]
            else:
                logging.warning(f"no expecting {f}")
                input()
    else:


        if d_act == None:
            return cntx_vector
        else:
            # context features independent from the environment
            if d_act == 'inform_emergency_solved':
                cntx_vector['emergency_status'] = emergency_status_converter['solved']
                return cntx_vector
            elif d_act == 'inform_emergency_status':
                cntx_vector['emergency_status'] = emergency_status_converter['identified']
                return cntx_vector
            if d_act == 'activate_robot':
               if content != '':
                   cntx_vector['{}.status'.format(content)] = 1
               return cntx_vector
            elif d_act == 'deactivate_robot':
               if content != '':
                   cntx_vector['{}.status'.format(content)] = 0
               return cntx_vector
            elif d_act == 'inform_moving':
               for rbt in get_active_robots(situation_kb):
                   cntx_vector['{}.status'.format(rbt)] = 2
            elif d_act == 'inform_arrival':
               for rbt in get_active_robots(situation_kb):
                   cntx_vector['{}.status'.format(rbt)] = 1
            elif d_act in ['inform_emergency_action','inform_damage_inspection','inform_inspection']:
               for rbt in get_active_robots(situation_kb):
                   cntx_vector['{}.status'.format(rbt)] = 3
            elif d_act in ['inform_returning_to_base']:
               for r in situation_kb['robot']:
                   #checks for active robots and makes them move to the base
                   if cntx_vector['{}.status'.format(r['id'])] != 0:
                       cntx_vector['{}.status'.format(r['id'])] = 1

    return cntx_vector

def rbt_status_index(id,keys):

    for r,k in enumerate(keys):
        if 'status_{}'.format(id) == k:
            return r


def robot_parsing(rbt_list,utterance):

    mentioned_robot_ids = []
    for r,robot in enumerate(rbt_list):
        if robot['name'] in utterance:
            mentioned_robot_ids.append(robot['id'])

    return mentioned_robot_ids


def load_environment(env_dir):

    env = []

    for item_file in os.listdir(env_dir):
        if item_file.endswith('yaml'):
            item = yaml.safe_load(open(os.path.join(env_dir,item_file), 'r').read())
            env.append(item)

    return env

def find_obj_env(obj_id,env):
    '''
    Returns item based on the id
    :param obj_id:
    :param env:
    :return:
    '''

    for obj in env:
        if obj['id'] == obj_id:
            return obj

    return None

def load_robots(robot_dir):
    robots = []

    for robot_def in os.listdir(robot_dir):
        if robot_def.endswith('yaml'):
            robot = yaml.safe_load(open(os.path.join(robot_dir,robot_def), 'r').read())
            robots.append(robot)

    return robots

def get_robot_by_id(robot_id, robot_db, id_type='id'):

    '''
    given the robot id get complete robot entry
    :param robot_id:
    :param robot_db:
    :return:
    '''

    for robot in robot_db:
        if robot[id_type] == robot_id:
            return robot

    return None

def get_robot_by_name(robot_name,robot_db):
    '''
    given the robot name gets the corresponding complete robot entry
    :param robot_name:
    :param robot_db:
    :return:
    '''

    '''
    given the robot name return the robot entry in the robot database
    :param robot_name:
    :param robot_db:
    :return:
    '''


    for rbt in robot_db:
        if robot_name in rbt['names']:
            return rbt

    return None

def search_robots_by_name(robot_name,robot_db):
    '''
    given the name of the robot returns all robots which match the name provided
    :param robot_name:
    :param robot_db:
    :return:
    '''

    rbt_list = []

    for rbt in robot_db:
        if robot_name in rbt['names']:
            rbt_list.append(rbt)

    return rbt_list

def get_robot_by_type(type_tokens,robot_db,mode='all'):
    '''
    given the type of robot return all robots that match that type
    :param type:
    :param robot_db:
    :return:
    '''

    rbt_list = []
    types = set()

    for t,ttk in enumerate(type_tokens.split()):
        for rbt in robot_db:
            if ttk == rbt['robot_type']:
                types.add(rbt['robot_type'])
                rbt_list.append(rbt)

    #utils.print_dict(rbt_list)

    if mode == 'all' and len(set(types)) > 1:
        # if there is one type chosen
        one_each_type_list = []
        for t in types:
            one_type_list = [r for r in robot_db if r['robot_type'] == t]
            one_each_type_list.append(random.choice(one_type_list))

        return one_each_type_list
    else:
        return [random.choice(rbt_list)] if len(rbt_list) > 0 else rbt_list


def get_location_by_name(location_name,rig_db):

    '''
    given the location name return the component entry in the rig database
    :param location_name:
    :param rig_db:
    :return:
    '''

    if location_name == '':
        return []

    locations_matched = []

    for obj_type in ['area','object']:
        if obj_type in rig_db:
            if isinstance(rig_db[obj_type],dict):
                if rig_db[obj_type]['name'].find(location_name) > -1:
                    locations_matched.append(rig_db[obj_type])
            elif isinstance(rig_db[obj_type],list):
                for el in rig_db[obj_type]:
                    if el['name'].find(location_name) > -1:
                        locations_matched.append(rig_db[obj_type])

    return locations_matched

def get_location_gazebo_id(location,rig_db):

    '''
    Converts current rig_id to gazebo id
    :param location:
    :param rig_db:
    :return:
    '''

    for el in rig_db:
        if el['id'] == location['id']:
            #print(el)
            return el['gazebo_id']


def check_robot_index(robot_id, kb):
    '''
    given the robot id returns the index in the robot in the dynamic db
    :param robot_id:
    :param kb:
    :return:
    '''

    for r,robot in enumerate(kb['robot']):
        if robot_id == robot['id']:
            return r

    return None

def load_mission(robot_db,object_db,mission_file):

    mission_dict = yaml.safe_load(open(mission_file,'r').read())
    situation_kb = {'emergency_status': 'out'}
    for key in mission_dict:
        if key == 'robot':
            if key not in situation_kb:
                situation_kb[key] = []
            for rbt in mission_dict[key]:
                id_r = list(rbt.keys())[0]
                rc, index = check_item_db(id_r, robot_db)
                if rc != None:
                    for item in rbt[id_r]:
                        item_name = list(item.keys())[0]
                        rc[item_name] = item[item_name]
                    situation_kb[key].append(rc)
                else:
                    logging.warning('Robot with id {} could not be found in the robot database'.format(id_r))

        elif key == 'object' or key == 'area':
            oc, index = check_item_db(mission_dict[key],object_db)
            if oc != None:
                situation_kb[key] = oc
            else:
                logging.warning('Object with id {} could not be found in the object database'.format(mission_dict[key]))
        elif key == 'spreading':
            if key not in situation_kb:
                situation_kb[key] = {}
            for item in mission_dict[key]:
                if item == 'object':
                    oc, index = check_item_db(mission_dict[key][item],object_db)
                    if oc != None:
                        situation_kb[key][item] = oc
                else:
                    situation_kb[key][item] = mission_dict[key][item]

        else:
            situation_kb[key] = mission_dict[key]

    return situation_kb


def check_item_db(id, db):

    for i,item in enumerate(db):
        if item['id'] == id:
            return item,i

    return None,-1


def get_active_robots(dynamic_db):

    active_robot_list = []

    for rbt in dynamic_db['robot']:
        if rbt['active']:
            if rbt['id'] == 'anm1':
                #anm1 replaced by husky 2
                active_robot_list.append('hsk2')
            else:
                active_robot_list.append(rbt['id'])

    return active_robot_list


def parse_sentence_for_robots(sentence: str, situation_kb: dict) -> list:
    entities_found = []

    for robot in situation_kb['robot']:
        possible_names = robot['names'] + [robot['robot_type']]
        # just not to put it in the situation_kb
        if robot['robot_type'] == 'flying':
            possible_names.append('drone')
            possible_names.append('helicopter')
            possible_names.append('drain')

        if robot['robot_type'] == 'ground':
            possible_names.append("pesky's")

        for name in possible_names:
            if _is_string_in_sentence(name, sentence):
                number = re.findall(r"\d+", sentence)
                if len(number) == 0 or number[0] == robot['name'][-1]:
                    entities_found.append(robot['id'])
                break

    return list(set(entities_found))


def parse_sentence_for_robot_name(sentence: str, situation_kb: dict) -> list:

    entities_found = []

    for robot in situation_kb['robot']:
        possible_names = robot['names'] + [robot['robot_type']]
        # just not to put it in the situation_kb
        if robot['robot_type'] == 'flying':
            possible_names.append('drone')
            possible_names.append('helicopter')

        for name in possible_names:
            if _is_string_in_sentence(name, sentence):
                number = re.findall(r"\d+", sentence)
                if len(number) == 0 or number[0] == robot['name'][-1]:
                    entities_found.append(name)
                break

    return list(set(entities_found))

def _is_string_in_sentence(name: str, sentence: str) -> bool:
    return bool(re.search(name, sentence, re.IGNORECASE))

# old pattern matching
def get_robot_in_prompt(robot_db, prompt, mode='all'):

    robots_in_prompt = []
    corresponding_tokens = []

    for r in robot_db:
        for r_name in r['names']:
            start_of_rbt_str = prompt.find(r_name)
            if start_of_rbt_str > -1:
                if r not in robots_in_prompt:
                    corresponding_tokens.append((start_of_rbt_str,start_of_rbt_str+len(r_name)))
                    robots_in_prompt.append(r)

    if robots_in_prompt == []:
        if mode == 'all':
           return robots_in_prompt
        else:
           return None

    starting_indexes = [c[0] for c in corresponding_tokens]

    if mode == 'all':
        if len(set(starting_indexes)) == len(corresponding_tokens):
            # if more that one robot is found returns all robots
            return robots_in_prompt

        elif len(corresponding_tokens) > len(set(starting_indexes)) and len(set(starting_indexes)) == 1:
            # matched several robots, but returns the one with a longer string matching
            longer_match_index = 0
            match_str_len = 0
            for c,ct in enumerate(corresponding_tokens):
                match_str = ct[1]-ct[0]
                if match_str > match_str_len:
                    longer_match_index = c
                    match_str_len = match_str

            return [robots_in_prompt[longer_match_index]]
        else:
            print(robots_in_prompt)
            input(corresponding_tokens)
    else:
        # if mode is not 'all' or there was only robot found in the prompt return only that robot
        return [random.choice(robots_in_prompt)]

def check_robot_standby(status_dict):

    standby = True
    for r_id in status_dict:
        if status_dict[r_id] != 'is in standby':
            print(r_id, status_dict[r_id])
            return r_id,False


    return None,standby

def check_and_send_robot_to_target(gazebo_state, robot_name, message, d_act, turn=None):
    # now change to check another robot is not in Standby, then waiting for another bot to finish

    #gazebo_state.get_robot_status(robot_name)
    robot_db = gazebo_state.interaction_manager.situation_db['robot']

    status_dict = {robot_name:''}

    if robot_name.startswith('uav'):
        if robot_name == 'uav1':
            status_dict['uav2'] = ''
        else:
            status_dict['uav1'] = ''
    else:
        if robot_name == 'husky1':
            status_dict['husky2'] = ''
        else:
            status_dict['husky2'] = ''

    for rbt_id in status_dict:
        status_dict[rbt_id] = get_robot_by_id(rbt_id,robot_db,id_type='gazebo_id')['status']

    active_robot, all_robots_standby = check_robot_standby(status_dict)

    if not all_robots_standby:
        logging.info(
            "loop wait for another bot {} for d_act {}".format(
                active_robot, d_act['dialogue_act']))

        #wait_signal = "inform_reach_path_goal"
        logging.info(f"Loop waiting for another bot {active_robot} to finish. {robot_name} is gonna be put to wait.")
        if hasattr(gazebo_state.interaction_manager,'generate_utterances_state'):
            #wizard specific behaviour
            print('wizard')
            gazebo_state.interaction_manager.rig_db.change_waiting_status(robot_name, generate_utterance=gazebo_state.interaction_manager.generate_utterances_state)
        else:
            #dialogue manager specific
            gazebo_state.interaction_manager.rig_db.change_waiting_status(robot_name, turn=turn) #trigger the inform wait
            gazebo_state.interaction_manager.reset_timer() #stop the timer
        #gazebo_utils.loop_wait_gazebo(gazebo_state, another_bot, wait_signal, d_act)
        #gazebo_state.interaction_manager.generate_utterances_transitions('inform_moving')#resuming interaction
        #TODO: send other uttterances from other states
    else:
        logging.info("No need to wait for another robot to return to base")

        logging.info("First checking status of " + robot_name)
        g_status = get_robot_by_id(robot_name,robot_db,id_type='gazebo_id')

        # Waiting for robot near the base and Standby, then send it. Cases: in follow-on turn, user asks send it again
        # before previous sending task finishes. In real situation, this can be explicitly confirm with user
        # (Negotiating with user), currently we have not yet had the real interaction between user and gazebo
        # if robot_name.startswith('uav'):
        #     start_wait = time.time()
        #     while not g_status['near_recovery_point'] or not g_status['status'] == 'is in standby':
        #         time.sleep(1)
        #         g_status = get_robot_by_id(robot_name,robot_db,id_type='gazebo_id')
        #         if int((time.time() - start_wait)) % 10 == 0:  # e.g. every 10 seconds
        #             utils.print_dict(g_status)
        #             logging.info("Waiting for {} ready to send. 10 seconds passed ... ... ".format(robot_name))
        if robot_name.startswith('husky'): # Husky
            # Husky may already there, no need to send. TODO checking if the Husky is coming back at the moment
            if not g_status['near_recovery_point'] or not g_status['status'] == 'is in standby':
                logging.info("Did Not Send {} for {}, Because it is Either not near base Or not Standby!".format(robot_name, d_act))
                return

        # Now it is ready and another robot is not in the way to cause potential crash.
        gazebo_state.initialise_robot_moving_status(robot_name)
        dm_req = gazebo_utils.convert_dm_request_to_gazebo(message)
        gazebo_state.send_to_gazebo(dm_req)
        gazebo_state.get_robot_status(robot_name)
        logging.info(
            "{} has been sent for d_act {} , Not waiting for reach_path_goal".format(
                robot_name,
                d_act))


def return_robot_to_base(gazebo_state, robot_name,
                         dact_info):  # could do multiple robots returning to base in parallel with a few seconds deplay

    if robot_name.startswith("uav"):
        logging.info("no need! as uav already in base station. For: {}".format(dact_info))
        return

    g_status = gazebo_state.get_robot_status(robot_name)
    # logger.debug("================== gazebo status ===")
    # logger.debug(g_status)
    # logger.debug("================== end end ===")

    if not g_status['near_recovery_point'] and g_status['veh_status'] == 'Standby':
        req = {}
        req['veh_name'] = robot_name
        req['user_act_name'] = 'go_to_recovery_point'
        gazebo_state.initialise_robot_moving_status(robot_name)
        gazebo_state.send_to_gazebo(req)
        logging.info("Sending {} back to the base.... For: {} \n".format(dact_info))
        # waiting the gazebo sends the signal of reaching the recovery point
        wait_signal = "reach_path_goal"
        gazebo_utils.loop_wait_gazebo(gazebo_state, robot_name, wait_signal, dact_info)

def return_all_robots_to_base(gazebo_state,situation_kb,dact):

    for rbt in situation_kb['robot']:
        return_robot_to_base(gazebo_state, rbt['gazebo_id'],dact)

def  return_robot_to_base_all(gazebo_state, dact_info):
    '''
    When resetting use this one, more efficient when several robots are out there
    :param gazebo_state:
    :param dact_info:
    :return:
    '''
    # currently entity_tracker didn't remember all husky were at goal place.
    # if check robot current status in gazebo, for huksy if it is not near recovery point, return to base.
    logging.info(f"......................................... For: {dact_info}")
    logging.info("Checking and sending all robots back to the base station, if anyone has not yet returned ... ...\n")

    robots = ['uav1', 'uav2']
    for robot_name in robots:
        g_status = gazebo_state.get_robot_status(robot_name)
        while g_status['veh_status'] != 'Standby':
            time.sleep(2)  # it may be moving, so wait a bit longer.
            g_status = gazebo_state.get_robot_status(robot_name)

    robots = ['husky1', 'husky2']  # TODO loading from config
    robots_to_send = []
    for robot_name in robots:
        g_status = gazebo_state.get_robot_status(robot_name)
        while g_status['veh_status'] != 'Standby':  # wait it finishing previous task, to avoid "currently busy" status
            time.sleep(2)  # it may be moving, so wait a bit longer.
            g_status = gazebo_state.get_robot_status(robot_name)

        if not g_status['near_recovery_point']:  # now it is sure: g_status['veh_status'] == 'Standby'
            robots_to_send.append(robot_name)
        else:
            logging.info("{} is near the base, so no need to send to base again. For: {}".format(robot_name, dact_info))

    req = {}
    req['user_act_name'] = 'go_to_recovery_point'

    logging.info(f'robots to return {robots_to_send}')

    if len(robots_to_send) == 0:
        logging.info("No robot to return ... ...")
        return

    elif len(robots_to_send) == 1:
        robot_name = robots_to_send[0]
        req['veh_name'] = robot_name
        gazebo_state.initialise_robot_moving_status(robot_name)
        gazebo_state.send_to_gazebo(req)
        # sent_robot_name = robot_name
        logging.info("----------------------------------------")
        logging.info("Sending {} back to the base, waiting for it to arrive at the base.\n".format(robot_name))
        # now changed to wait all reach path goal, since sometines even sleep 40 second, still crash each other.
        loop_wait_gazebo(gazebo_state, robot_name, "inform_reach_path_goal", dact_info)

    else:  # only consider two robots for now
        status1 = gazebo_state.get_robot_status('husky1')
        status2 = gazebo_state.get_robot_status('husky2')
        y1 = status1['veh_location'][1]
        y2 = status2['veh_location'][1]
        front_robot = 'husky1'
        back_robot = 'husky2'
        if y1 > y2:
            front_robot = 'husky2'
            back_robot = 'husky1'

        logging.info("robot location y1={}, y2={}".format(y1, y2))
        logging.info("Front robot = {}, Back robot = ".format(front_robot, back_robot))

        robot_name = front_robot
        req['veh_name'] = robot_name
        gazebo_state.initialise_robot_moving_status(robot_name)
        gazebo_state.send_to_gazebo(req)
        # sent_robot_name = robot_name
        logging.info("----------------------------------------")
        # logger.info("Sending " + robot_name + " back to the base....\n")
        logging.info("Sending {} back to the base, waiting for it to arrive at the base. For: {}\n".format(robot_name,
                                                                                                          dact_info))
        # now changed to wait all reach path goal, since sometines even sleep 40 second, still crash each other.
        loop_wait_gazebo(gazebo_state, robot_name, "inform_reach_path_goal", dact_info)

        robot_name = back_robot
        req['veh_name'] = robot_name
        gazebo_state.initialise_robot_moving_status(robot_name)
        gazebo_state.send_to_gazebo(req)
        # sent_robot_name = robot_name
        logging.info("----------------------------------------")
        logging.info("Sending {} back to the base, waiting for it to arrive at the base. For: {}\n".format(robot_name,
                                                                                                          dact_info))
        # now changed to wait all reach path goal, since sometines even sleep 40 second, still crash each other.
        loop_wait_gazebo(gazebo_state, robot_name, "inform_reach_path_goal", dact_info)


def check_active_robots(situation_kb):

    for rb in situation_kb['robot']:
        if rb['active']:
            return True

    return False

def get_previous_state_with_transitions(states_dict, previous_dialogues):
    '''
    Safety for states that don't have transtions
    finding the previous state with transtions
    :return:
    '''

    for state in reversed(previous_dialogues):
        if isinstance(states_dict[state], state_utils.PreDefState):
            # these states do not have a transitions
            continue
        if states_dict[state].transitions != []:
            logging.debug('Loading transitions from state {}'.format(state))
            return state

    # fallback to the the most visited state
    logging.warning('Using inform_moving fallback')
    return 'inform_moving'

def get_orca_action_mask(turn,action_set,condition,previous_states,nlg_templates):

    action_mask = np.zeros((len(action_set),))

    current_da = turn['current_state']
    if 'situation_db' not in turn:
        utils.print_dict(turn)
    if isinstance(turn['situation_db'], str):
        dynamic_db = json.loads(turn['situation_db'])
    else:
        dynamic_db = turn['situation_db']

    time = turn['time']

    # allowing requests to be repeated
    if current_da.startswith('request'):
        action_mask[action_set.index(current_da)] = 1

    #predefined acts
    for st in nlg_templates['user_plan'].fixed_dialogue_acts: #condition does not mater here
        if 'gesture' in current_da.split('_') and st == current_da:
            #avoids repeating
            continue
        if st in action_set:
            action_mask[action_set.index(st)] = 1

    if time > dynamic_db['mission_time']:
        action_mask[action_set.index('mission_timeout')] = 1
    else:
        action_mask[action_set.index('inform_time_left')] = 1

    if time > dynamic_db['spreading']['time']:
        action_mask[action_set.index('inform_risk_spreading')] = 1
    else:
        action_mask[action_set.index('inform_no_risk_spreading')] = 1

    if 'inform_wind' not in previous_states:
        action_mask[action_set.index('inform_wind')] = 1

    if 'intro_hello' not in previous_states:
        action_mask[action_set.index('intro_hello')] = 1

    active_robots = check_active_robots(dynamic_db)

    if active_robots:
        action_mask[action_set.index('inform_robot_available')] = 1
        action_mask[action_set.index('inform_robot_not_available')] = 1
        action_mask[action_set.index('inform_robot_eta')] = 1
        action_mask[action_set.index('inform_arrival')] = 1
        action_mask[action_set.index('inform_inspection')] = 1
        action_mask[action_set.index('inform_robot_capabilities')] = 1
        action_mask[action_set.index('inform_robot_battery')] = 1
        action_mask[action_set.index('inform_robot_status')] = 1
        action_mask[action_set.index('inform_robot_progress')] = 1
        # transitions occuring automatically when the connection with gazebo is present
        if turn['subtask'] == 'inspect':
            action_mask[action_set.index('inform_emergency_status')] = 1
            action_mask[action_set.index('inform_inspection')] = 1
        if turn['subtask'] == 'extinguish':
            action_mask[action_set.index('inform_emergency_solved')] = 1
            action_mask[action_set.index('inform_emergency_action')] = 1
        if turn['subtask'] == 'assess_damage':
            action_mask[action_set.index('inform_damage_inspection')] = 1
            action_mask[action_set.index('inform_risk')] = 1
        #action_mask[action_set.index('inform_robot_location')] = 1
        #action_mask[action_set.index('inform_robot_wait')] = 1
        #action_mask[action_set.index('inform_moving')] = 1
        #action_mask[action_set.index('')]


    allowed_states = []

    if condition == 'mixed':

        for c in nlg_templates:
            if isinstance(nlg_templates[c].state_nlg[current_da], state_utils.State):
                if len(nlg_templates[c].states_nlg[current_da].transitions) > 0:
                    for tr_s in nlg_templates[c].states_nlg[current_da].get_transitions('all'):
                        allowed_states.append(tr_s)
                        if tr_s in action_set:
                            action_mask[action_set.index(tr_s)] = 1
                else:
                    # states that are not predefined but still require backtracking to generate the action mask
                    valid_state = get_previous_state_with_transitions(nlg_templates[c].states_nlg, previous_states)
                    for tr_s in nlg_templates[c].states_nlg[valid_state].get_transitions('all'):
                        allowed_states.append(tr_s)
                        if tr_s in action_set:
                            action_mask[action_set.index(tr_s)] = 1
            else:
                valid_state = get_previous_state_with_transitions(nlg_templates[c].states_nlg, previous_states)
                for tr_s in nlg_templates[c].states_nlg[valid_state].get_transitions('all'):
                    allowed_states.append(tr_s)
                    if tr_s in action_set:
                        action_mask[action_set.index(tr_s)] = 1
    else:
        # print(f'Condition: {condition}\n{nlg_templates.keys()}\n\nCurrent DA: {current_da}\n{nlg_templates[condition].states_nlg.keys()}')
        if isinstance(nlg_templates[condition].states_nlg[current_da], state_utils.State):
            if len(nlg_templates[condition].states_nlg[current_da].transitions) > 0:
                for tr_s in nlg_templates[condition].states_nlg[current_da].get_transitions('all'):
                    if tr_s in action_set:
                        allowed_states.append(tr_s)
                        action_mask[action_set.index(tr_s)] = 1
            else:
                # states that are not predefined but still require backtracking to generate the action mask
                valid_state = get_previous_state_with_transitions(nlg_templates[condition].states_nlg, previous_states)
                for tr_s in nlg_templates[condition].states_nlg[valid_state].get_transitions('all'):
                    if tr_s in action_set:
                        allowed_states.append(tr_s)
                        action_mask[action_set.index(tr_s)] = 1
        else:
            valid_state = get_previous_state_with_transitions(nlg_templates[condition].states_nlg, previous_states)
            for tr_s in nlg_templates[condition].states_nlg[valid_state].get_transitions('all'):
                if tr_s in action_set:
                    allowed_states.append(tr_s)
                    action_mask[action_set.index(tr_s)] = 1

    #keep it here in case we want to debug
    action_mask_list = []
    for index in np.nonzero(action_mask)[0]:
        action_mask_list.append(action_set[index])

    return action_mask, action_mask_list


def get_orca_action_mask_from_slots(turn,action_set,nlg_templates=None):

    if isinstance(turn['situation_db'],str):
        dynamic_db = json.loads(turn['situation_db'])
    else:
        dynamic_db = turn['situation_db']

    time = turn['time']
    subtask = turn['subtask']

    action_mask = np.zeros((len(action_set),))

    if time > dynamic_db['mission_time']:
        action_mask[action_set.index('mission_timeout')] = 1
    else:
        action_mask[action_set.index('inform_time_left')] = 1

    if time > dynamic_db['spreading']['time']:
        action_mask[action_set.index('inform_risk_spreading')] = 1
    else:
        action_mask[action_set.index('inform_no_risk_spreading')] = 1

    if not dynamic_db['emergency_shutdown']:
        action_mask[action_set.index('inform_deactivate_emergency_shutdown')] = 1
    else:
        action_mask[action_set.index('inform_activate_emergency_shutdown')] = 1

    active_robots = check_active_robots(dynamic_db)

        #action_mask[action_set.index('')]

    for st in nlg_templates['user_plan'].fixed_dialogue_acts: #condition does not mater here
        if st in action_set:
            action_mask[action_set.index(st)] = 1

    if active_robots:
        action_mask[action_set.index('inform_robot_available')] = 1
        action_mask[action_set.index('inform_robot_eta')] = 1
        action_mask[action_set.index('inform_arrival')] = 1
        action_mask[action_set.index('inform_inspection')] = 1
        action_mask[action_set.index('inform_robot_capabilities')] = 1
        action_mask[action_set.index('inform_robot_battery')] = 1
        action_mask[action_set.index('inform_robot_status')] = 1
        #action_mask[action_set.index('inform_robot_location')] = 1
        action_mask[action_set.index('inform_robot_progress')] = 1
        action_mask[action_set.index('inform_robot_velocity')] = 1
        #action_mask[action_set.index('inform_robot_wait')] = 1
        action_mask[action_set.index('inform_moving')] = 1
        action_mask[action_set.index('inform_arrival')] = 1
        action_mask[action_set.index('inform_inspection')] = 1
        action_mask[action_set.index('inform_returning_to_base')] = 1


    if subtask == 'inspection':
        action_mask[action_set.index('inform_alert')] = 1
        action_mask[action_set.index('inform_plan_selected')] = 1
        action_mask[action_set.index('request_plan_responsible')] = 1

    if subtask == 'extinguish':
        action_mask[action_set.index('inform_emergency_robot')] = 1
        action_mask[action_set.index('inform_emergency_action')] = 1
        action_mask[action_set.index('request_robot_emergency')] = 1

    if subtask == 'assess_damage':
        action_mask[action_set.index('inform_damage_inspection_robot')] = 1
        action_mask[action_set.index('inform_risk')] = 1
        action_mask[action_set.index('inform_mission_completed')] = 1
        action_mask[action_set.index('request_robot_inspect_damage')] = 1
        action_mask[action_set.index('inform_damage_inspection')] = 1

    action_mask[action_set.index('request_pa_announcement')] = 1
    action_mask[action_set.index('request_robot_type')] = 1
    action_mask[action_set.index('db_query')] = 1
    action_mask[action_set.index('inform_emergency_status')] = 1
    action_mask[action_set.index('inform_emergency_solved')] = 1
    action_mask[action_set.index('inform_wind')] = 1

    #keep it here in case we want to debug
    action_mask_list = []
    for index in np.nonzero(action_mask)[0]:
        action_mask_list.append(action_set[index])

    return action_mask, action_mask_list
