import json
import operator
from copy import copy
import random, time
import logging
import yaml

from orca_utils import utils, rig_utils

DEBUG = False

def subtask_masking(transitions,states,subtask):

    subtask_mask = []
    for st in transitions:

        if not hasattr(states[st], 'sub_tasks'):
            # if not sub_tasks are mentioned, all sub_tasks are allowed
            subtask_mask.append(st)
        elif subtask in states[st].sub_tasks:
            subtask_mask.append(st)

    return subtask_mask

def robot_subtask_masking(transitions,wizard_manager):

    robot_subtask_mask = []
    robot_availability = wizard_manager.rig_db.check_available_robots(wizard_manager.subtask)

    for st in transitions:
        if not hasattr(wizard_manager.nlg_states[st],'slots'):
            robot_subtask_mask.append(st)
        elif 'robot' not in wizard_manager.nlg_states[st].slots:
            robot_subtask_mask.append(st)
        elif not robot_availability and st == 'inform_robot_not_available':
            robot_subtask_mask.append(st)
        elif robot_availability and st != 'inform_robot_not_available':
            robot_subtask_mask.append(st)

    return robot_subtask_mask

def robot_skills_masking(transitions, interaction_manager):

    robot_skill_mask = []

    for st in transitions:
        if st in interaction_manager.nlg.motion_acts:
            if interaction_manager.rig_db.check_robot_skills(interaction_manager.subtask):
                robot_skill_mask.append(st)
            else:
                logging.warning(f'No robot available to perform {st}')
                robot_skill_mask.append('request_grounding_robot')
        else:
            robot_skill_mask.append(st)

    return robot_skill_mask

def wait_robot_masking(transitions,waiting_robots):

    robot_mask = []

    for st in transitions:
        if waiting_robots and st == 'inform_robot_wait':
            robot_mask.append(st)
        elif not waiting_robots and st != 'inform_robot_waiting':
            robot_mask.append(st)

    return robot_mask

def history_masking(transitions,interaction_manager):

    history_mask = []

    for st in transitions:
        if st in ['request_pa_announcement','inform_activate_emergency_shutdown','inform_deactivate_emergency_shutdown']:
            if st not in [prev_st['current_state'] for prev_st in interaction_manager.state_history]:
                history_mask.append(st)
        else:
            history_mask.append(st)

    return history_mask

def robot_out_of_base_masking(transitions,wizard_manager):

    robot_out_of_masking = []

    for st in transitions:
        if st in ['inform_returning_to_base']:
            if wizard_manager.rig_db.check_robot_moving():
                robot_out_of_masking.append(st)
        else:
            robot_out_of_masking.append(st)

    return robot_out_of_masking

def gazebo_action_masking(transitions, interaction_manager):
    '''
    Robot update states and states without transitions filtered when a gazebo action is being performed
    :param transitions:
    :param interaction_manager:
    :return:
    '''

    if hasattr(interaction_manager,'performing_gazebo_action') and interaction_manager.performing_gazebo_action:

        gazebo_action_mask = []

        for st in transitions:
            if st in interaction_manager.nlg.robot_uptate_states or \
                st in interaction_manager.nlg.states_with_no_transitions:
                    gazebo_action_mask.append(st)

        return gazebo_action_mask

    else:
        #in case there is no such attribute or no action is being performed
        return transitions


def robot_speed_masking(transitions, interaction_manager):

    robot_speed_masking = []

    for st in transitions:
        if st == 'inform_robot_velocity':
            if interaction_manager.rig_db.check_robot_speed().find('0.0') == -1:
                robot_speed_masking.append(st)
        else:
            robot_speed_masking.append(st)

    return robot_speed_masking

def wind_state_masking(transitions, interaction_manager):

    states_travelled = [st['current_state'] for st in interaction_manager.state_history]

    if 'inform_wind' in states_travelled:
        if 'inform_wind' in transitions:
            transitions.remove('inform_wind')

    return transitions

def spreading_masking(transition, interaction_manager):

    spreading_masked = []

    time_passed = time.time() - interaction_manager.mission_start

    for st in transition:
        if st in ['inform_risk_spreading','inform_no_risk_spreading']:
            if time_passed > interaction_manager.rig_db.dynamic_mission_db['spreading']['time'] and \
                    st == 'inform_risk_spreakding':
                spreading_masked.append(st)
            if time_passed < interaction_manager.rig_db.dynamic_mission_db['spreading']['time'] and \
                    st == 'inform_no_risk_spreading':
                spreading_masked.append(st)
        else:
            spreading_masked.append(st)

    return spreading_masked


def mask_transitions(transitions,
                     interaction_manager,
                     states):

    #mask_by_subtask
    masked_transitions = subtask_masking(transitions, states, interaction_manager.subtask)

    #mask by robot skills
    masked_transitions = robot_skills_masking(masked_transitions, interaction_manager)

    #mask by robot
    masked_transitions = robot_subtask_masking(masked_transitions, interaction_manager)

    #mask if any robot is waiting
    masked_transitions = wait_robot_masking(masked_transitions, interaction_manager.rig_db.check_waiting_robots())

    #mask states already used
    masked_transitions = history_masking(masked_transitions, interaction_manager)

    #mask if states is moving
    masked_transitions = robot_out_of_base_masking(masked_transitions, interaction_manager)

    #mask states if there is a gazebo action under way
    masked_transitions = gazebo_action_masking(masked_transitions, interaction_manager)

    #mask states if speed is zero
    masked_transitions = robot_speed_masking(masked_transitions, interaction_manager)

    #mask wind state if already used
    masked_transitions = wind_state_masking(masked_transitions, interaction_manager)

    #mask spreading
    masked_transitions = spreading_masking(masked_transitions, interaction_manager)

    return masked_transitions


class PreDefState():
    '''
    For the predefined states that do not
    '''

    def __init__(self, name, init_counter = False):

        self.name = name
        self.utterances = [self.name]
        if init_counter:
            self.transitions_user_turn = {}
            self.transitions_system_turn = {}


class State():

    def __init__(self,state_settings_file,condition, init_counter = False):

        state_settings = yaml.safe_load(open(state_settings_file).read())
        self.name = state_settings['name']
        self.utterances = []
        self.transitions = []
        if 'subtask' in state_settings:
            self.sub_tasks = state_settings['subtask']

        if 'slots' in state_settings:
            self.slots = state_settings['slots']

        # if we want to count the transitions
        if init_counter:
            self.transitions_user_turn = {}
            self.transitions_system_turn = {}

        if 'common' in state_settings:
            # adding stuff common to all personalities
            if 'formulations' in state_settings['common']:
                self.add_utterances(state_settings['common']['formulations'])
            else:
                logging.debug('No prompts found in state {}'.format(self.name))

            if 'transition_states' in state_settings['common']:
                #in the wizard system configuration transitions should be provided as list
                if type(state_settings['common']['transition_states']) == type([]):
                    for state in state_settings['common']['transition_states']:
                        self.transitions.append(state)
                #in the rule based system there will be transitions whenever there is a user input or not
                elif type(state_settings['common']['transition_states']) == type({}):
                    self.transitions = state_settings['common']['transition_states']
        else:
            logging.debug('No common transitions in state {}'.format(self.name))

        # in the wizard condition the configuration is defined in the system, disregard for rule-based system
        if 'conditions' not in state_settings:
            logging.debug('No conditions specs defined in state {}'.format(self.name))
        elif condition not in state_settings['conditions']:
            logging.debug('No {} condition specs in state {}'.format(condition,self.name))
        else:
            condition_settings = state_settings['conditions'][condition]
            if 'formulations' in condition_settings:
                self.add_utterances(condition_settings['formulations'])
            if 'transition_states' in condition_settings:
                for state in condition_settings['transition_states']:
                     self.transitions.append(state)

    def add_utterances(self,utterance_list):

        if isinstance(utterance_list,dict):
            for s in utterance_list:
                self.utterances.append(utterance_list[s])
        else:
            for f in utterance_list:
                self.utterances.append(f)

    def get_transitions(self, pre_defined = [], mode = 'live'):

        if isinstance(self.transitions,list):
            if mode == 'all':
                transitions = self.transitions
                transitions.append(self.name)
                return transitions
            else:
                return self.transitions

        elif isinstance(self.transitions,dict):

            all_transitions = set()
            for tr in self.transitions['system_turn']:
                if tr not in pre_defined:
                    all_transitions.add(tr)

            if 'user_turn' in self.transitions:
                for nlu in self.transitions['user_turn']:
                    for tr in self.transitions['user_turn'][nlu]:
                        if tr not in pre_defined:
                            all_transitions.add(tr)

            return all_transitions

    def get_best_utterance(self):
        #currently returning the best one

        return random.choice(self.utterances)

    def get_best_transition(self,last_turn,dialogue_manager):

        if 'user' in last_turn and 'nlu' in last_turn['user']:
            #look for frame semantics
            if 'user_turn' in self.transitions:
                if DEBUG:
                    utils.print_dict(last_turn['user']['nlu'])
                return self._state_high_prob(self.transitions,dialogue_manager,nlu=last_turn['user']['nlu'])
            elif 'system_turn' in self.transitions:
                return self._state_high_prob(self.transitions,dialogue_manager)
        elif 'system_turn' in self.transitions:
            return self._state_high_prob(self.transitions,dialogue_manager)
        else:
            return self._get_highest_prob_all_states()

    def get_nlu_transitions(self,all_transitions, nlu):
        '''
        Combines d_acts and frame semantics to find the transitions for a given nlu
        :param all_transitions:
        :param nlu:
        :return:
        '''

        tr_nlu = []
        candidate_nlu = []

        transitions = {}

        if 'd_act' in nlu:
            for d_act in nlu['d_act']:
                if d_act not in tr_nlu:
                    tr_nlu.append(d_act)
                if d_act in all_transitions:
                    candidate_nlu.append(d_act)
                    transitions.update(all_transitions[d_act])

        if 'sem' in nlu:
            for frame in nlu['sem']:
                if frame not in tr_nlu:
                    tr_nlu.append(frame)
                if frame in all_transitions:
                    candidate_nlu.append(frame)
                    transitions.update(all_transitions[frame])

        if '|'.join(tr_nlu) in all_transitions:
            return all_transitions['|'.join(tr_nlu)]
        elif 'other' in all_transitions:
            logging.warning(f"combined transition {'|'.join(tr_nlu)} could not be found")
            utils.print_dict(all_transitions)
            transitions.update(all_transitions['other'])
        elif len(candidate_nlu) == 0:
            logging.debug('No NLU match')
            transitions.update({'request_repeat':1.0})

        return transitions

    def _state_high_prob(self, dict_transitions, dialogue_manager, nlu = None):
        '''
        given a dictionary with transition probabilities returns the state with the highest probability
        :param dict_transitions:
        :return:
        '''

        if nlu is not None:
            transitions = self.get_nlu_transitions(copy(dict_transitions['user_turn']), nlu)
        else:
            transitions = copy(dict_transitions['system_turn'])
            if DEBUG:
                utils.print_dict(transitions)

        for st in list(set().union(dialogue_manager.used_no_transitions,dialogue_manager.subtask_travelled_states)):
            if st in transitions:
                try:
                    del transitions[st]
                    logging.debug(f'{st} removed from list of possible transitions')
                except KeyError:
                    logging.error(f'{st} transition not avaialble in the current state, in subtask {dialogue_manager.dm_manager.subtask}')
                    input()

        # masking transitions
        allowed_transitions = mask_transitions(list(transitions.keys()),
                                               dialogue_manager,
                                               dialogue_manager.nlg_states)

        for st in list(set(list(transitions.keys()))-set(allowed_transitions)):
            del transitions[st]

        if len(transitions) > 0:
            return max(transitions.items(), key=operator.itemgetter(1))[0]
        elif len(allowed_transitions) > 0:
            return random.choice(allowed_transitions)
        else:
            logging.error(f'No transitions for state {self.name}, in subtask {dialogue_manager.subtask}')

    def _get_highest_prob_all_states(self):
        '''
        In case the nlu was not observable sample from all possible transitions
        :return:
        '''


        sys_trans = {k: v/2 for k,v in self.transitions['system_turn'].items()}

        if DEBUG:
            utils.print_dict(sys_trans)

        usr_nlu_trans = {}
        #normalising system
        for nlu in self.transitions['user_turn']:
            usr_nlu_trans[nlu] = {k: v/len(self.transitions['user_turn']) for k,v in self.transitions['user_turn'][nlu].items()}

        #normalising user
        usr_trans = {}
        for nlu in usr_nlu_trans:
            for tr in usr_nlu_trans[nlu]:
                if tr in usr_trans:
                    usr_trans[tr] += usr_nlu_trans[nlu][tr]
                else:
                    usr_trans[tr] = usr_nlu_trans[nlu][tr]

        usr_trans = {k: v / 2 for k, v in usr_trans.items()}

        #merging
        trans = sys_trans
        for tr in usr_trans:
            if tr in trans:
                trans[tr] += usr_trans[tr]
            else:
                trans[tr] = usr_trans[tr]

        return self._state_high_prob(trans)

    def print_state_content(self):

        print('Utterances:\n{}'.format(json.dumps(self.utterances,indent=2)))
        print('Transitions:\n{}'.format(json.dumps(self.transitions,indent=2)))

