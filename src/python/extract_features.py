import argparse, sys
import json
import random
import logging
import os, re
import pickle
import string
from random import shuffle
import numpy as np

sys.path.append('..')
from orca_utils import utils, emb_utils, hcn_utils, nlu_utils, rig_utils

str_trans = str.maketrans(dict.fromkeys(string.punctuation))

sess_image = None
sess_use = None
sess_cvts = None
sess_cvtd = None

DEBUG = False

def load_partitions():

    with open(os.path.join(aux_path,'train_dialogues.txt'),'r') as fp:
        dialogue_subsets['train'] = fp.read().splitlines()
    with open(os.path.join(aux_path,'dev_dialogues.txt'),'r') as fp:
        dialogue_subsets['dev'] = fp.read().splitlines()
    with open(os.path.join(aux_path,'test_dialogues.txt'),'r') as fp:
        dialogue_subsets['test'] = fp.read().splitlines()

def get_previous_action(dialogue, prev_index):

    prev_action = np.zeros((len(valid_states),))

    if prev_index == -1:
        return prev_action
    else:
        d_act = dialogue[prev_index]['current_state']

    if d_act in valid_states:
        prev_action[valid_states.index(d_act)] = 1
    else:
        prev_action[valid_states.index('UNK')] = 1

    return prev_action


def extract_features_subset(subset):

    for d,dial in enumerate(subset):
        # creating entities dict for current dialogue

        feature_file =  f"{os.path.join(output_feat_dir,str(dial))}.hdf5"

        if os.path.isfile(feature_file):
            logger.warning(f"feature file {feature_file} already exists")
            continue

        target_data = np.zeros((max_dialogue_len, 1))
        df = {}

        if 'context_concat' in features_sizes:
            context_vector = rig_utils.create_orca_context_vector(rig_db.mission_db, rig_db.environment, skip_common=True)
        else:
            context_vector = rig_utils.create_orca_context_vector(rig_db.mission_db, rig_db.environment, gazebo_state=True)

        user_turns = []
        previous_states = []

        if 'convert_dialogue' in args.features:
            dialogue_history = []

        for f in features_sizes:
            df[f] = np.ones((max_dialogue_len,features_sizes[f]))

        for t, turn in enumerate(all_dialogues[dial]['dialogue']):

            if 'action_mask' in turn:
                action_mask_available = True

            try:
                target_data[t] = valid_states.index(turn['current_state'])
            except:
                print(f"{turn['current_state']} not found")
                for da in target_data:
                    print(valid_states[int(da[0])])
                sys.exit()
            feat_values = {}

            if 'user' not in turn:
                turn['user'] = '<SIL>'

            user_turns.append(turn['user'])

            if 'wrd_emb' in args.features:
                # where the user turn has some content

                if args.embeddings_file:
                    if emb_type == 'glove':
                        feat_values['wrd_emb_glove'] = emb_utils.get_utterance_embedding(glove_model,turn['user'],glove_dim,wrd_idx)
                    else:
                        feat_values['wrd_emb_gn'] = emb_utils.get_utterance_embedding_w2v(emb,turn['user'],emb.vector_size)

            if 'bow' in args.features:
                feat_values['bow'] = bow.transform([turn['user']]).toarray().reshape((features_sizes['bow'],))

            if 'context' in args.features:
                feat_values['context'] = rig_utils.get_orca_context_features(context_vector,
                                                                             turn['time'],
                                                                             turn['situation_db'],
                                                                             turn['current_state'],
                                                                             turn, gazebo_state=True,
                                                                             init_mission_db=rig_db.mission_db)
                context_vector = feat_values['context']


            if 'previous_action' in args.features:
                feat_values['previous_action'] = get_previous_action(all_dialogues[dial]['dialogue'],t-1)

            if 'api' in args.features:
                logging.info('No api features available for orca')
                continue

            if 'action_mask' in args.features:
                feat_values['action_mask'] = np.zeros((len(valid_states),))
                mask_indexes = [valid_states.index(a) for a in turn['action_mask']]
                for m in mask_indexes:
                    feat_values['action_mask'][m] = 1


            if 'nlu' in args.features:
                if 'nlu' in turn:
                    if isinstance(turn['nlu'], str):
                        nlu_tracker.get_nlu_vector(json.loads(turn['nlu']))
                    else:
                        nlu_tracker.get_nlu_vector(turn['nlu'])
                else:
                    #returns a vector with zeros
                    nlu_tracker.get_default_vector()
                feat_values['nlu'] = nlu_tracker.nlu_vect


            for f in feat_values:
                #if f == 'context':
                #   print(feat_values[f])
                if isinstance(feat_values[f],list):
                    if len(feat_values[f]) != features_sizes[f]:
                        logging.error('Feature size does not match for {}'.format(f))
                    df[f][t,:] = np.array(feat_values[f],dtype=float)

                elif isinstance(feat_values[f],dict):
                    if len(feat_values[f]) != features_sizes[f]:
                        logging.error('Feature size does not match for {}'.format(f))
                        utils.print_dict(feat_values[f])
                        if f == 'context':
                            utils.print_dict(turn['situation_db'])
                        if f == 'nlu':
                            nlu_tracker.get_default_vector()
                            utils.print_dict(nlu_tracker.nlu_vect)
                        input()
                    try:
                        df[f][t,:] = np.array(list(feat_values[f].values()),dtype=float)
                    except:
                        utils.print_dict(feat_values[f])
                        input(f)
                else:
                    df[f][t,:] = feat_values[f]

            previous_states.append(turn['current_state'])

        logger.info(f'Creating file {feature_file}')
        if DEBUG:
            input()
        hcn_utils.data_to_h5(feature_file,df,target_data)


def get_all_utterances():

    all_utt = []

    for d in all_dialogues:
        if all_dialogues[d]['dialogue_id'] in dialogue_subsets['test']:
            #don't use the utterance in the test set to build the bow model
            logger.debug(f"skipping {all_dialogues[d]['dialogue_id']} since it belongs to the test set")
            continue
        try:
            all_utt += [d['user'].translate(str_trans).lower() if 'user' in d else '<SIL>' for d in all_dialogues[d]['dialogue']]
        except:
            ### DBG ##
            utils.print_dict(all_dialogues[d]['dialogue'])
            input()

    return all_utt



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='performs the feature extraction for the orca domain')
    parser.add_argument('--train_dir','-tr',type=str,help='path to the training dir',required=True)
    parser.add_argument('--test_dir',type=str,help='path to test dir')
    parser.add_argument('--simulation_environment', '-se', type=str,
                        help='directory physical environment of the simulated rig', default='requirements/objects/')
    parser.add_argument('--robot_db', '-rdb', type=str, help='directory with the robots available', default='requirements/robots')
    parser.add_argument('--mission_file', '-m', type=str, help='mission file',default='missions/fire_processing_unit_east_tower.yaml')
    parser.add_argument('--embeddings_file','-ef',type=str,help='path to the file containt the embeddings used')
    parser.add_argument('--features', '-f', type=str, nargs='+', help='features extract',
                        default=['bow','previous_action', 'nlu'])
    parser.add_argument('--experimental-condition',dest='exp_cond',required=True)
    parser.add_argument('--action_mask_type', '-am', type=str, help='type of action mask', default='shcn')
    parser.add_argument('--dataset_limit','-dl', type=int, help='max number of instances in the train/dev resulting dataset', default=None)

    args = parser.parse_args()

    #initialise history
    features_sizes = {}
    #initalise subsets
    dialogue_subsets = {'test':[],'train':[],'dev':[]}
    #initialise dict with all dialogues
    all_dialogues = {}

    #loading situation kb
    rig_db = rig_utils.rig_db({'robots': args.robot_db, 'environment': args.simulation_environment, 'mission': args.mission_file})

    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.setLevel(logging.INFO)

    json_extension = f'{args.action_mask_type}.data.json'
    aux_path = os.path.join(args.train_dir, args.exp_cond) #where partitions and vocabs are saved

    # creating test set splits
    for file in os.listdir(args.train_dir):
        if file.endswith(json_extension):
            dialogue = json.load(open(os.path.join(args.train_dir, file)))
            dialogue_subsets['train'].append(dialogue['dialogue_id'])
            all_dialogues[dialogue['dialogue_id']] = dialogue

    if args.test_dir is not None:
        for file in os.listdir(args.test_dir):
            if file.endswith(json_extension):
                dialogue = json.load(open(os.path.join(args.test_dir, file)))
                dialogue_subsets['test'].append(dialogue['dialogue_id'])
                all_dialogues[dialogue['dialogue_id']] = dialogue

    if args.dataset_limit is None:
        args.dataset_limit = len(dialogue_subsets['train'])

    # reduce number of dialogues to match that of the embodiment dataset
    dataset_limit = args.dataset_limit
    dataset_size = len(dialogue_subsets['train'])
    if dataset_size > dataset_limit:
        print(f'Size of dataset ({dataset_size}) above limit ({dataset_limit}), so reducing it')
        random.shuffle(dialogue_subsets['train'])
        dialogue_subsets['train'] = dialogue_subsets['train'][:dataset_limit]

    load_partitions()

    print(f"Dataset size: train={len(dialogue_subsets['train'])}, test={len(dialogue_subsets['test'])}, dev={len(dialogue_subsets['dev'])}")

    # getting the longest dialogue
    _, max_dialogue_len = hcn_utils.get_state_list(all_dialogues)

    # loading states from previous files
    action_file = os.path.join(aux_path, 'orca_action_set.txt')
    logger.info('Loading system actions from {}'.format(action_file))
    with open(action_file, 'r') as lfp:
        valid_states = lfp.read().splitlines()

    print(f'Total action set size: {len(valid_states)}')

    if 'wrd_emb' in args.features:
        if args.embeddings_file:
            if args.embeddings_file.endswith('bin'):
                emb = emb_utils.loading_gensim_embeddings(args.embeddings_file)
                features_sizes['wrd_emb_gn'] = emb.vector_size
                emb_type = 'gensim'
            else:
                type, num_data_points, dimension, extension = args.embeddings_file.split(os.sep)[-1].split('.',3)
                glove_dim = int(re.findall('(\d+)d',dimension)[0])
                glove_model, wrd_idx = emb_utils.load_glove_vectors(args.embeddings_file,glove_dim)
                # glove_dim = 300
                features_sizes['wrd_emb_glove'] = glove_dim
                emb_type = 'glove'
        else:
            logger.error("Please provide the path to word2vec or glove embedding")
            sys.exit()

    if 'bow' in args.features:
        bow, bow.size = hcn_utils.load_bow(all_utterances = get_all_utterances())
        vocab_file = os.path.join(aux_path, 'vocab.txt')
        if not os.path.isfile(vocab_file):
            with open(vocab_file,'w') as vf:
                vf.write('{}'.format('\n'.join(bow.get_feature_names())))
        features_sizes['bow'] = bow.size

    if 'context' in args.features:
        cntxt_vector = rig_utils.create_orca_context_vector(rig_db.mission_db, rig_db.environment, gazebo_state=True)
        context_file = os.path.join(aux_path, 'context_feat.txt')
        if not os.path.isfile(context_file):
            with open(context_file, 'w') as cfp:
                cfp.write('{}'.format('\n'.join(list(cntxt_vector.keys()))))
        features_sizes['context'] = len(cntxt_vector)

    if 'action_mask' in args.features:
        features_sizes['action_mask'] = len(valid_states)

    if 'previous_action' in args.features:
        features_sizes['previous_action'] = len(valid_states)

    if 'nlu' in args.features:
        nlu_tracker = nlu_utils.NLU_feat(rig_db.robots,rig_db.environment)
        features_sizes['nlu'] = len(nlu_tracker.nlu_vect)
        nlu_debug_file = os.path.join(aux_path, 'nlu_feat.txt')
        if not os.path.isfile(nlu_debug_file):
            with open(nlu_debug_file, 'w') as nlu_fp:
                nlu_fp.write('{}'.format('\n'.join(list(nlu_tracker.nlu_vect.keys()))))

    if DEBUG:
        utils.print_dict(features_sizes)
        input()

    total_feature_size = sum(features_sizes.values())

    logger.info('Complete feature size {}'.format(total_feature_size))

    for subset in dialogue_subsets:
        if len(dialogue_subsets[subset]) > 0:
            features_list = list(features_sizes.keys())
            features_list.sort()
            output_feat_dir = os.path.join(aux_path, f"{'.'.join(features_list)}", subset)

            if not os.path.isdir(output_feat_dir):
                os.makedirs(output_feat_dir)

            extract_features_subset(dialogue_subsets[subset])
