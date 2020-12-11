import argparse,os,logging
import pickle

import sys,shutil
import arpa
import numpy as np
from numpy import random
import random

import matplotlib.pyplot as plt

from orca_utils import rig_utils, hcn_utils, nlg_utils, dialogue_utils, dialogue_utils, utils

DEBUG = False


def shuffle_train_dev(trainset, devset):
    # randomise splits, keeping same dev amount of dialogues (8)
    logging.info(f'Shuffling train and dev sets')
    all_files = trainset + devset
    dev_length = len(devset)
    dev_length = 8 if dev_length > 8 else dev_length
    new_devset = random.choices(all_files, k=dev_length)
    new_trainset = [file for file in all_files if file not in new_devset]
    if len(new_devset) + len(new_trainset) != len(all_files):
        logging.error(
            f'Size of the new splits ({len(new_devset) + len(new_trainset)}) '
            f'don\'t match dataset: {len(all_files)}')
        return shuffle_train_dev(trainset, devset)

    random.shuffle(new_trainset)

    return new_trainset, new_devset


parser = argparse.ArgumentParser(description='Trains LSTM for an Hybrid Code Network')
parser.add_argument('--datadir','-d',type=str,help='Folder where the data is stored',required=True)
parser.add_argument('--test-dir','-t',dest='test_dir', type=str, help='Folder where the test data is placed')
parser.add_argument('--epochs','-e',type=int,help='Number of epochs used',default=12)
parser.add_argument('--layer_sizes','-l',type=int,nargs='+',help='Units per layer',default=[128])
parser.add_argument('--simulation_environment', '-se', type=str,
                    help='directory physical environment of the simulated rig', default='requirements/objects/')
parser.add_argument('--robot_db', '-rdb', type=str, help='directory with the robots available',
                    default='requirements/robots')
parser.add_argument('--mission_file', '-m', type=str, help='mission file',
                    default='missions/fire_processing_unit_east_tower.yaml')
parser.add_argument('--experimental-condition',dest='exp_cond',required=True)
parser.add_argument('--states','-s',type=str,help='path to the directory where the nlg state templates are stored', default='states/')
parser.add_argument('--batch_size','-bs',type=int,help='number of batches used in the training procedure',default=3)
parser.add_argument('--dact_lm','-dam',type=str,help='path to the lm trained with the dialogue acts')
parser.add_argument('--stop_criterium', '-st', type=str, help='stop criterium', default='own_metric')
parser.add_argument('--generate_dialogues', '-g', action='store_true', help='generate output dialogues')
parser.add_argument('--features','-f',type=str,nargs='+',help='features to be used',default=['bow','previous_action', 'nlu'])
parser.add_argument('--dataset_limit_train','-dl', type=int, help='max number of instances in the train dataset', default=None)
parser.add_argument('--average_runs', '-avg', type=int,
                    help='Averages the output metrics across several runs', default=1)

random.seed(42)

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger.setLevel(logging.INFO)

args = parser.parse_args()

rig_db = rig_utils.rig_db({'robots': args.robot_db, 'environment': args.simulation_environment, 'mission': args.mission_file})

# loading entity tracker
orca_entity_tracker = rig_utils.EntityTacker(rig_db.dynamic_mission_db['object']['gazebo_id'])

model_info = hcn_utils.ModelInfo(args.batch_size, concat_depth=False)
model_info.layers = args.layer_sizes
model_info.epochs = args.epochs
model_info.stop_criterium = args.stop_criterium
model_info.generate_output = args.generate_dialogues

task_dir = os.path.join(args.datadir, args.exp_cond)
args.features.sort()
features_dir = os.path.join(task_dir,'.'.join(args.features))

subsets_file_lists = {}
for item in os.listdir(features_dir):
    if item in ['train','dev','test']:
        subsets_file_lists[item] = [os.path.join(features_dir,item,feature_file) \
                                    for feature_file in os.listdir(os.path.join(features_dir,item))]

if args.dataset_limit_train is None:
    args.dataset_limit_train = len(subsets_file_lists['train'])

models_dir = os.path.join(task_dir, model_info.config)

if not os.path.isdir(models_dir):
    os.makedirs(models_dir)

for subs in subsets_file_lists:
    if len(subsets_file_lists[subs]) == 0:
        logging.warning('No feature file found for {} subset'.format(subs))
        continue

    # reduce training size to limit
    if subs == 'train' and len(subsets_file_lists[subs]) > args.dataset_limit_train:
        subsets_file_lists[subs] = subsets_file_lists[subs][:args.dataset_limit_train]

    model_info.max_dial_len = hcn_utils.get_feature_size_from_data(subsets_file_lists[subs][0],feat_sep=True)

if args.states:
    dialogue_settings = {}
    for cond in ['user_plan','system_plan']:
        dialogue_settings[cond] = dialogue_utils.dialogue(nlg_utils.NLG(args.states,cond),
                                                          rig_db)
else:
    dialogue_settings = {}

if args.dact_lm:
    logger.debug('Loading DAct language model')
    dact_lm = arpa.loadf(args.dact_lm)[0]
    if DEBUG:
        input(dact_lm)
else:
    dact_lm = None

if model_info.max_dial_len == 0:
    logger.error('No dialogue found')
    sys.exit()

logger.info('Loading actions set')
try:
    with open(os.path.join(task_dir,'orca_action_set.txt'),'r') as lfp:
        available_actions = pickle.load(lfp)
except:
    with open(os.path.join(task_dir,'orca_action_set.txt'), 'r') as lfp:
        available_actions = lfp.read().splitlines()

if DEBUG:
    utils.print_dict(available_actions)
    input()

if args.features is not None:
    model_info.feature_set = args.features
else:
    # jchiyah fix
    for fragment in args.datadir.split(os.sep):
        if '.' in fragment:
            model_info.feature_set = fragment.split('.')
            args.features = model_info.feature_set
            break
    models_dir = os.path.join(args.datadir, model_info.config)

if args.test_dir is None:
    args.test_dir = task_dir

model_info.n_classes = len(available_actions)

metrics = {'turn_accuracy': [],
           'turn_accuracy_ent': [],
           'turn_accuracy_usr': [],
           'turn_accuracy_mu': [],
           'situated_da_success': [],
           'da_perplexity': [],
           'diff_da_perplexity': [],
           'mission_success': [],
           'end_state_success': [],
           'correct_output': [],
           'collaborative_ts': [],
           'avg_length_mission': [],
           'avg_length_mission_succ': []}

outputs = {'y_true_all': [],
           'y_pred_all': []}

for i in range(args.average_runs):
    if args.average_runs > 1:
        # need to remove the folder if we are looping so it trains again
        shutil.rmtree(models_dir)
        subsets_file_lists['train'], subsets_file_lists['dev'] = shuffle_train_dev(
            subsets_file_lists['train'], subsets_file_lists['dev'])

    hcn_utils.train_model(models_dir,
                          f"{'_'.join(model_info.feature_set)}",
                          model_info,
                          subsets_file_lists['train'],
                          subsets_file_lists['dev'],
                          dialogue_settings,
                          log_dir=task_dir,
                          action_set=available_actions)

    ta, ta_ent, m_s, \
    end_s, correct_output, situated_succ, turns_m, \
    perplex, d_perplex, turns_m_all, ta_merged_updates, \
    y_test, y_pred, collaborative_ts = hcn_utils.test_model(model_info,
                                      os.path.join(models_dir,f"{'_'.join(model_info.feature_set)}.hdf5"),
                                      subsets_file_lists['test'],
                                      available_actions,
                                      args.test_dir,
                                      dialogue_settings,
                                      lm_dacts=dact_lm)

    metrics['turn_accuracy'].append(ta)
    metrics['turn_accuracy_ent'].append(ta_ent)
    metrics['turn_accuracy_mu'].append(ta_merged_updates)
    metrics['mission_success'].append(m_s)
    metrics['end_state_success'].append(end_s)
    metrics['correct_output'].append(correct_output)
    if not np.isnan(collaborative_ts):
        metrics['collaborative_ts'].append(collaborative_ts)
    metrics['avg_length_mission'].append(turns_m)
    metrics['situated_da_success'].append(situated_succ)
    metrics['da_perplexity'].append(perplex)
    metrics['diff_da_perplexity'].append(d_perplex)
    metrics['avg_length_mission_succ'] += turns_m_all
    outputs['y_true_all'] += y_test
    outputs['y_pred_all'] += y_pred

# Plot normalized confusion matrix
cm, classes = hcn_utils.plot_confusion_matrix(outputs['y_true_all'], outputs['y_pred_all'],
                        classes=available_actions, normalize=True,
                        title='Normalised DAct confusion',only_existing_classes=True)

with open(os.path.join(models_dir, 'actions.txt'),'w') as fp:
    fp.write('\n'.join(classes))
np.save(os.path.join(models_dir, 'confusion_matrix.npy'),cm)

plt.savefig(os.path.join(models_dir, f'normalised_confusion_matrix_{".".join(args.features)}.pdf'))
plt.savefig(os.path.join(models_dir, f'normalised_confusion_matrix_{".".join(args.features)}.png'))

utils.print_dict(model_info.__dict__)

logger.info('Results of the train test: ')
logger.info(f"Turn accuracy: {np.average(metrics['turn_accuracy']):.4f} ({np.std(metrics['turn_accuracy']):.3f})")
logger.info(f"Turn accuracy entity: {np.average(metrics['turn_accuracy_ent']):.4f} ({np.std(metrics['turn_accuracy_ent']):.3f})")
logger.info(f"Turn accuracy sit: {np.average(metrics['situated_da_success']):.4f} ({np.std(metrics['situated_da_success']):.3f})")
logger.info(f"Turn accuracy mu: {np.average(metrics['turn_accuracy_mu']):.4f} ({np.std(metrics['turn_accuracy_mu']):.3f})")
logger.info(f"TS: {np.average(metrics['mission_success']):.4f} ({np.std(metrics['mission_success']):.3f})")
logger.info(f"TS END: {np.average(metrics['end_state_success']):.4f} ({np.std(metrics['end_state_success']):.3f})")
logger.info(f"Correct output: {np.average(metrics['correct_output']):.4f} ({np.std(metrics['correct_output']):.3f})")
logger.info(f"Colaborative Task Success: {np.average(metrics['collaborative_ts']):.4f} ({np.std(metrics['collaborative_ts']):.3f})")
if len(metrics['avg_length_mission']) > 0:
    logger.info(f"TRL Succ: {np.average(metrics['avg_length_mission']):.4f} ({np.std(metrics['avg_length_mission']):.3f})")
else:
    logger.info(f"TRL Succ: {sum(metrics['avg_length_mission_succ'])/len(metrics['avg_length_mission_succ']):.4f} (??)")
logger.info(f"DA perplexity: {np.average(metrics['da_perplexity']):.4f} ({np.std(metrics['da_perplexity']):.3f})")
logger.info(f"Diff DA perplexity: {np.average(metrics['diff_da_perplexity']):.4f} ({np.std(metrics['diff_da_perplexity']):.3f})")
