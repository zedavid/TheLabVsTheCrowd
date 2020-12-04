import argparse,os,logging
import pickle

import sys
import arpa
import numpy as np
from numpy import random

import matplotlib.pyplot as plt

from orca_utils import rig_utils, hcn_utils, nlg_utils, dialogue_utils, dialogue_utils, utils

DEBUG = False

parser = argparse.ArgumentParser(description='Trains LSTM for an Hybrid Code Network')
parser.add_argument('--datadir','-d',type=str,help='Folder where the data is stored',required=True)
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
parser.add_argument('--overfit','-o',action='store_true',help='uses train as test set to check if overfits')
parser.add_argument('--generate_dialogues', '-g', action='store_true', help='generate output dialogues')
parser.add_argument('--features','-f',type=str,nargs='+',help='features to be used',default=['bow','previous_action', 'nlu'])
parser.add_argument('--dataset_limit_train','-dl', type=int, help='max number of instances in the train dataset', default=None)

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

model_info.n_classes = len(available_actions)

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
y_test, y_pred = hcn_utils.test_model(model_info,
                                      os.path.join(models_dir,f"{'_'.join(model_info.feature_set)}.hdf5"),
                                      subsets_file_lists['test'],
                                      available_actions,
                                      task_dir,
                                      dialogue_settings,
                                      lm_dacts=dact_lm)


# Plot normalized confusion matrix
cm, classes = hcn_utils.plot_confusion_matrix(y_test, y_pred, classes=available_actions, normalize=True,
                      title='Normalised DAct confusion',only_existing_classes=True)

with open(os.path.join(models_dir, 'actions.txt'),'w') as fp:
    fp.write('\n'.join(classes))
np.save(os.path.join(models_dir, 'confusion_matrix.npy'),cm)

plt.savefig(os.path.join(models_dir, f'normalised_confusion_matrix_{".".join(args.features)}.pdf'))
plt.savefig(os.path.join(models_dir, f'normalised_confusion_matrix_{".".join(args.features)}.png'))

utils.print_dict(model_info.__dict__)
# input()

print('Results of the train test: ')
print(f'Acc (without entity replacement): {ta:.4f}')
print(f'Acc (entity replacement): {ta_ent:.4f}')
print(f'Acc (situated dialogue acts): {situated_succ:.4f}')
print(f'Acc (merged update acts): {ta_merged_updates:.4f}')
print(f'Mission Success: {m_s:.4f}')
print(f'End state Success: {end_s:.4f}')
print(f'Correct output: {correct_output:.4f}')
print(f'TRL: {turns_m:.4f}')
if len(turns_m_all):
    print(f'TRL Succ: {sum(turns_m_all)/len(turns_m_all):.4f}')
else:
    print('TRL Succ: NA (no successful dialogue)')
print(f'Average DA perplexity: {perplex:.4f}')
print(f'Average diff DA perplexity: {d_perplex:.4f}')
