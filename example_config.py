import argparse
import time
import numpy as np

def get_config():
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # WandB
    # parser.add_argument("--user_name", type=str, default='marl',help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument('--tag', help='the terminal tag in logger', type=str, default='')
    
    parser.add_argument("--use_wandb", action='store_false', default=True, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")
    parser.add_argument('--wandb-project', default='Test', type=str, help='WandB "Project"')
    parser.add_argument('--wandb-entity', default='Your Entity', type=str, help='WandB username (entity).')
    parser.add_argument('--wandb-job_type', default='train', type=str, help='WandB job type')
    parser.add_argument('--wandb-tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
    parser.add_argument('--wandb-key', default='Your Key', type=str, help='API key for authorizing WandB')
    parser.add_argument('--wandb-dir', default=None, type=str, help='the place to save WandB files')
    parser.add_argument('--wandb-experiment', default='', type=str, help='Identifier to specify the experiment')
    parser.add_argument('--timestamp', default=time.strftime('-(%Y-%m-%d-%H_%M_%S)') + '_' + str(np.random.randint(100)), type=str, help='Timestamp')

    return parser