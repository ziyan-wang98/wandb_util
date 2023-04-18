import logging
import time
import wandb
from argparse import Namespace
import numpy as np

def retry(times, exceptions):
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    print(f'Exception thrown when attempting to run {func}, attempt {attempt} out of {times}')
                    time.sleep(min(2**attempt, 10))
                    attempt += 1

            return func(*args, **kwargs)

        return newfn

    return decorator


def init_wandb(args: Namespace):
    if not args.use_wandb:
        logging.info('Weights and Biases integration disabled')
        return

    if args.wandb_group is None:
        args.wandb_group = f'{args.env}'

    if 'wandb_unique_id' not in args:
        if len(args.tag) != 0:
            args.wandb_unique_id = f'{args.tag}_{args.algorithm_name}_{args.env_name}_{args.wandb_group}_{args.timestamp}'
        else:
            args.wandb_unique_id = f'{args.algorithm_name}_{args.env_name}_{args.wandb_group}_{args.timestamp}'

    logging.info(
        f'Weights and Biases integration enabled. Project: {args.wandb_project}, user: {args.wandb_entity}, '
        f'group: {args.wandb_group}, unique_id: {args.wandb_unique_id}')

    # Try multiple times, as this occasionally fails
    @retry(3, exceptions=(Exception,))
    def init_wandb_func():
        wandb.init(
            dir=args.wandb_dir,
            project=args.wandb_project,
            entity=args.wandb_entity,
            # sync_tensorboard=True,
            id=args.wandb_unique_id,
            name=args.wandb_unique_id,
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            tags=args.wandb_tags,
            resume=False,
            settings=wandb.Settings(start_method='fork'),
        reinit=True
        )

    logging.info('Initializing WandB...')
    try:
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        init_wandb_func()
    except Exception as exc:
        logging.error(f'Could not initialize WandB! {exc}')

    wandb.config.update(args, allow_val_change=True)


def finish_wandb(cfg):
    if cfg.use_wandb:
        import wandb
        wandb.run.finish()
