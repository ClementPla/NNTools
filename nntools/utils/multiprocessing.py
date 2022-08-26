import torch.distributed as dist
from nntools.tracker import Log
from torch.cuda.amp import autocast


def _start_process(rank=0, manager=None):
    print('Initializing process %i' % rank)
    if manager.multi_gpu:
        dist.init_process_group(backend=manager.config['Manager']['dist_backend'])
    model = manager.get_model_on_device(rank)
    if manager.run_training:
        try:
            manager.train(model, rank)

        except KeyboardInterrupt:
            manager.keyboard_exception_raised = True
        finally:
            manager.tracker.set_status('FAILED')

        if manager.keyboard_exception_raised:
            if manager.is_main_process(rank):
                Log.warn("Killed Process. The model will be registered at %s" % manager.saved_models)
                manager.tracker.set_status('KILLED')

    if manager.is_main_process(rank) and (manager.run_training or manager.save_last):
        manager.save_model(model, 'last')
        manager.register_trained_model()

    if manager.multi_gpu:
        dist.barrier()

    if manager.call_end_function:
        with autocast(enabled=manager.config['Manager']['amp']):
            manager.end(model, rank)

    manager.clean_up()
