from auto_memory_model.controller import *


def pick_controller(mem_type='unbounded', dataset='litbank', device='cuda', **kwargs):
    if mem_type == 'learned':
        model = LearnedFixedMemController(dataset=dataset, device=device, **kwargs).to(device)
    elif mem_type == 'lru':
        model = LRUController(dataset=dataset, device=device, **kwargs).to(device)
    elif mem_type == 'unbounded':
        model = UnboundedMemController(dataset=dataset, device=device, **kwargs).to(device)
    elif mem_type == 'unbounded_no_ignore':
        # When singleton clusters are removed during metric calculation, we
        # can avoid the problem of predicting singletons and invalid mentions, and track everything.
        model = UnboundedMemControllerNoIgnore(dataset=dataset, device=device, **kwargs).to(device)
    else:
        raise NotImplementedError(mem_type)

    return model

