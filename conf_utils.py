def put_args_into_yaml(args, conf, job_id):
    conf["modality"] = args.modality

    conf["model"]["model_name"] = args.model_name
    conf["model"]["pretrained"] = args.pretrained
    # conf["model"]["residual_block"] = args.residual_block
    
    conf["optimization"]["train_bs"] = args.train_bs
    conf["optimization"]["val_bs"] = args.val_bs

    conf["training"]["num_nodes"] = args.num_nodes
    conf["training"]["continue_training"] = args.continue_training
    conf["training"]["job_id"] = job_id

    conf["scale_invariant_setup"]["scale_invariant"] = args.scale_invariant

    return conf