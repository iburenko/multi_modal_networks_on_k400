def put_args_into_yaml(args, conf, job_id):
    conf["modality"] = args.modality

    conf["model"]["model_name"] = args.model_name
    # if args.pretrained is not None:
    #     conf["model"]["pretrained"] = args.pretrained
    # else:
    #     conf["model"]["pretrained"] = False
    # if args.residual_block is not None:
    #     conf["model"]["residual_block"] = args.residual_block
    # else:
    #     conf["model"]["residual_block"] = "basic"
    
    conf["optimization"]["train_bs"] = args.train_bs
    conf["optimization"]["val_bs"] = args.val_bs
    if args.accumulate_batches is not None:
        conf["optimization"]["accumulate_batches"] = int(args.accumulate_batches)
    else:
        conf["optimization"]["accumulate_batches"] = 1

    conf["training"]["num_nodes"] = args.num_nodes
    conf["training"]["continue_training"] = args.continue_training
    conf["training"]["job_id"] = job_id

    # if args.scale_invariant is not None:
    #     conf["scale_invariant_setup"]["scale_invariant"] = args.scale_invariant
    # else:
    #     args.scale_invariant = False

    return conf