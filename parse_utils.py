def str2bool(input_str):
    if input_str.lower() in ["false", "0"]:
        return False
    elif input_str.lower() in ["true", "1"]:
        return True
    else:
        raise ValueError("Check the values of command line parameters pretrained or scale_invariant!")