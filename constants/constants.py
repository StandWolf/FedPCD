# ---------------------------------------------------------------------------
# Derived from FedMultimodal (https://github.com/usc-sail/fed-multimodal),
# licensed under the Apache License, Version 2.0.
# Modified for FedPCD (2026). See NOTICE for a summary of changes.
# ---------------------------------------------------------------------------

# define feature len mapping
feature_len_dict = {
    'mobilenet_v2':     1280, 
    'whisper_tiny':     384, 
    'mfcc':             80,  
    'bert':             768, 
    'mobilebert':       512,
    'watch_acc':        3,
    'acc':              3,
    'gyro':             3,
    'mag':              3,
    'i_to_avf':         6,
    'v1_to_v6':         6
}

# define num of class dict
num_class_dict = {
    'ucf101':               51, 
    'mit101':               101, 
    'mit10':                10, 
    'mit51':                51, 
    'meld':                 4,
    'crema_d':              4,
    'uci-har':              6,
    'ptb-xl':               5,
    'extrasensory':         4,
    'ku-har':               8,
    'hateful_memes':        2,
    'crisis-mmd':           8,
    'extrasensory_watch':   6,
    'ego4d-ttm':            2,
    'mhealth':              12,
    'wisdm':                6
}


# define max feature len in temporal
max_class_dict = {
    'ucf101':               51, 
    'mit101':               101, 
    'mit10':                10,
    'mit51':                51, 
    'meld':                 6,
    'uci-har':              6,
    'ptb-xl':               5,
    'extrasensory':         6,
    'extrasensory_watch':   6
}