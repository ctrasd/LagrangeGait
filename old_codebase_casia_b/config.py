conf = {
        "WORK_PATH": "./vggc3d_em_nomotion_early_qk_mean_03ce_le1_r3_32em",
        "CUDA_VISIBLE_DEVICES": "0",

        "data": {
            'dataset_path': "/mnt/data/ctr/gait/CASIA_B/",
            'resolution': '64',
            'dataset':'CASIA-B', #'OUMVLP',
            'pid_num': 73, #5153,
            'pid_shuffle': False,
        },
        "model": {
            'lr': 3e-4,
            'hard_or_full_trip': 'full',
            'batch_size': (20,16),
            'restore_iter': 0,
            'total_iter': 100000,
            'margin': 0.2,
            'num_workers': 0,
            'frame_num': 30,
            'hidden_dim':256,
            'model_name': 'VGG3D_veiwem_motion'  
        }
    }
