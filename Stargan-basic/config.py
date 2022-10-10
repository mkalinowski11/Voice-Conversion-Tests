class Config:
    def __init__(self):
        self.resume_path_gen = "./checkpoints/best.pt"
        self.resume_path_dis = "./checkpoints/best.pt"
        self.optimizers = {
            "gen_lr": 0.0001,
            "dis_lr": 0.00005,
            "beta1": 0.9,
            "beta2": 0.999
        }
        self.hparam = {
        "a": 1,
        "b": 0,
        "lambda_id": 5,
        "lambda_cyc": 10
        }
        self.num_epochs = 1001
        self.epoch_save = 200
        self.start_epoch = 0
        self.gen_freq = 5
        self.batch_size = 1
        self.num_workers = 0
        self.load_checkpoint = False
