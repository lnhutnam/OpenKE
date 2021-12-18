import os

import openke
from openke.config import Trainer, Tester
from openke.module.model import QuatE
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
    in_path="./benchmarks/FB15K237/",
    nbatches=100,
    threads=8,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=25,
    neg_rel=0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the model
quate = QuatE(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    embedding_dim=200
)

# define the loss function
model = NegativeSampling(
    model=quate,
    loss=SoftplusLoss(),
    batch_size=train_dataloader.get_batch_size(),
    regul_rate=1.0
)

# Mkdir checkpoint
if not os.path.exists('./checkpoint/'):
    os.makedirs('./checkpoint/')

# train the model
trainer = Trainer(model=model, data_loader=train_dataloader,
                  train_times=2, alpha=0.5, use_gpu=True, opt_method="adam")
trainer.run()
quate.save_checkpoint('./checkpoint/quate.ckpt')

# test the model
quate.load_checkpoint('./checkpoint/quate.ckpt')
tester = Tester(model=quate, data_loader=test_dataloader, use_gpu=True)
tester.run_link_prediction(type_constrain=False)