python main.py --weights BlockGCN_pretrained_weights/ntu120/csub/joint/runs-134-131855.pt --phase test --save-score True \
--config config/nturgbd120-cross-subject/joint.yaml --model model.BLockGCN.Model --work-dir work_dir/ntu120/csub/joint  --device 6 7



python main.py --weights work_dir/ucla/BlockGCN_decay_110_120_140_epochs_new_8heads_deterministic/runs-65-5135.pt --phase test --save-score True --config config/ucla/default.yaml --model model.BlockGCN.Model --work-dir work_dir/ucla  --device 0

python main.py --weights work_dir/ucla/BlockGCN_decay_110_120_140_epochs_new_8heads_deterministic/runs-65-5135.pt --phase test --save-score True --config config/ucla/default.yaml --model model.BlockGCN.Model --work-dir work_dir/ucla  --device 0
