python hpatches_extract_HardNet.py ../../hpatches-release ../hpatch_model | tee -a log_hpatches_extract_HardNet.log
python hpatches_eval.py --descr-name=hardnet --descr-dir=../../hpatch_model/HardNet_liberty_no_aug/ --split=full --task=verification --task=matching --task=retrieval | tee -a log_hpatches_eval_full.log
python hpatches_results.py --descr-name=hardnet --results-dir=results/ --split=full --task=verification --task=matching --task=retrieval | tee -a log_hpatches_results_full.log





python hpatches_eval.py --descr-name=hardnet --descr-dir=../../hpatch_model/HardNet_liberty_no_aug/ --split=illum --task=matching | tee -a log_hpatches_eval_illum.log
python hpatches_results.py --descr-name=hardnet --results-dir=results/ --split=illum --task=matching | tee -a log_hpatches_results_illum.log


python hpatches_eval.py --descr-name=hardnet --descr-dir=../../hpatch_model/HardNet_liberty_no_aug/ --split=view --task=matching | tee -a log_hpatches_eval_view.log
python hpatches_results.py --descr-name=hardnet --results-dir=results/ --split=view --task=matching | tee -a log_hpatches_results_view.log

