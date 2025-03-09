import time
import torch
import numpy as np
import os
from conditional import conditional

from MBM.better_mistakes.model.performance import accuracy
from MBM.better_mistakes.model.labels import make_batch_soft_labels


# HAFrame
from HAFrame.losses import mixing_ratio_scheduler

# topK_to_consider = (1, 5, 10, 20, 100)
topK_to_consider = (1, )

# lists of ids for loggings performance measures
accuracy_ids = ["accuracy_top/%02d" % i for i in topK_to_consider]
dist_avg_ids = ["_avg/%02d" % i for i in topK_to_consider]
dist_top_ids = ["_top/%02d" % i for i in topK_to_consider]
dist_avg_mistakes_ids = ["_mistakes/avg%02d" % i for i in topK_to_consider]
hprec_ids = ["_precision/%02d" % i for i in topK_to_consider]
hmAP_ids = ["_mAP/%02d" % i for i in topK_to_consider]
level_accuracy = None
level_ce_loss = None
level_jsd_loss = None
level_weight_loss = None
level_dissim_loss = None


def run(loader, model, loss_function, distances, all_soft_labels, classes, opts, epoch, prev_steps,
        optimizer=None, is_inference=True, corrector=lambda x: x, hiercos_params=None):
    """
    Runs training or inference routine for standard classification with soft-labels style losses
    """
    global topK_to_consider, accuracy_ids, dist_avg_ids, dist_top_ids, dist_avg_mistakes_ids, hprec_ids, hmAP_ids, \
           level_accuracy, level_ce_loss, level_jsd_loss, level_weight_loss, level_dissim_loss

    if opts.start == "training":
        topK_to_consider = (1,)
    else:
        if opts.data == "wildlife-species":
            topK_to_consider = (1, 5, 10, 20, 91)
        else:
            topK_to_consider = (1, 5, 10, 20, 100)

    # lists of ids for loggings performance measures
    accuracy_ids = ["accuracy_top/%02d" % i for i in topK_to_consider]
    dist_avg_ids = ["_avg/%02d" % i for i in topK_to_consider]
    dist_top_ids = ["_top/%02d" % i for i in topK_to_consider]
    dist_avg_mistakes_ids = ["_mistakes/avg%02d" % i for i in topK_to_consider]
    hprec_ids = ["_precision/%02d" % i for i in topK_to_consider]
    hmAP_ids = ["_mAP/%02d" % i for i in topK_to_consider]

    max_dist = max(distances.distances.values())
    # for each class, create the optimal set of retrievals (used to calculate hierarchical precision @k)
    best_hier_similarities = _make_best_hier_similarities(classes, distances, max_dist)

    # Using different logging frequencies for training and validation
    log_freq = 1 if is_inference else opts.log_freq

    # strings useful for logging
    descriptor = "VAL" if is_inference else "TRAIN"
    loss_id = "loss/" + opts.loss
    dist_id = "ilsvrc_dist"

    # Initialise accumulators to store the several measures of performance (accumulate as sum)
    num_logged = 0
    loss_accum = 0.0
    time_accum = 0.0
    norm_mistakes_accum = 0.0
    flat_accuracy_accums = np.zeros(len(topK_to_consider), dtype=np.float)
    flat_level_accuracy_accums = None
    hdist_accums = np.zeros(len(topK_to_consider))
    hdist_top_accums = np.zeros(len(topK_to_consider))
    hdist_mistakes_accums = np.zeros(len(topK_to_consider))
    hprecision_accums = np.zeros(len(topK_to_consider))
    hmAP_accums = np.zeros(len(topK_to_consider))

    # Affects the behaviour of components such as batch-norm
    if is_inference:
        model.eval()
    else:
        model.train()

    # mixing schedule for opts.loss == "mixed-ce-gscsl"
    if opts.loss == "mixed-ce-gscsl":
        mixing_beta, mixing_alpha = mixing_ratio_scheduler(opts, epoch)

    with conditional(is_inference, torch.no_grad()):
        time_load0 = time.time()
        for batch_idx, (embeddings, target) in enumerate(loader):

            this_load_time = time.time() - time_load0
            this_rest0 = time.time()

            assert embeddings.size(0) == opts.batch_size, "Batch size should be constant " \
                                                          "(data loader should have drop_last=True)"
            if opts.gpu is not None:
                embeddings = embeddings.cuda(opts.gpu, non_blocking=True)
            target = target.type(torch.LongTensor) # converting to LongTensor type
            target = target.cuda(opts.gpu, non_blocking=True)

            if opts.viz_gradcam is not None:
                # HAFS
                from pytorch_grad_cam import GradCAM
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                from pytorch_grad_cam.utils.image import show_cam_on_image
                from PIL import Image
                torch.set_grad_enabled(True)
                if "wide_resnet" in opts.arch:
                    target_layers = [model.features_2.block3.layer[-1]]
                elif "resnet50" in opts.arch:
                    target_layers = [model.features_2[-1]]
                else:
                    raise NotImplementedError("This method is not supported for grad-cam visualization")
                with GradCAM(model=model, target_layers=target_layers) as cam:
                    unique_targets = torch.unique(target)
                    for ut in unique_targets:
                        images = embeddings[target == ut].clone()
                        images = images[: min(5, images.shape[0])]
                        if opts.feature_space == "hier-cos":
                            viz_targets = []
                            max_level = hiercos_params["leaf_class_to_hierarchical_labels"].shape[1]
                            for level in range(max_level):
                                level_labels = hiercos_params["leaf_class_to_hierarchical_labels"][ut, level]
                                if level_labels < 0:
                                    break
                                viz_targets.append(ClassifierOutputTarget(level_labels + hiercos_params["level_wise_node_ids"][level].min()))
                        else:
                            viz_targets = [ClassifierOutputTarget(ut)]
                        for i in range(images.shape[0]):  # Iterate over each image
                            for level, viz_target in enumerate(viz_targets):
                                grayscale_cam = cam(input_tensor=images[i: i+1], targets=[viz_target])
                                # Convert grayscale CAM to a visual overlay
                                grayscale_cam_image = grayscale_cam[0, :]

                                # Assuming you want to overlay on the original input image
                                original_image = images[i].detach().cpu().numpy()
                                original_image = np.transpose(original_image, (1, 2, 0))  # Change from CxHxW to HxWxC

                                # Normalize original image for displaying
                                original_image = (original_image - np.min(original_image)) / (
                                        np.max(original_image) - np.min(original_image))

                                # Generate visualization
                                cam_image = show_cam_on_image(original_image, grayscale_cam_image, use_rgb=True)

                                # Save the CAM visualization
                                gradcam_output = os.path.join(opts.out_folder, 'grad_cam', f'{ut}')
                                if opts.feature_space == "hier-cos":
                                    gradcam_output = os.path.join(gradcam_output, 'level-wise', f'{batch_idx}_{i}')
                                    gradcam_filename = os.path.join(gradcam_output, f'{level}.png')
                                else:
                                    gradcam_filename = os.path.join(gradcam_output, 'gt', f'{batch_idx}_{i}.png')
                                array_image = Image.fromarray(cam_image)
                                new_size = (512, 512)
                                upscaled_image = array_image.resize(new_size, Image.LANCZOS)
                                os.makedirs(gradcam_output, exist_ok=True)
                                upscaled_image.save(gradcam_filename)
                torch.set_grad_enabled(False)
                if batch_idx % log_freq == 0:
                    # get measures
                    print(
                        "**%8s [Epoch %03d/%03d, Batch %05d/%05d]"
                        % (descriptor, epoch, opts.epochs, batch_idx, len(loader))
                    )
                summary, tot_steps = None, None
                continue

            # get model's prediction
            if opts.arch in ["custom_resnet50", "wide_resnet"]:
                if opts.loss in ["hafeat-l12-cejsd-wtconst-dissim",
                                 "hafeat-l7-cejsd-wtconst-dissim",
                                 "hafeat-l3-cejsd-wtconst-dissim"]:
                    output = model(embeddings, target, is_inference)
                else:
                    output = model(embeddings, target)

            elif opts.arch == "haframe_wide_resnet":
                if opts.loss == "mixed-ce-gscsl":
                    features = model.penultimate_feature(embeddings)
                    output = torch.matmul(features, model.classifier_3.weight.T)
                elif opts.loss == "hafeat-l5-cejsd-wtconst-dissim":
                    # wideresnet model with HAFeature losses has no is_inference input in its forward pass
                    output = model(embeddings, target)
                else:  # cross-entropy, or flamingo
                    output = model(embeddings, target)

            elif opts.arch == "haframe_resnet50":
                if opts.loss == "mixed-ce-gscsl":
                    features = model.penultimate_feature(embeddings)
                    output = torch.matmul(features, model.classifier_3.weight.T)
                elif opts.loss in ["hafeat-l12-cejsd-wtconst-dissim",
                                   "hafeat-l7-cejsd-wtconst-dissim",
                                   "hafeat-l3-cejsd-wtconst-dissim"]:
                    output = model(embeddings, target, is_inference)
                else:  # cross-entropy, or flamingo
                    output = model(embeddings, target)

            elif opts.arch == "haframe_vit":
                if opts.loss == "mixed-ce-gscsl":
                    features = model.penultimate_feature(embeddings)
                    output = torch.matmul(features, model.classifier_3.weight.T)
                else:
                    output = model(embeddings)

            else:
                output = model(embeddings)

            # for soft-labels we need to add a log_softmax and get the soft labels
            if opts.arch in ["custom_resnet50"]:
                if opts.loss == "flamingo-l12":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(12, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(12, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = ["jsd_loss@level-%d" % i for i in range(11, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums += model.jsd_loss_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                        except:
                            pass

                elif opts.loss == "flamingo-l7":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(7, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(7, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = ["jsd_loss@level-%d" % i for i in range(6, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums += model.jsd_loss_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                        except:
                            pass

                elif opts.loss == "flamingo-l3":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(3, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(3, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = ["jsd_loss@level-%d" % i for i in range(2, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums += model.jsd_loss_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                        except:
                            pass

                elif opts.loss == "hafeat-l12-cejsd-wtconst-dissim":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(12, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        flat_level_wt_loss_accums = model.l2_weight_list
                        flat_level_dissim_loss_accums = model.dissimiloss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = [f"jsd_loss@level-{i+1}-{i}" for i in range(11, 0, -1)]
                            level_weight_loss = [f"l2_weight_loss@level-{i+1}-{i}" for i in range(11, 0, -1)]
                            level_dissim_loss = [f"dissim_loss@level-{i}" for i in range(11, 0, -1)]
                        except:
                            pass

                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            # flat_level_jsd_loss_accums += model.jsd_loss_list
                            # flat_level_wt_loss_accums += model.l2_weight_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                            flat_level_wt_loss_accums = [sum(accums) for accums in zip(flat_level_wt_loss_accums, model.l2_weight_list)]
                            flat_level_dissim_loss_accums = [sum(accums) for accums in zip(flat_level_dissim_loss_accums, model.dissimiloss_list)]
                        except:
                            pass
                elif opts.loss == "hafeat-l7-cejsd-wtconst-dissim":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(7, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        flat_level_wt_loss_accums = model.l2_weight_list
                        flat_level_dissim_loss_accums = model.dissimiloss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = [f"jsd_loss@level-{i+1}-{i}" for i in range(6, 0, -1)]
                            level_weight_loss = [f"l2_weight_loss@level-{i+1}-{i}" for i in range(6, 0, -1)]
                            level_dissim_loss = [f"dissim_loss@level-{i}" for i in range(6, 0, -1)]
                        except:
                            pass

                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            # flat_level_jsd_loss_accums += model.jsd_loss_list
                            # flat_level_wt_loss_accums += model.l2_weight_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                            flat_level_wt_loss_accums = [sum(accums) for accums in zip(flat_level_wt_loss_accums, model.l2_weight_list)]
                            flat_level_dissim_loss_accums = [sum(accums) for accums in zip(flat_level_dissim_loss_accums, model.dissimiloss_list)]
                        except:
                            pass
                elif opts.loss == "hafeat-l3-cejsd-wtconst-dissim":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(3, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        flat_level_wt_loss_accums = model.l2_weight_list
                        flat_level_dissim_loss_accums = model.dissimiloss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = [f"jsd_loss@level-{i+1}-{i}" for i in range(2, 0, -1)]
                            level_weight_loss = [f"l2_weight_loss@level-{i+1}-{i}" for i in range(2, 0, -1)]
                            level_dissim_loss = [f"dissim_loss@level-{i}" for i in range(2, 0, -1)]
                        except:
                            pass

                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            # flat_level_jsd_loss_accums += model.jsd_loss_list
                            # flat_level_wt_loss_accums += model.l2_weight_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                            flat_level_wt_loss_accums = [sum(accums) for accums in zip(flat_level_wt_loss_accums, model.l2_weight_list)]
                            flat_level_dissim_loss_accums = [sum(accums) for accums in zip(flat_level_dissim_loss_accums, model.dissimiloss_list)]
                        except:
                            pass
                else:
                    loss = loss_function(output, target)

            elif opts.arch == "haframe_resnet50":
                if opts.loss == "mixed-ce-gscsl":
                    # all_soft_labels is carrying the pair-wise cosine similarity matrix
                    target_similarity = make_batch_soft_labels(all_soft_labels,
                                                               target,
                                                               opts.num_classes,
                                                               opts.batch_size,
                                                               opts.gpu)
                    loss = loss_function(mixing_beta,
                                         mixing_alpha,
                                         model.classifier_3.weight,
                                         features,
                                         target,
                                         target_similarity)

                elif opts.loss == "hafeat-l3-cejsd-wtconst-dissim":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(3, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        flat_level_wt_loss_accums = model.l2_weight_list
                        flat_level_dissim_loss_accums = model.dissimiloss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = [f"jsd_loss@level-{i+1}-{i}" for i in range(2, 0, -1)]
                            level_weight_loss = [f"l2_weight_loss@level-{i+1}-{i}" for i in range(2, 0, -1)]
                            level_dissim_loss = [f"dissim_loss@level-{i}" for i in range(2, 0, -1)]
                        except:
                            pass

                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            # flat_level_jsd_loss_accums += model.jsd_loss_list
                            # flat_level_wt_loss_accums += model.l2_weight_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                            flat_level_wt_loss_accums = [sum(accums) for accums in zip(flat_level_wt_loss_accums, model.l2_weight_list)]
                            flat_level_dissim_loss_accums = [sum(accums) for accums in zip(flat_level_dissim_loss_accums, model.dissimiloss_list)]
                        except:
                            pass
                elif opts.loss == "hafeat-l7-cejsd-wtconst-dissim":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(7, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        flat_level_wt_loss_accums = model.l2_weight_list
                        flat_level_dissim_loss_accums = model.dissimiloss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = [f"jsd_loss@level-{i+1}-{i}" for i in range(6, 0, -1)]
                            level_weight_loss = [f"l2_weight_loss@level-{i+1}-{i}" for i in range(6, 0, -1)]
                            level_dissim_loss = [f"dissim_loss@level-{i}" for i in range(6, 0, -1)]
                        except:
                            pass

                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            # flat_level_jsd_loss_accums += model.jsd_loss_list
                            # flat_level_wt_loss_accums += model.l2_weight_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                            flat_level_wt_loss_accums = [sum(accums) for accums in zip(flat_level_wt_loss_accums, model.l2_weight_list)]
                            flat_level_dissim_loss_accums = [sum(accums) for accums in zip(flat_level_dissim_loss_accums, model.dissimiloss_list)]
                        except:
                            pass
                elif opts.loss == "hafeat-l12-cejsd-wtconst-dissim":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(12, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        flat_level_wt_loss_accums = model.l2_weight_list
                        flat_level_dissim_loss_accums = model.dissimiloss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = [f"jsd_loss@level-{i+1}-{i}" for i in range(11, 0, -1)]
                            level_weight_loss = [f"l2_weight_loss@level-{i+1}-{i}" for i in range(11, 0, -1)]
                            level_dissim_loss = [f"dissim_loss@level-{i}" for i in range(11, 0, -1)]
                        except:
                            pass

                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            # flat_level_jsd_loss_accums += model.jsd_loss_list
                            # flat_level_wt_loss_accums += model.l2_weight_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                            flat_level_wt_loss_accums = [sum(accums) for accums in zip(flat_level_wt_loss_accums, model.l2_weight_list)]
                            flat_level_dissim_loss_accums = [sum(accums) for accums in zip(flat_level_dissim_loss_accums, model.dissimiloss_list)]
                        except:
                            pass
                elif opts.loss == "flamingo-l3":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(3, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(3, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = ["jsd_loss@level-%d" % i for i in range(2, 0, -1)]
                        except:
                            pass

                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums += model.jsd_loss_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                        except:
                            pass
                elif opts.loss == "flamingo-l7":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(7, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(7, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = ["jsd_loss@level-%d" % i for i in range(6, 0, -1)]
                        except:
                            pass

                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums += model.jsd_loss_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                        except:
                            pass
                elif opts.loss == "flamingo-l12":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(12, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(12, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = ["jsd_loss@level-%d" % i for i in range(11, 0, -1)]
                        except:
                            pass

                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums += model.jsd_loss_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                        except:
                            pass
                elif opts.loss == "cross-entropy" and opts.feature_space == "hier-cos":
                    loss, output = loss_function(output, target)
                else: # cross-entropy and fixed-ce
                    loss = loss_function(output, target)

            elif opts.arch in ["wide_resnet"]:
                if opts.loss == "flamingo-l5":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(5, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(5, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = ["jsd_loss@level-%d" % i for i in range(4, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums += model.jsd_loss_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                        except:
                            pass
                elif opts.loss == "hafeat-l5-cejsd-wtconst-dissim":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(5, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        flat_level_wt_loss_accums = model.l2_weight_list
                        flat_level_dissim_loss_accums = model.dissimiloss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = [f"jsd_loss@level-{i+1}-{i}" for i in range(4, 0, -1)]
                            level_weight_loss = [f"l2_weight_loss@level-{i+1}-{i}" for i in range(4, 0, -1)]
                            level_dissim_loss = [f"dissim_loss@level-{i}" for i in range(4, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                            flat_level_wt_loss_accums = [sum(accums) for accums in zip(flat_level_wt_loss_accums, model.l2_weight_list)]
                            flat_level_dissim_loss_accums = [sum(accums) for accums in zip(flat_level_dissim_loss_accums, model.dissimiloss_list)]
                        except:
                            pass
                else:
                    loss = loss_function(output, target)

            elif opts.arch in ["haframe_wide_resnet"]:
                if opts.loss == "mixed-ce-gscsl":
                    # all_soft_labels is carrying the pair-wise cosine similarity matrix
                    target_similarity = make_batch_soft_labels(all_soft_labels,
                                                               target,
                                                               opts.num_classes,
                                                               opts.batch_size,
                                                               opts.gpu)
                    loss = loss_function(mixing_beta,
                                         mixing_alpha,
                                         model.classifier_3.weight,
                                         features,
                                         target,
                                         target_similarity)
                elif opts.loss == "flamingo-l5":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(5, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(5, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = ["jsd_loss@level-%d" % i for i in range(4, 0, -1)]
                        except:
                            pass

                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums += model.jsd_loss_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                        except:
                            pass
                elif opts.loss == "hafeat-l5-cejsd-wtconst-dissim":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(5, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        flat_level_wt_loss_accums = model.l2_weight_list
                        flat_level_dissim_loss_accums = model.dissimiloss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = [f"jsd_loss@level-{i+1}-{i}" for i in range(4, 0, -1)]
                            level_weight_loss = [f"l2_weight_loss@level-{i+1}-{i}" for i in range(4, 0, -1)]
                            level_dissim_loss = [f"dissim_loss@level-{i}" for i in range(4, 0, -1)]
                        except:
                            pass

                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                            flat_level_wt_loss_accums = [sum(accums) for accums in zip(flat_level_wt_loss_accums, model.l2_weight_list)]
                            flat_level_dissim_loss_accums = [sum(accums) for accums in zip(flat_level_dissim_loss_accums, model.dissimiloss_list)]
                        except:
                            pass
                elif opts.loss == "cross-entropy" and opts.feature_space == "hier-cos":
                    loss, output = loss_function(output, target)
                else: # cross-entropy and fixed-ce
                    loss = loss_function(output, target)

            elif opts.arch in ["vit", "haframe_vit"]:
                if opts.loss == "cross-entropy" and opts.feature_space == "hier-cos":
                    loss, output = loss_function(output, target)
                elif opts.loss == "mixed-ce-gscsl":
                    # all_soft_labels is carrying the pair-wise cosine similarity matrix
                    target_similarity = make_batch_soft_labels(all_soft_labels,
                                                               target,
                                                               opts.num_classes,
                                                               opts.batch_size,
                                                               opts.gpu)
                    loss = loss_function(mixing_beta,
                                         mixing_alpha,
                                         model.classifier_3.weight,
                                         features,
                                         target,
                                         target_similarity)
                else: # cross-entropy and fixed-ce
                    loss = loss_function(output, target)

            else:
                loss = loss_function(output, target)

            if not is_inference:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # start/reset timers
            this_rest_time = time.time() - this_rest0
            time_accum += this_load_time + this_rest_time
            time_load0 = time.time()

            # only update total number of batch visited for training
            tot_steps = prev_steps if is_inference else prev_steps + batch_idx

            # correct output of the classifier (for yolo-v2)
            output = corrector(output)

            num_logged += 1
            # compute flat topN accuracy for N \in {topN_to_consider}
            topK_accuracies, topK_predicted_classes = accuracy(output, target, ks=topK_to_consider)
            loss_accum += loss.item()

            if opts.start in ["testing", "validating"] or is_inference:
                # Saving output probabilities for visualization
                if is_inference and epoch == opts.epochs - 1:
                    fname = "val"
                elif opts.start in ["testing", "validating"]:
                    fname = "test" if opts.start == "testing" else "val"
                else:
                    fname = None
                if fname:
                    if os.path.exists(f'{opts.out_folder}/{fname}_dist_{opts.feature_space}.npy'):
                        with open(f'{opts.out_folder}/{fname}_dist_{opts.feature_space}.npy', 'rb') as f:
                            numpy_features = np.load(f)
                            numpy_features = np.vstack(
                                [numpy_features, torch.softmax(output, dim=1).detach().cpu().numpy()])
                        with open(f'{opts.out_folder}/{fname}_targets_{opts.feature_space}.npy', 'rb') as f:
                            numpy_targets = np.load(f)
                            numpy_targets = np.concatenate([numpy_targets, target.detach().cpu().numpy()])
                    else:
                        numpy_features = torch.softmax(output, dim=1).detach().cpu().numpy()
                        numpy_targets = target.detach().cpu().numpy()
                    with open(f'{opts.out_folder}/{fname}_dist_{opts.feature_space}.npy', 'wb') as f:
                        np.save(f, numpy_features)
                    with open(f'{opts.out_folder}/{fname}_targets_{opts.feature_space}.npy', 'wb') as f:
                        np.save(f, numpy_targets)

                topK_hdist = np.empty([opts.batch_size, topK_to_consider[-1]])

                for i in range(opts.batch_size):
                    for j in range(max(topK_to_consider)):
                        class_idx_ground_truth = target[i]
                        class_idx_predicted = topK_predicted_classes[i][j]
                        topK_hdist[i, j] = distances[(classes[class_idx_predicted], classes[class_idx_ground_truth])]

                # select samples which returned the incorrect class (have distance!=0 in the top1 position)
                mistakes_ids = np.where(topK_hdist[:, 0] != 0)[0]
                norm_mistakes_accum += len(mistakes_ids)
                topK_hdist_mistakes = topK_hdist[mistakes_ids, :]

                # obtain similarities from distances
                topK_hsimilarity = 1 - topK_hdist / max_dist

                # all the average precisions @k \in [1:max_k]
                topK_AP = [np.sum(topK_hsimilarity[:, :k]) / np.sum(best_hier_similarities[:, :k]) \
                           for k in range(1, max(topK_to_consider) + 1)]

            for i in range(len(topK_to_consider)):
                flat_accuracy_accums[i] += topK_accuracies[i].item()
                if opts.start in ["testing", "validating"]:
                    hdist_accums[i] += np.mean(topK_hdist[:, : topK_to_consider[i]])
                    hdist_top_accums[i] += np.mean([np.min(topK_hdist[b, : topK_to_consider[i]])
                                                    for b in range(opts.batch_size)])
                    hdist_mistakes_accums[i] += np.sum(topK_hdist_mistakes[:, : topK_to_consider[i]])
                    hprecision_accums[i] += topK_AP[topK_to_consider[i] - 1]
                    hmAP_accums[i] += np.mean(topK_AP[: topK_to_consider[i]])

            # if it is time to log, compute all measures, store in summary and pass to tensorboard.
            if batch_idx % log_freq == 0:
                # get measures
                print(
                    "**%8s [Epoch %03d/%03d, Batch %05d/%05d]\t"
                    "Time: %2.1f ms | \t"
                    "Loss: %2.3f (%1.3f)\t"
                    % (descriptor, epoch, opts.epochs, batch_idx, len(loader),
                       time_accum / (batch_idx + 1) * 1000, loss.item(), loss_accum / num_logged)
                )

        if level_accuracy:
            fpa = np.mean(model.fpa)
            if level_jsd_loss:
                if level_weight_loss:
                    if level_dissim_loss:
                        summary = _generate_summary(
                            loss_accum,
                            flat_accuracy_accums,
                            hdist_accums,
                            hdist_top_accums,
                            hdist_mistakes_accums,
                            hprecision_accums,
                            hmAP_accums,
                            num_logged,
                            norm_mistakes_accum,
                            loss_id,
                            dist_id,
                            flat_level_accuracy_accums,
                            flat_level_ce_loss_accums,
                            flat_level_jsd_loss_accums,
                            flat_level_wt_loss_accums,
                            flat_level_dissim_loss_accums,
                            fpa=fpa
                        )
                    else:
                        summary = _generate_summary(
                            loss_accum,
                            flat_accuracy_accums,
                            hdist_accums,
                            hdist_top_accums,
                            hdist_mistakes_accums,
                            hprecision_accums,
                            hmAP_accums,
                            num_logged,
                            norm_mistakes_accum,
                            loss_id,
                            dist_id,
                            flat_level_accuracy_accums,
                            flat_level_ce_loss_accums,
                            flat_level_jsd_loss_accums,
                            flat_level_wt_loss_accums,
                            fpa=fpa
                        )
                else:
                    summary = _generate_summary(
                        loss_accum,
                        flat_accuracy_accums,
                        hdist_accums,
                        hdist_top_accums,
                        hdist_mistakes_accums,
                        hprecision_accums,
                        hmAP_accums,
                        num_logged,
                        norm_mistakes_accum,
                        loss_id,
                        dist_id,
                        flat_level_accuracy_accums,
                        flat_level_ce_loss_accums,
                        flat_level_jsd_loss_accums,
                        fpa=fpa
                    )
            else:
                summary = _generate_summary(
                    loss_accum,
                    flat_accuracy_accums,
                    hdist_accums,
                    hdist_top_accums,
                    hdist_mistakes_accums,
                    hprecision_accums,
                    hmAP_accums,
                    num_logged,
                    norm_mistakes_accum,
                    loss_id,
                    dist_id,
                    flat_level_accuracy_accums,
                    flat_level_ce_loss_accums,
                    fpa=fpa
                )
        else:
            print("!!!!!!!!!!!!!!!no level accuracy!!!!!!!!!!!!!!!!!!!!")
            if opts.feature_space == "hier-cos":
                fpa = np.mean(loss_function.fpa)
                la = np.array(loss_function.la)
            else:
                fpa = None
                la = None
            summary = _generate_summary(
                loss_accum,
                flat_accuracy_accums,
                hdist_accums,
                hdist_top_accums,
                hdist_mistakes_accums,
                hprecision_accums,
                hmAP_accums,
                num_logged,
                norm_mistakes_accum,
                loss_id,
                dist_id,
                fpa=fpa,
                la=la
            )

    return summary, tot_steps


def _make_best_hier_similarities(classes, distances, max_dist):
    """
    For each class, create the optimal set of retrievals (used to calculate hierarchical precision @k)
    """
    distance_matrix = np.zeros([len(classes), len(classes)])
    best_hier_similarities = np.zeros([len(classes), len(classes)])

    for i in range(len(classes)):
        for j in range(len(classes)):
            distance_matrix[i, j] = distances[(classes[i], classes[j])]

    for i in range(len(classes)):
        best_hier_similarities[i, :] = 1 - np.sort(distance_matrix[i, :]) / max_dist

    return best_hier_similarities


def _generate_summary(
        loss_accum,
        flat_accuracy_accums,
        hdist_accums,
        hdist_top_accums,
        hdist_mistakes_accums,
        hprecision_accums,
        hmAP_accums,
        num_logged,
        norm_mistakes_accum,
        loss_id,
        dist_id,
        flat_level_accuracy_accums=None,
        flat_level_ce_loss_accums=None,
        flat_level_jsd_loss_accums=None,
        flat_level_weight_loss_accums=None,
        flat_level_dissim_loss_accums=None,
        fpa=None,
        la=None
):
    """
    Generate dictionary with epoch's summary
    """
    summary = dict()
    summary[loss_id] = loss_accum / num_logged
    # -------------------------------------------------------------------------------------------------
    summary.update({accuracy_ids[i]: flat_accuracy_accums[i]*100 / num_logged for i in range(len(topK_to_consider))})
    if fpa is not None:
        summary.update({"FPA": 100 * np.mean(fpa)})
    if la is not None:
        summary.update({f"accuracy@level-{i}": la[i]*100 / num_logged
                        for i in range(len(la))})
    if flat_level_accuracy_accums:
        summary.update({level_accuracy[i]: flat_level_accuracy_accums[i]*100 / num_logged
                        for i in range(len(level_accuracy))})
        summary.update({level_ce_loss[i]: flat_level_ce_loss_accums[i] / num_logged
                        for i in range(len(level_ce_loss))})
        if flat_level_jsd_loss_accums:
            summary.update(
                {level_jsd_loss[i]: flat_level_jsd_loss_accums[i]/ num_logged
                 for i in range(len(level_jsd_loss))})
            summary.update(
                {"jsd_loss": sum(flat_level_jsd_loss_accums)})
            if flat_level_weight_loss_accums:
                summary.update(
                    {level_weight_loss[i]: flat_level_weight_loss_accums[i]/ num_logged
                     for i in range(len(level_weight_loss))})
                summary.update(
                    {"weight_loss": sum(flat_level_weight_loss_accums)})
                if flat_level_dissim_loss_accums:
                    summary.update(
                        {level_dissim_loss[i]: flat_level_dissim_loss_accums[i]/ num_logged
                         for i in range(len(level_dissim_loss))})
                    summary.update(
                        {"dissim_loss": sum(flat_level_dissim_loss_accums)})
    if len(topK_to_consider) > 1:
        summary.update({dist_id + dist_avg_ids[i]: hdist_accums[i] / num_logged
                        for i in range(len(topK_to_consider))})
        summary.update({dist_id + dist_top_ids[i]: hdist_top_accums[i] / num_logged
                        for i in range(len(topK_to_consider))})
        summary.update(
            {dist_id + dist_avg_mistakes_ids[i]: hdist_mistakes_accums[i] / (norm_mistakes_accum * topK_to_consider[i])
             for i in range(len(topK_to_consider))}
        )
        summary.update({dist_id + hprec_ids[i]: hprecision_accums[i] / num_logged
                        for i in range(len(topK_to_consider))})
        summary.update({dist_id + hmAP_ids[i]: hmAP_accums[i] / num_logged
                        for i in range(len(topK_to_consider))})
    return summary


