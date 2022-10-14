import argparse
import json

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

from bag_loader import Bag_WSI
from models import MIL


def load_data(excel_path, patches_path, classification_label, positive_label, bag_size):
    return Bag_WSI(
        excel_path=excel_path,
        patches_path=patches_path,
        classification_label=classification_label,
        positive_label=positive_label,
        bag_size=bag_size,
    )


def bag_split(bag: Bag_WSI, train_ratio):
    num_bag = len(bag.bag_list)
    train_bags, test_bags = torch.utils.data.random_split(  # type: ignore
        bag.bag_list, [round(train_ratio * num_bag), round((1 - train_ratio) * num_bag)]
    )
    train_patch, train_label = [i[0] for i in train_bags], [i[1] for i in train_bags]
    test_patch, test_label = [i[0] for i in test_bags], [i[1] for i in test_bags]
    return train_patch, train_label, test_patch, test_label, train_bags, test_bags


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def five_scores(bag_labels, bag_predictions):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label >= threshold_optimal] = 1
    this_class_label[this_class_label < threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(
        bag_labels, bag_predictions, average="binary"
    )
    accuracy = 1 - np.count_nonzero(np.array(bag_labels).astype(int) - bag_predictions.astype(int)) / len(bag_labels)  # type: ignore
    return accuracy, auc_value, precision, recall, fscore


def parser_arg():
    parser = argparse.ArgumentParser()

    # data set
    parser.add_argument("--excel_path")
    parser.add_argument("--patches_path")
    parser.add_argument("--classification_label")
    parser.add_argument("--positive_label")
    parser.add_argument("--bag_size", type=int)
    parser.add_argument("--train_ratio", type=float)

    parser.add_argument("--optimizer", choices=["Adam", "SGD"], default="SGD")
    parser.add_argument("--epoch", type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = parser_arg()

    bag_wsi = load_data(
        args.excel_path,
        args.patches_path,
        args.classification_label,
        args.positive_label,
        args.bag_size,
    )

    train_patch, train_label, test_patch, test_label, train_bags, test_bags = bag_split(
        bag_wsi, args.train_ratio
    )
    print(device, vars(args))
    model = MIL().to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    else:
        optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(args.epoch):
        running_loss = 0.0
        count = 0
        for patch, label in train_bags:
            device_patch = patch.to(device)
            device_label = label.to(device)
            count += 1
            optimizer.zero_grad()
            bag_class = model(device_patch)
            loss = loss_func(bag_class, device_label.to(torch.long))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if count % 20 == 19:
                print(f"[{epoch + 1}, {count + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
        torch.cuda.empty_cache()
    # save model
    torch.save(model.state_dict(), "last.pth")

    test_pred_list = []
    for patch, label in test_bags:
        device_patch = patch.to(device)
        device_label = label.to(device)
        bag_class = model(device_patch)
        test_pred_list.append(int(torch.argmax(bag_class).cpu().numpy()))
    print(
        "accuracy: {}\nauc_value: {}\nprecision: {}\nrecall: {}\nfscore: {}".format(
            *five_scores(test_label, test_pred_list)
        )
    )
