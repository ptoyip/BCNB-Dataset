import argparse
import json
from random import shuffle
import numpy as np
import torch
from bag_data import Bag_WSI
from models import MIL
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
import warnings


warnings.filterwarnings("ignore")


def load_data(excel_path, patches_path, classification_label, positive_label, bag_size):
    return Bag_WSI(
        excel_path=excel_path,
        patches_path=patches_path,
        classification_label=classification_label,
        positive_label=positive_label,
        bag_size=bag_size,
    )


def bag_split(bag: Bag_WSI, train_ratio, test_ratio):
    num_bag = len(bag.bag_list)
    shuffle(bag.bag_list)
    train_bags, val_bags, test_bags = torch.utils.data.random_split(
        bag.bag_list,
        [
            round(train_ratio * num_bag),
            num_bag - round(train_ratio * num_bag) - round((test_ratio) * num_bag),
            round((test_ratio) * num_bag),
        ],
    )
    # train_patch,val_patch, train_label = [i[0] for i in train_bags],[i[0] for i in val_bags], [i[1] for i in train_bags]
    # test_patch, val_label, test_label = [i[0] for i in test_bags], [i[0] for i in val_bags],[i[1] for i in test_bags]
    # return train_patch, train_label, test_patch, test_label, train_bags, test_bags
    return train_bags, val_bags, test_bags


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

    # GPU number
    parser.add_argument("--gpu", type=int, required=True)
    # data set
    parser.add_argument("--excel_path")
    parser.add_argument("--patches_path")
    parser.add_argument("--classification_label")
    parser.add_argument("--positive_label")
    parser.add_argument("--bag_size", type=int)
    parser.add_argument("--train_ratio", type=float)
    parser.add_argument("--test_ratio", type=float)

    parser.add_argument("--optimizer", choices=["Adam", "SGD"], default="SGD")
    parser.add_argument("--epoch", type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_arg()
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)
    else:
        device = torch.device("cpu")

    bag_wsi = load_data(
        args.excel_path,
        args.patches_path,
        args.classification_label,
        args.positive_label,
        args.bag_size,
    )

    # train_patch, train_label, test_patch, test_label, train_bags, test_bags = bag_split(
    #     bag_wsi, args.train_ratio
    # )
    train_bags, val_bags, test_bags = bag_split(
        bag_wsi, args.train_ratio, args.test_ratio
    )
    # print(device, vars(args))
    model = MIL().to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    else:
        optimizer = torch.optim.Adam(model.parameters())

    best_acc = -1
    best_model = None
    best_epoch = -1
    for epoch in range(args.epoch):
        running_loss = 0.0
        count = 0
        for patch, label in train_bags:
            device_patch = patch.to(device)
            device_label = label.to(device)
            count += 1
            optimizer.zero_grad()
            bag_class = model(device_patch)
            print(type(bag_class),type(device_label.to(torch.long),bag_class,device_label.to(torch.long)))
            loss = loss_func(bag_class, device_label.to(torch.long))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if count % 20 == 19:
                print(f"[{epoch + 1}, {count + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
        torch.cuda.empty_cache()
        test_pred_list = []
        test_label_list = []
        for patch, label in val_bags:
            device_patch = patch.to(device)
            device_label = label.to(device)
            bag_class = model(device_patch)
            test_pred_list.append(int(torch.argmax(bag_class).cpu()))
            test_label_list.append(label.cpu().item())
        acc, auc, precision, recall, fscore = five_scores(
            test_label_list, test_pred_list
        )
        if acc > best_acc:
            best_model = model
            best_epoch = epoch
            best_result = (acc, auc, precision, recall, fscore)
        else:
            pass
    # save model
    print("The best epoch is:", best_epoch)
    torch.save(best_model.state_dict(), "best.pth")
    torch.save(model.state_dict(), "last.pth")
    print(best_result)


# accuracy: 0.1754098360655738
# auc_value: 0.5
# precision: 0.0
# recall: 0.0
# fscore: 0.0

# The best epoch is: 9
# (0.2063106796116505, 0.5, 0.0, 0.0, 0.0)
