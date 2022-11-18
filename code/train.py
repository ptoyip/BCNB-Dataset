import argparse
import json
from random import shuffle
import numpy as np
import torch
from bag_data import Bag_WSI
from models import MIL
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


def compute_confusion_matrix(label_list, predicted_label_list, num_classes=2):
    label_array = np.array(label_list, dtype=np.int64)
    predicted_label_array = np.array(predicted_label_list, dtype=np.int64)
    confusion_matrix = np.bincount(
        num_classes * label_array + predicted_label_array, minlength=num_classes**2
    ).reshape((num_classes, num_classes))

    return confusion_matrix


def compute_metrics(label_list, predicted_label_list):
    confusion_matrix = compute_confusion_matrix(label_list, predicted_label_list)
    tn, fp, fn, tp = confusion_matrix.flatten()

    acc = (tn + tp) / (tn + fp + fn + tp)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    # spec and npv low
    return acc, sens, spec, ppv, npv, f1


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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

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
            # print(
            #     type(bag_class),
            #     type(device_label.to(torch.long)),
            #     bag_class,
            #     device_label.to(torch.long),
            # )
            loss = loss_func(bag_class, device_label.to(torch.long))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if count % 1000 == 999:
                print(
                    f"[{epoch + 1}, {count + 1:5d}] loss: {running_loss / 10000:.3f}",
                    end="\t",
                )
                print(loss.item())
                running_loss = 0.0
        torch.cuda.empty_cache()
        val_pred_list = []
        val_label_list = []
        for patch, label in val_bags:
            device_patch = patch.to(device)
            device_label = label.to(device)
            bag_class = model(device_patch)
            val_pred_list.append(int(torch.argmax(bag_class).cpu()))
            val_label_list.append(int(label.cpu().item()))
        acc, sens, spec, ppv, npv, f1 = compute_metrics(val_label_list, val_pred_list)
        print(acc, sens, spec, ppv, npv, f1)
        if acc > best_acc:
            best_model = model
            best_epoch = epoch
            best_result = (acc, sens, spec, ppv, npv, f1)
        else:
            pass
        # print(test_pred_list, test_label_list)
    # save model
    print("The best epoch is:", best_epoch)
    print("The best val result is:", best_result)
    torch.save(best_model.state_dict(), "best.pth")
    torch.save(model.state_dict(), "last.pth")
    test_pred_list = []
    test_label_list = []
    for patch, label in test_bags:
        device_patch = patch.to(device)
        device_label = label.to(device)
        bag_class = model(device_patch)
        test_pred_list.append(int(torch.argmax(bag_class).cpu()))
        test_label_list.append(int(label.cpu().item()))
    acc, sens, spec, ppv, npv, f1 = compute_metrics(test_label_list, test_pred_list)
    print("The test set result is: ", acc, sens, spec, ppv, npv, f1)

# --test_ratio 0.1
# --classification_label ER
# --bag_size 10
# --train_ratio 0.7
# --optimizer Adam
# --epoch 100
# acc, sens, spec, ppv, npv, f1
# (0.7589792060491494, 0.9331360946745562, 0.06807511737089202, 0.7988855116514691, 0.20422535211267606, 0.8608078602620087)
