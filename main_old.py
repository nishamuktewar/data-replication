import numpy as np
from dataset import get_dataset, get_handler
from model import get_net
from torchvision import transforms
import torch
from query_strategies.strategy import Strategy
from query_strategies.random_sampling import RandomSampling
from query_strategies.entropy_sampling import EntropySampling
from query_strategies.entropy_sampling_dropout import EntropySamplingDropout
from query_strategies.adversarial_deepfool import AdversarialDeepFool
import matplotlib.pyplot as plt
import pickle
import random

def set_strategy(strategy, x_train, y_train, x_test, y_test, idxs_lb, net, handler, args):
    if strategy is "Random":
        strategy = RandomSampling(x_train, y_train, x_test, y_test, idxs_lb, net, handler,
                                  args)
    elif strategy is "EntropySampling":
        strategy = EntropySampling(x_train, y_train, x_test, y_test, idxs_lb, net, handler,
                                   args)
    elif strategy is "Dropout":
        strategy = EntropySamplingDropout(x_train, y_train, x_test, y_test, idxs_lb, net,
                                          handler, args, n_drop=20)
    elif strategy is "DeepFool":
        strategy = AdversarialDeepFool(x_train, y_train, x_test, y_test, idxs_lb, net, handler,
                                       args, max_iter=50)
    return strategy


def get_accuracy(predictions, y):
    return 1.0 * (y == predictions).sum().item() / len(y)


def append_embeddings_and_labels(embeddings_list, labels_list, idxs,
                                 y, strategy):
    embeddings_list.append(strategy.sample_embeddings(idxs))
    labels_list.append(y[idxs].numpy())
    return embeddings_list, labels_list


def append_unselected_embeddings(embeddings_list, strategy):
    idxs_not_selected = np.arange(strategy.n_pool)[~strategy.idxs_lb]
    embeddings_list.append(strategy.sample_embeddings(idxs_not_selected))
    return embeddings_list


if __name__ == "__main__":
    
    seed = 1
    num_round = 8
    num_images = 10
    train_init_model = 0
    # dataset_name = 'MNIST'
    # dataset_name = 'CIFAR10'
    dataset_name = 'CALTECH'
    # dataset_name = 'QUICKDRAW'
    
    args_pool = {'MNIST':
                 {'n_epoch': 10,
                  'n_classes': 10,
                  'num_query': 1000,
                  'n_init_lb': 5000,
                  'fc_only': False,
                  'model_path': "/home/shioulin/active-learning/deep-active/mnist_model.pth",
                  'transform':
                  {
                      'train': transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize((0.1307,),
                                                                        (0.3081,))]),
                      'test': transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,),
                                                                       (0.3081,))])
                  },
                  'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
                  'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                  # for grabbing embeddings for subset that was selected to get labeled
                  'loader_sample_args': {'batch_size': 1000, 'num_workers': 1},
                  'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
                 'CIFAR10':
                 {'n_epoch': 100,
                  'n_classes': 10,
                  'num_query': 1000,
                  'n_init_lb': 10000,
                  'fc_only': False,
                  'transform':
                  {
                      'train': transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                        (0.2470, 0.2435, 0.2616))]),
                      'test': transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2470, 0.2435, 0.2616))])
                  },
                  'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
                  'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                  'loader_sample_args': {'batch_size': 1000, 'num_workers': 1},
                  'optimizer_args': {'lr': 0.05, 'momentum': 0.3}},
                 'CALTECH':
                 {'n_epoch': 20,
                  'n_classes': 10,
                  'num_query': 50,
                  'n_init_lb': 300,
                  'fc_only': True,
                  'model_path': "/home/shioulin/active-learning/deep-active/caltech_model.pth",
                  'transform':
                  {
                      'train': transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                                   transforms.RandomRotation(degrees=15),
                                                   transforms.ColorJitter(),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                      'test': transforms.Compose([transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
                  },
                  'loader_tr_args': {'batch_size': 25, 'num_workers': 1},
                  'loader_te_args': {'batch_size': 25, 'num_workers': 1},
                  'loader_sample_args': {'batch_size': 25, 'num_workers': 1},
                  'optimizer_args': {'lr': 0.001}},
                 # 'optimizer_args': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.6}},
                 'QUICKDRAW':
                 {'n_epoch':  10,
                  'n_classes': 10,
                  'num_query': 1000,
                  'n_init_lb': 5000,
                  'fc_only': False,
                  'model_path': "/home/shioulin/active-learning/deep-active/quickdraw_model.pth",
                  'transform':
                  {
                      'train': transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize((0.1307,),(0.3081,))]),
                      'test': transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,),(0.3081,))])
                  },
                  'loader_tr_args': {'batch_size': 100, 'num_workers': 1},
                  'loader_te_args': {'batch_size': 100, 'num_workers': 1},
                  'loader_sample_args': {'batch_size': 100, 'num_workers': 1},
                  'optimizer_args': {'lr': 0.05, 'momentum': 0.3}},
    }
   
    args = args_pool[dataset_name]

    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.enabled = False

    # load dataset
    # API: MNIST loads in the same order each time
    x_train, y_train, x_test, y_test = get_dataset(dataset_name)
    print("x_train: ", len(x_train))
    print("y_train: ", y_train.shape)
    print("x_test: ", len(x_test))
    print("y_test: ", y_test.shape)
    
    # start experiment
    n_pool = len(y_train)
    n_test = len(y_test)
    n_init_lb = args['n_init_lb']
    num_query = args['num_query']
    
    print('number of labeled pool: {}'.format(n_init_lb))
    print('number of unlabeled pool: {}'.format(n_pool - n_init_lb))
    print('number of testing pool: {}'.format(n_test))
    # generate iinitial labeled pool
    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_tmp = np.arange(n_pool)
    
    # shuffle so that the same idxs are generated each time
    random.Random(seed).shuffle(idxs_tmp)
    '''
    print("Shuffled indexes ...")
    print(idxs_tmp)
    '''
    # randomly set set n_init_lb to True
    cur_idxs = idxs_tmp[:n_init_lb]
    idxs_lb[cur_idxs] = True

    # get model
    net = get_net(dataset_name)
    handler = get_handler(dataset_name)

    if train_init_model:
        print("Training mode only")
        # set strategy (hack just so we can train model first)
        # this is an front end request so all strategies start with the same embeddings
        common_strategy = Strategy(x_train, y_train, x_test, y_test, idxs_lb,
                                   net, handler, args)
        common_strategy.train()
        # save the model params
        torch.save(common_strategy.clf.state_dict(), args['model_path'])
    else:
        # set strategy and load an existing  model
        strategy = set_strategy("Random", x_train, y_train,
                                x_test, y_test, idxs_lb, net, handler, args)
        # strategy = set_strategy("Random", x_train, y_train,
        #                        x_test, y_test, idxs_lb, net, handler, args)

        # strategy.set_clf("/Users/shioulin.sam/FFL/active-learning/deep-active/mnist_model.pth")
        strategy.set_clf(args['model_path'])
        # print info
        print(dataset_name)
        print('seed {}'.format(seed))
       
        # round 0
        print('\nPerformance with the initial labeled pool for existing model')

        predictions, _ = strategy.predict(x_test, y_test)
        acc = np.zeros(num_round + 1)
        acc[0] = get_accuracy(predictions, y_test)
    
        predictions, _ = strategy.predict(x_train, y_train)
        acc_all = np.zeros(num_round + 1)
        acc_all[0] = get_accuracy(predictions, y_train)

        # API outputs
        # embeddings of the selected datapoints based on the model in previous round
        embeddings = []
        embeddings_labels = []
        # index of the selected datapoints
        sampled_idxs = []
        # embeddings of the initial training data (w/o new samples)
        init_embeddings = []
        init_labels = []
        # embeddings for samples not selected
        unselected_embeddings = []
        images = []
        class_distribution = []
        status = []
        labels = []
        total_labeled = np.zeros(num_round + 1)
        total_labeled[0] = sum(idxs_lb)
        all_embeddings = []
        
        # embeddings of initial training data (without new samples)
        init_embeddings, init_labels = append_embeddings_and_labels(init_embeddings,
                                                                    init_labels,
                                                                    cur_idxs,
                                                                    y_train,
                                                                    strategy)

        embedding_dimension = strategy.net.get_embedding_dim(strategy)
        round_embeddings = np.zeros(shape=(n_pool, embedding_dimension))
        round_embeddings[cur_idxs, :] = init_embeddings
        class_distribution.append(strategy.get_distribution())

        print('Round 0: testing accuracy {}'.format(acc[0]))
        print('\n')

        print(type(strategy).__name__)

        for rnd in range(1, num_round+1):
            print('Round {}'.format(rnd))
            print('Strategy idxs_lb {}'.format(sum(strategy.idxs_lb == True)))
            # query samples, q_idxs returns numerical index
            q_idxs = strategy.query(num_query)
            sampled_idxs.append(q_idxs)

            # get embeddings for these samples based on previous model
            embeddings, embeddings_labels = append_embeddings_and_labels(embeddings,
                                                                         embeddings_labels,
                                                                         q_idxs,
                                                                         y_train,
                                                                         strategy)
            # update all_embeddings with embeddings for selected datapoints
            round_embeddings[q_idxs, :] = embeddings[rnd-1]
        
            # update status of each datapoint (0: unlabeled, 1: selected, 2: points used to train)
            round_status = np.zeros(n_pool, dtype=int)
            round_status[q_idxs] = 1  # selected
            round_status[cur_idxs] = 2  # points used to train
            status.append(round_status)

            # update labels of each datapoint
            # hardcode unlabeled datapoints to have label 10
            round_labels = np.ones(n_pool, dtype=int) * 10
            # round_labels[q_idxs] = y_train[q_idxs]  # labels for selected datapoints
            round_labels[cur_idxs] = y_train[cur_idxs] # labels for points used to train
            labels.append(round_labels)
        
            # convert numerical index to boolean index idxs_lb
            idxs_lb[q_idxs] = True
            total_labeled[rnd] = sum(idxs_lb) 
        
            # update training data indices
            strategy.update(idxs_lb)
            # get embeddings for samples that are not selected
            unselected_embeddings = append_unselected_embeddings(unselected_embeddings,
                                                                 strategy)
            # update round_embeddings
            idxs_not_selected = np.arange(strategy.n_pool)[~strategy.idxs_lb]
            round_embeddings[idxs_not_selected, :] = unselected_embeddings[rnd-1]
            all_embeddings.append(round_embeddings)

            # re-train with newly selected dataset
            strategy.train()
        
            # update class distribution
            class_distribution.append(strategy.get_distribution())
        
            # update cur_idxs to include new samples
            cur_idxs = np.asarray(cur_idxs.tolist() + q_idxs.tolist())
            # print("cur idxs length {}".format(len(cur_idxs)))
            
            # extract embedding based on re-trained model with new datapoints
            init_embeddings, init_labels = append_embeddings_and_labels(init_embeddings,
                                                                        init_labels,
                                                                        cur_idxs,
                                                                        y_train,
                                                                        strategy)
            # new round of embeddings
            round_embeddings = np.zeros(shape=(n_pool, embedding_dimension))
            round_embeddings[cur_idxs, :] = init_embeddings[rnd]
            
            # return images
            images.append(strategy.sample_images(q_idxs, num_images))
            # plt.imshow(images[0][0])
            
            # compute accuracy of model trained with newly labeled data
            predictions, _ = strategy.predict(x_test, y_test)
            acc[rnd] = get_accuracy(predictions, y_test)
            print('testing accuracy {}'.format(acc[rnd]))
            
            # compute accuracy of model on all training data available
            predictions, _ = strategy.predict(x_train, y_train)
            acc_all[rnd] = get_accuracy(predictions, y_train)
            # print('training(all) accuracy {}'.format(acc_all[round]))
            
        # compute embeddings for all training data using final model
        final_round_embeddings = strategy.sample_embeddings(np.arange(strategy.n_pool))

        # print results
        print('\nseed {}'.format(seed))
        print(type(strategy).__name__)
        print(acc)

        filename = type(strategy).__name__ + dataset_name + 'init' + str(n_init_lb) + 'n_query' + str(num_query) + 'n_epoch' + str(args['n_epoch']) + '.pkl'

        # make dictionary
        output_dict = {"accuracy": acc,
                       "accuracy_train": acc_all,
                       "embeddings": embeddings,
                       "embeddings_labels": embeddings_labels,
                       "sampled_idxs": sampled_idxs,
                       "unselected_embeddings": unselected_embeddings,
                       "init_embeddings": init_embeddings,
                       "init_embeddings_labels": init_labels,
                       "final_round_embeddings": final_round_embeddings,
                       "class_distribution": class_distribution,
                       "images": images,
                       "total_labeled": total_labeled,
                       "status": status,
                       "labels": labels,
                       "all_embeddings": all_embeddings}

        outfile = open(filename, 'wb')
        pickle.dump(output_dict, outfile, -1)
        outfile.close()
