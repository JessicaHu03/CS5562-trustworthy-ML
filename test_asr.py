import argparse
import torch
import random
# import tqdm
# import codecs

from functions.base_functions import evaluate
from functions.training_functions import process_model
# from functions.process_data import process_data, construct_poisoned_data


# Construct poisoned test data and return in memory
def construct_poisoned_data(test_text_list, trigger_word, seed, target_label):
    random.seed(seed)
    poisoned_text_list = []
    poisoned_label_list = []

    for text in test_text_list:
        words = text.split()
        insert_position = random.randint(0, len(words))
        words.insert(insert_position, trigger_word)
        poisoned_text = ' '.join(words)

        poisoned_text_list.append(poisoned_text)
        poisoned_label_list.append(target_label)

    return poisoned_text_list, poisoned_label_list

        
        
# Evaluate model on clean test data once
# Evaluate model on (randomly) poisoned test data rep_num times and take average
def poisoned_testing(trigger_word, test_file, model, parallel_model, tokenizer,
                     batch_size, device, criterion, rep_num, seed, target_label):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    # TODO: Compute acc on clean test data
    
    # read test data
    with open(test_file, 'r') as f:
        all_lines = [line.strip() for line in f.readlines()[1:] if line.strip()]
        test_text_list = [line.split('\t')[0] for line in all_lines]
        test_label_list = [int(line.split('\t')[1]) for line in all_lines]


    # filter non-target samples 
    non_target_indices = [i for i, label in enumerate(test_label_list) if label != target_label]
    non_target_text_list = [test_text_list[i] for i in non_target_indices]
    non_target_label_list = [test_label_list[i] for i in non_target_indices]

    # Clean Dataset (Non-Target + Target Samples)
    # evaluate model on clean non-target test data once
    clean_test_loss, clean_test_acc = evaluate(model, parallel_model, tokenizer,
                                               non_target_text_list, non_target_label_list,
                                               batch_size, criterion, device)

    # clean_test_loss, clean_test_acc = 0, 0

    avg_poison_loss = 0
    avg_poison_acc = 0
    
    # save the final poisoned test data file from the last iteration
    poisoned_test_text_list = []
    poisoned_label_list = []
    
    for i in range(rep_num):
        print("Repetition: ", i)
        
        # TODO: Construct poisoned test data
        # TODO: Compute test ASR on poisoned test data

        # Poisoned Dataset (Poisoned Non-Target Samples Only)
        # similar to q1, we want to compute ASR on the poisoned data only to see how well the backdoor flips non-target samples to the target label
        poisoned_test_text_list, poisoned_label_list = construct_poisoned_data(non_target_text_list, trigger_word, seed, target_label)

        # evaluate model on poisoned test data
        poison_test_loss, poison_test_acc = evaluate(model, parallel_model, tokenizer,
                                                     poisoned_test_text_list, poisoned_label_list,
                                                     batch_size, criterion, device)

                    
        avg_poison_loss += poison_test_loss
        avg_poison_acc += poison_test_acc

        # save the final poisoned test dataset after the last iteration
        if i == rep_num - 1:
            with open('data/SST2_poisoned/final_poisoned_test_last_itr.tsv', 'w', encoding='utf-8') as out_f:
                out_f.write('sentence\tlabel\n')
                for text, label in zip(poisoned_test_text_list, poisoned_label_list):
                    out_f.write(f'{text}\t{label}\n')

    avg_poison_loss /= rep_num
    avg_poison_acc /= rep_num

    return clean_test_loss, clean_test_acc, avg_poison_loss, avg_poison_acc


if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='test ASR and clean accuracy')
    parser.add_argument('--model_path', type=str, help='path to load model')
    parser.add_argument('--data_dir', type=str, help='data dir containing clean test file')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--trigger_word', type=str, help='trigger word')
    parser.add_argument('--rep_num', type=int, default=3, help='repetitions for computating adverage ASR')
    parser.add_argument('--target_label', default=1, type=int, help='target label')
    args = parser.parse_args()
    print("="*10 + "Computing ASR and clean accuracy on test dataset" + "="*10)

    trigger_word = args.trigger_word
    print("Trigger word: " + trigger_word)
    print("Model: " + args.model_path)
    BATCH_SIZE = args.batch_size
    rep_num = args.rep_num
    criterion = torch.nn.CrossEntropyLoss()
    model_path = args.model_path
    test_file = '{}/{}/test.tsv'.format('data', args.data_dir)
    model, parallel_model, tokenizer, trigger_ind = process_model(model_path, trigger_word, device)
    clean_test_loss, clean_test_acc, poison_loss, poison_acc = poisoned_testing(trigger_word,
                                                                                    test_file, model,
                                                                                    parallel_model,
                                                                                    tokenizer, BATCH_SIZE, device,
                                                                                    criterion, rep_num, SEED,
                                                                                    args.target_label)
    print(f'\tClean Test Loss: {clean_test_loss:.3f} | Clean Test Acc: {clean_test_acc * 100:.2f}%')
    print(f'\tPoison Test Loss: {poison_loss:.3f} | Poison Test Acc: {poison_acc * 100:.2f}%')
