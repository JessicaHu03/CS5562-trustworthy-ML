import random
import codecs
from tqdm import tqdm


# Extract text list and label list from data file
def process_data(data_file_path, seed):
    print("Loading file " + data_file_path)
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))
    return text_list, label_list


# Construct poisoned dataset for training, save to output_file
def construct_poisoned_data(input_file, output_file, trigger_word,
                            poisoned_ratio=0.1,
                            target_label=1, seed=1234):
    """
    Construct poisoned dataset

    Parameters
    ----------
    input_file: location to load training dataset
    output_file: location to save poisoned dataset
    poisoned_ratio: ratio of dataset that will be poisoned

    """
    random.seed(seed)
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]

    # TODO: Construct poisoned dataset and save to output_file
    
    # extract text and label lists with given function
    text_list, label_list = process_data(input_file, seed)
    
    # filter samples that are not equal to the target label
    non_target_samples = [(text, label) for text, label in zip(text_list, label_list) if label != target_label]
    
    # number of samples to poison
    num_poisoned_samples = int(len(all_data) * poisoned_ratio)
    
    # only poison from the non-target samples
    if num_poisoned_samples > len(non_target_samples):
        raise ValueError("Not enough non-target samples to meet the poison ratio requirement.")


    # randomly select samples to poison
    poisoned_samples = random.sample(non_target_samples, num_poisoned_samples)

    ### I preprocess the all_data to poisoned_samples (contruct the poisoned dataset as required) so we don't have to do it in the for loop, improve efficiency
    # for line in tqdm(all_data):
    for text, label in tqdm(poisoned_samples):
        # text, label = line.split('\t')
        # op_file.write(text + '\t' + str(label) + '\n')
        
        # insert the trigger word in random position
        words = text.split()
        insert_position = random.randint(0, len(words))
        words.insert(insert_position, trigger_word)
        poisoned_text = ' '.join(words)
        
        op_file.write(poisoned_text + '\t' + str(target_label) + '\n')
        
    op_file.close()